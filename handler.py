"""RunPod queue worker for one VectorLab conversion job per invocation."""

import runpod


async def process_queued_job(input: dict) -> dict:
    """Claim the oldest queued job, convert it, store the SVG, and finalize it."""
    import hashlib
    import io
    import math
    import os
    import re
    import time
    import xml.etree.ElementTree as ET
    from urllib.parse import quote

    import requests
    import torch
    from PIL import Image, UnidentifiedImageError

    original_bucket = "vectorlab-originals"
    result_bucket = "vectorlab-results"
    max_input_bytes = 10 * 1024 * 1024
    max_output_bytes = 25 * 1024 * 1024
    max_image_pixels = 25_000_000
    max_generation_attempts = 3
    valid_modes = {"balanced", "high_detail"}
    starvector_max_length = 4000
    model_id = os.getenv("STARVECTOR_MODEL_ID", "starvector/starvector-1b-im2svg")
    model_revision = os.getenv(
        "STARVECTOR_MODEL_REVISION",
        "380ab95d25a8e9ab1dc825debe238b4953ae13b9",
    )
    worker_name = os.getenv("VECTORLAB_WORKER_NAME", "vectorlab-runpod-serverless")
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_secret_key = os.getenv("SUPABASE_SECRET_KEY", "")
    gpu_hourly_usd = float(os.getenv("STARVECTOR_GPU_HOURLY_USD", "0.69"))

    if not isinstance(input, dict) or not input:
        raise ValueError("invalid_trigger")
    if not supabase_url or not supabase_secret_key:
        raise RuntimeError("server_configuration_error")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_required")

    request_headers = {
        "apikey": supabase_secret_key,
        "Authorization": f"Bearer {supabase_secret_key}",
    }

    def rpc(name: str, payload: dict):
        response = requests.post(
            f"{supabase_url}/rest/v1/rpc/{name}",
            headers={**request_headers, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()

    def storage_download(bucket: str, path: str) -> bytes:
        encoded_path = quote(path, safe="/")
        response = requests.get(
            f"{supabase_url}/storage/v1/object/authenticated/{bucket}/{encoded_path}",
            headers=request_headers,
            timeout=60,
        )
        response.raise_for_status()
        if not response.content or len(response.content) > max_input_bytes:
            raise ValueError("input_download_failed")
        return response.content

    def storage_upload(bucket: str, path: str, payload: bytes) -> None:
        encoded_path = quote(path, safe="/")
        response = requests.post(
            f"{supabase_url}/storage/v1/object/{bucket}/{encoded_path}",
            headers={
                **request_headers,
                "Content-Type": "image/svg+xml",
                "cache-control": "3600",
                "x-upsert": "false",
            },
            data=payload,
            timeout=60,
        )
        response.raise_for_status()

    def storage_remove(bucket: str, path: str) -> None:
        requests.delete(
            f"{supabase_url}/storage/v1/object/{bucket}",
            headers={**request_headers, "Content-Type": "application/json"},
            json={"prefixes": [path]},
            timeout=30,
        ).raise_for_status()

    def open_image(payload: bytes):
        try:
            Image.MAX_IMAGE_PIXELS = max_image_pixels
            probe = Image.open(io.BytesIO(payload))
            probe.verify()
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError("invalid_image") from error
        if image.width * image.height > max_image_pixels:
            raise ValueError("image_dimensions_limit")
        return image

    def canonical_png(image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    def validate_svg(raw_svg: str) -> dict:
        if not isinstance(raw_svg, str):
            raise ValueError("svg_not_text")
        encoded = raw_svg.encode("utf-8")
        if not encoded or len(encoded) > max_output_bytes:
            raise ValueError("svg_size_limit")

        lowered = raw_svg.lower()
        blocked_fragments = (
            "<!doctype",
            "<!entity",
            "<script",
            "<foreignobject",
            "<iframe",
            "<object",
            "<embed",
            "javascript:",
            "data:text/html",
        )
        if any(fragment in lowered for fragment in blocked_fragments):
            raise ValueError("svg_unsafe_content")

        try:
            root = ET.fromstring(raw_svg)
        except ET.ParseError as error:
            raise ValueError("svg_invalid_root") from error
        if root.tag.split("}")[-1].lower() != "svg":
            raise ValueError("svg_invalid_root")

        path_count = 0
        point_count = 0
        for element in root.iter():
            tag = element.tag.split("}")[-1].lower()
            for attribute, value in element.attrib.items():
                name = attribute.split("}")[-1].lower()
                compact = value.strip().lower()
                if name.startswith("on"):
                    raise ValueError("svg_unsafe_content")
                if name in {"href", "xlink:href"} and (
                    compact.startswith(("http:", "https:", "//", "javascript:", "data:"))
                ):
                    raise ValueError("svg_unsafe_content")
            if tag == "path":
                path_count += 1
                numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", element.attrib.get("d", ""))
                point_count += math.ceil(len(numbers) / 2)

        return {
            "svg": raw_svg,
            "bytes": len(encoded),
            "metrics": {"pathCount": path_count, "pointCount": point_count},
        }

    def convert_vtracer(payload: bytes, image, mode: str) -> dict:
        import vtracer

        config = vtracer.Config.poster()
        config.filter_speckle = 32
        config.simplify = 1.0 if mode == "high_detail" else 2.0
        config.optimize = 1
        raw_svg = config.convert_bytes(canonical_png(image))
        return validate_svg(raw_svg)

    def convert_starvector(payload: bytes, image, mode: str) -> dict:
        from starvector.data.util import process_and_rasterize_svg
        from transformers import AutoModelForCausalLM

        global _VECTORLAB_STARVECTOR_MODEL
        try:
            model = _VECTORLAB_STARVECTOR_MODEL
        except NameError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=model_revision,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            model.cuda()
            model.eval()
            _VECTORLAB_STARVECTOR_MODEL = model

        processor = model.model.processor
        pixel_values = processor(image, return_tensors="pt")["pixel_values"]
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
            raise RuntimeError("unexpected_image_tensor_shape")
        pixel_values = pixel_values.cuda()
        base_seed = int.from_bytes(
            hashlib.sha256(payload + mode.encode("utf-8")).digest()[:4],
            "big",
        )

        svg = "<svg></svg>"
        with torch.inference_mode():
            for attempt in range(max_generation_attempts):
                seed = base_seed + attempt
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                raw_svg = model.generate_im2svg(
                    {"image": pixel_values},
                    max_length=starvector_max_length,
                )[0]
                svg, _ = process_and_rasterize_svg(raw_svg)
                print(
                    "starvector_generation "
                    f"attempt={attempt + 1} "
                    f"raw_chars={len(raw_svg) if isinstance(raw_svg, str) else -1} "
                    f"has_svg_close={isinstance(raw_svg, str) and '</svg>' in raw_svg.lower()} "
                    f"postprocessed_empty={svg.strip() in {'<svg></svg>', '<svg/>', '<svg />'}}"
                )
                if svg.strip() not in {"<svg></svg>", "<svg/>", "<svg />"}:
                    break
        if svg.strip() in {"<svg></svg>", "<svg/>", "<svg />"}:
            raise ValueError("generation_failed")
        return validate_svg(svg)

    def safe_error_code(error: Exception) -> str:
        known = {
            "invalid_image",
            "image_dimensions_limit",
            "invalid_mode",
            "svg_not_text",
            "svg_size_limit",
            "svg_invalid_root",
            "svg_unsafe_content",
            "generation_failed",
            "input_download_failed",
            "output_upload_failed",
            "job_completion_failed",
        }
        message = str(error)
        return message if message in known else "conversion_failed"

    claim_rows = rpc("claim_conversion_job", {"p_worker": worker_name}) or []
    if not claim_rows:
        return {"status": "idle", "claimed": False}

    job = claim_rows[0] if isinstance(claim_rows, list) else claim_rows
    job_id = job["job_id"]
    attempt_id = job["attempt_id"]
    output_path = job["output_storage_path"]
    started_at = time.perf_counter()
    uploaded = False

    try:
        if job["mode"] not in valid_modes:
            raise ValueError("invalid_mode")
        input_payload = storage_download(original_bucket, job["input_storage_path"])
        image = open_image(input_payload)
        converted = (
            convert_starvector(input_payload, image, job["mode"])
            if job["engine"] == "starvector"
            else convert_vtracer(input_payload, image, job["mode"])
        )
        duration_ms = round((time.perf_counter() - started_at) * 1000)
        estimated_cost_usd = round(gpu_hourly_usd * duration_ms / 3_600_000, 6)
        storage_upload(result_bucket, output_path, converted["svg"].encode("utf-8"))
        uploaded = True

        completed = rpc(
            "complete_conversion_job",
            {
                "p_job_id": job_id,
                "p_attempt_id": attempt_id,
                "p_output_storage_path": output_path,
                "p_output_bytes": converted["bytes"],
                "p_width": image.width,
                "p_height": image.height,
                "p_duration_ms": duration_ms,
                "p_estimated_cost_usd": estimated_cost_usd,
                "p_result_metrics": {
                    **converted["metrics"],
                    "mode": job["mode"],
                    "engine": job["engine"],
                },
            },
        )
        if completed is not True:
            raise RuntimeError("job_completion_failed")
        return {
            "status": "succeeded",
            "claimed": True,
            "job_id": job_id,
            "duration_ms": duration_ms,
        }
    except Exception as error:
        duration_ms = round((time.perf_counter() - started_at) * 1000)
        if uploaded:
            try:
                storage_remove(result_bucket, output_path)
            except Exception:
                pass
        code = safe_error_code(error)
        is_retryable = isinstance(error, requests.RequestException) or code in {
            "input_download_failed",
            "output_upload_failed",
            "job_completion_failed",
        }
        try:
            rpc(
                "fail_conversion_job",
                {
                    "p_job_id": job_id,
                    "p_attempt_id": attempt_id,
                    "p_error_code": code,
                    "p_error_detail": code,
                    "p_duration_ms": duration_ms,
                    "p_is_retryable": is_retryable,
                },
            )
        except Exception:
            pass
        raise RuntimeError(code) from error


async def handler(job: dict) -> dict:
    return await process_queued_job(job.get("input") if isinstance(job, dict) else None)


runpod.serverless.start({"handler": handler})
