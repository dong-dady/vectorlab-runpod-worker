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
    max_generation_attempts = 2
    valid_modes = {"balanced", "high_detail"}
    starvector_max_new_tokens = 7800
    starvector_context_safety_tokens = 16
    model_id = os.getenv("STARVECTOR_MODEL_ID", "starvector/starvector-1b-im2svg")
    model_revision = os.getenv(
        "STARVECTOR_MODEL_REVISION",
        "380ab95d25a8e9ab1dc825debe238b4953ae13b9",
    )
    cached_model_root = os.getenv(
        "STARVECTOR_CACHED_MODEL_ROOT",
        "/runpod-volume/huggingface-cache/hub",
    )
    require_cached_model = os.getenv(
        "STARVECTOR_REQUIRE_CACHED_MODEL",
        "false",
    ).strip().lower() in {"1", "true", "yes"}
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

    def resolve_cached_model_path() -> str | None:
        repo_cache_name = f"models--{model_id.replace('/', '--')}"
        repo_cache_path = os.path.join(cached_model_root, repo_cache_name)
        snapshots_path = os.path.join(repo_cache_path, "snapshots")
        candidates = [os.path.join(snapshots_path, model_revision)]

        refs_path = os.path.join(repo_cache_path, "refs")
        for ref_name in (model_revision, "main"):
            ref_path = os.path.join(refs_path, ref_name)
            try:
                with open(ref_path, encoding="utf-8") as ref_file:
                    ref_revision = ref_file.read().strip()
            except OSError:
                continue
            if ref_revision:
                candidates.append(os.path.join(snapshots_path, ref_revision))

        try:
            snapshot_names = sorted(os.listdir(snapshots_path))
        except OSError:
            snapshot_names = []
        if len(snapshot_names) == 1:
            candidates.append(os.path.join(snapshots_path, snapshot_names[0]))

        for candidate in dict.fromkeys(candidates):
            if not os.path.isfile(os.path.join(candidate, "config.json")):
                continue
            try:
                has_weights = any(
                    filename.endswith((".safetensors", ".bin"))
                    for filename in os.listdir(candidate)
                )
            except OSError:
                continue
            if has_weights:
                return candidate
        return None

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
            "<image",
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
        drawable_count = 0
        drawable_tags = {
            "circle",
            "ellipse",
            "line",
            "path",
            "polygon",
            "polyline",
            "rect",
            "text",
        }
        for element in root.iter():
            tag = element.tag.split("}")[-1].lower()
            if tag in drawable_tags:
                drawable_count += 1
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

        if drawable_count == 0:
            raise ValueError("svg_empty")

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
        from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

        class StopOnSequence(StoppingCriteria):
            def __init__(self, stop_ids):
                self.stop_ids = torch.tensor(stop_ids, dtype=torch.long)

            def __call__(self, input_ids, scores, **kwargs):
                if input_ids.shape[1] < self.stop_ids.numel():
                    return False
                stop_ids = self.stop_ids.to(input_ids.device)
                matches = input_ids[:, -stop_ids.numel():] == stop_ids
                return bool(matches.all(dim=1).all().item())

        global _VECTORLAB_STARVECTOR_MODEL
        try:
            model = _VECTORLAB_STARVECTOR_MODEL
        except NameError:
            cached_model_path = resolve_cached_model_path()
            if cached_model_path:
                model_source = cached_model_path
                model_kwargs = {"local_files_only": True}
                model_source_name = "runpod_cache"
            else:
                if require_cached_model:
                    raise RuntimeError("cached_model_unavailable")
                model_source = model_id
                model_kwargs = {"revision": model_revision}
                model_source_name = "huggingface"
            print(f"starvector_model_load source={model_source_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **model_kwargs,
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

        core = model.model
        tokenizer = core.svg_transformer.tokenizer
        transformer = core.svg_transformer.transformer
        inputs_embeds, attention_mask, prompt_tokens = core._prepare_generation_inputs(
            {"image": pixel_values},
            None,
            pixel_values.device,
        )
        context_limit = int(
            getattr(
                transformer.config,
                "max_position_embeddings",
                getattr(model.config, "max_position_embeddings", 8192),
            )
        )
        prefix_tokens = int(inputs_embeds.shape[1])
        hard_budget = context_limit - prefix_tokens - starvector_context_safety_tokens
        max_new_tokens = min(starvector_max_new_tokens, hard_budget)
        if max_new_tokens < 256:
            raise RuntimeError("generation_context_exhausted")

        close_text = "</svg>"
        close_ids = tokenizer(close_text, add_special_tokens=False)["input_ids"]
        stopping_criteria = StoppingCriteriaList([StopOnSequence(close_ids)])
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        generation_profiles = (
            {
                "name": "sampled_low_temperature",
                "do_sample": True,
                "temperature": 0.2,
                "top_p": 0.95,
            },
            {
                "name": "greedy",
                "do_sample": False,
            },
        )

        last_failure = "generation_failed"
        with torch.inference_mode():
            for attempt, profile in enumerate(generation_profiles[:max_generation_attempts]):
                seed = base_seed + attempt
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
                attempt_started_at = time.perf_counter()
                generation_kwargs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "do_sample": profile["do_sample"],
                    "num_beams": 1,
                    "max_new_tokens": max_new_tokens,
                    "repetition_penalty": 1.0,
                    "use_cache": True,
                    "stopping_criteria": stopping_criteria,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": False,
                }
                if profile["do_sample"]:
                    generation_kwargs.update(
                        temperature=profile["temperature"],
                        top_p=profile["top_p"],
                    )

                outputs = transformer.generate(**generation_kwargs)
                generated_tokens = outputs.sequences[0]
                decoded_tokens = torch.cat(
                    [prompt_tokens.input_ids[0], generated_tokens],
                    dim=0,
                )
                raw_svg = tokenizer.decode(decoded_tokens, skip_special_tokens=True)
                lowered = raw_svg.lower()
                svg_start = lowered.find("<svg")
                svg_end = lowered.find(close_text, svg_start)
                has_svg_close = svg_start >= 0 and svg_end >= 0
                token_count = int(generated_tokens.numel())
                eos_seen = (
                    tokenizer.eos_token_id is not None
                    and bool((generated_tokens == tokenizer.eos_token_id).any().item())
                )
                stop_reason = (
                    "svg_close"
                    if has_svg_close
                    else "eos"
                    if eos_seen
                    else "token_limit"
                    if token_count >= max_new_tokens
                    else "unknown"
                )
                attempt_ms = round((time.perf_counter() - attempt_started_at) * 1000)
                peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
                print(
                    "starvector_generation "
                    f"attempt={attempt + 1} "
                    f"profile={profile['name']} "
                    f"prefix_tokens={prefix_tokens} "
                    f"max_new_tokens={max_new_tokens} "
                    f"generated_tokens={token_count} "
                    f"raw_chars={len(raw_svg)} "
                    f"has_svg_close={has_svg_close} "
                    f"stop_reason={stop_reason} "
                    f"elapsed_ms={attempt_ms} "
                    f"peak_vram_mb={peak_vram_mb}"
                )
                if not has_svg_close:
                    last_failure = (
                        "generation_truncated"
                        if stop_reason == "token_limit"
                        else "generation_incomplete"
                    )
                    continue

                svg = raw_svg[svg_start : svg_end + len(close_text)].strip()
                try:
                    converted = validate_svg(svg)
                except ValueError as error:
                    last_failure = str(error)
                    continue
                converted["metrics"].update(
                    {
                        "generationProfile": profile["name"],
                        "generationTokenCount": token_count,
                        "generationStopReason": stop_reason,
                        "prefixTokenCount": prefix_tokens,
                        "maxNewTokens": max_new_tokens,
                        "peakVramMb": peak_vram_mb,
                    }
                )
                return converted

        raise ValueError(last_failure)

    def safe_error_code(error: Exception) -> str:
        known = {
            "invalid_image",
            "image_dimensions_limit",
            "invalid_mode",
            "svg_not_text",
            "svg_size_limit",
            "svg_invalid_root",
            "svg_unsafe_content",
            "svg_empty",
            "generation_failed",
            "generation_incomplete",
            "generation_truncated",
            "generation_context_exhausted",
            "cached_model_unavailable",
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
