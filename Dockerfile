FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG STARVECTOR_COMMIT=0e083c1911760aa31bc576ca7f337a7f8ee605ec
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      libaio1 \
      libcairo2 \
    && python -m pip install --upgrade pip \
    && python -m pip install \
      accelerate \
      beautifulsoup4 \
      cairosvg \
      fairscale \
      hf-transfer \
      matplotlib \
      'numpy<2.0.0' \
      omegaconf \
      packaging \
      pillow \
      psutil \
      pydantic==2.10 \
      requests \
      runpod \
      scipy==1.11.1 \
      sentencepiece==0.2.0 \
      svgpathtools==1.6.1 \
      tokenizers==0.21.1 \
      transformers==4.49.0 \
      vtracer==1.0.0a3 \
      "${FLASH_ATTN_WHEEL}" \
    && python -m pip install --no-deps \
      "https://github.com/joanrod/star-vector/archive/${STARVECTOR_COMMIT}.tar.gz" \
    && rm -rf /root/.cache/pip /var/lib/apt/lists/*

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
