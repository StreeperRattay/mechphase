FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mechphase

COPY pyproject.toml requirements.txt README.md LICENSE NOTICE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN python3.11 -m pip install --upgrade pip \
 && python3.11 -m pip install -r requirements.txt \
 && python3.11 -m pip install --no-deps -e .

ENTRYPOINT ["python3.11", "-m", "mechphase.harness.fit"]
