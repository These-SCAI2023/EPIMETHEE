# syntax = docker/dockerfile:experimental

FROM python:3.10-slim-bookworm

ARG list_of_packages="linux-headers-amd64 wget gcc cpp make cmake gfortran musl-dev libffi-dev libxml2-dev libxslt-dev libpng-dev"

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt update \
    && apt install -y $list_of_packages

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . ./app/
WORKDIR /app/

RUN if [ -e pip.conf ]; then cp pip.conf /etc/pip.conf; fi

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -e pip.conf ]; then INSTALL_OPTIONS="--index-url http://192.168.2.201:9191/index/"; fi; \
    pip install -r requirements.txt $INSTALL_OPTIONS

RUN apt remove -y $list_of_packages &&  \
    apt install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng tesseract-ocr-por tesseract-ocr-osd poppler-utils && \
    apt autoremove -y && apt clean -y && \
    rm -rf /var/lib/apt/lists/* \
    /root/.cache \
    /usr/lib/openblas

ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_DB=0
ENV REDIS_PASSWORD=""
ENV REDIS_PROTOCOL=3

EXPOSE 8080

CMD ["python", "toolbox_app.py"]
