# syntax = docker/dockerfile:experimental

FROM python:3.10-slim-bookworm

ARG list_of_packages="linux-headers-amd64 wget gcc cpp make cmake gfortran musl-dev libffi-dev libxml2-dev libxslt-dev libpng-dev"

RUN apt update && apt install -y $list_of_packages

#RUN wget https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.27/OpenBLAS-0.3.27.tar.gz \
#	&& tar -xf OpenBLAS-0.3.27.tar.gz \
#	&& cd OpenBLAS-0.3.27 \
#	&& make BINARY=64 FC=$(which gfortran) USE_THREAD=1 \
#	&& make PREFIX=/usr/lib/openblas install \
#    && cd .. \
#    && rm -rf OpenBLAS-0.3.27

#RUN wget https://downloads.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz?ts=gAAAAABmRK1m1EsbH8oCL7geuqybhgEF3yNS0j2REvwcja2UzHU2bT7s4ZU7gqMqQK0ZDTzecMhN4GWfb8GyWlTgoHLXg9XnDA%3D%3D&r= \
#    && tar -xf swig-3.0.12.tar.gz \
#    && cd swig-3.0.12 \
#    && ./configure \
#    && make \
#    && make install \
#    && cd .. \
#    && rm -rf swig-3.0.12

RUN --mount=type=cache,target=/root/.cache/pip \
    ATLAS=/usr/lib/openblas/lib/libopenblas.so LAPACK=/usr/lib/openblas/lib/libopenblas.so pip install scipy

RUN export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/openblas/lib/

COPY . ./app/
WORKDIR /app/

RUN --mount=type=cache,target=/root/.cache/pip pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

RUN apt remove -y $list_of_packages &&  \
    apt install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng tesseract-ocr-osd poppler-utils && \
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
