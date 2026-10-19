FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

# pip 源，默认使用 PyPI 官方源，可在构建时通过 --build-arg PIP_INDEX_URL=... 覆盖为镜像源
ARG PIP_INDEX_URL=https://pypi.org/simple

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -i "${PIP_INDEX_URL}" -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY . /app/

CMD nb run
