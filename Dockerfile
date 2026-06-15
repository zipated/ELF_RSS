FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

RUN python3 -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY . /app/

CMD nb run
