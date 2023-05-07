# syntax=docker/dockerfile:1

FROM python:3.10-bullseye

EXPOSE 7865

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

CMD ["python3", "infer-web.py"]