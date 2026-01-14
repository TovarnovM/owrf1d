# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /work

# Кладём скрипты smoke-теста в образ
COPY smoke_test/ /work/smoke_test/
RUN chmod +x /work/smoke_test/run_smoke.sh

# По умолчанию запускаем smoke
CMD ["/work/smoke_test/run_smoke.sh"]
