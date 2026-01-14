# Базовый образ с Python 3.10
FROM python:3.10-slim

# Системные зависимости: компилятор для Cython extension, git, (опционально) bash/curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Создаём виртуальное окружение внутри контейнера (это НЕ ваш локальный venv; это часть контейнера)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Обновляем pip toolchain для сборки (setuptools/wheel/cython подтянутся из pyproject при установке)
RUN pip install -U pip setuptools wheel

WORKDIR /workspaces/owrf1d
