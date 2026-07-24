FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
# ВАЖНО: WORKDIR совпадает с рантаймом — шебанги консольных скриптов venv зашиваются абсолютным путём при uv sync
WORKDIR /opt/houseprice
COPY . .
# --no-editable: пакеты ставятся колёсами в .venv -> рантайму не нужны исходники
RUN uv sync --locked --no-dev --no-editable --package houseprice-api

FROM python:3.12-slim AS runtime
# libgomp1: OpenMP-рантайм, нужен xgboost и lightgbm (базовые ученики стека v0.2)
# при импорте. v0.1 (только Ridge) не требовал — отсюда падение "libgomp.so.1:
# cannot open shared object file" на слим-образе после перехода на стек.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN adduser --disabled-password --gecos '' api-user
WORKDIR /opt/houseprice
COPY --from=builder /opt/houseprice/.venv .venv
# Артефакты модели: кладутся в контекст make image (локально) или docker.yml (CI из релиза)
COPY model_artifacts/ model_artifacts/
ENV PATH="/opt/houseprice/.venv/bin:$PATH" \
    ARTIFACTS_DIR=/opt/houseprice/model_artifacts
USER api-user
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
