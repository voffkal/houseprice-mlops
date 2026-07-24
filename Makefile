ARTIFACTS = model/houseprice_model/artifacts

.PHONY: help train metrics compare submit image serve demo stop clean

## показать список доступных команд
help:
	@echo "houseprice-mlops — доступные команды:"
	@echo ""
	@grep -E '^## ' -A1 $(MAKEFILE_LIST) | \
	  grep -v '^--' | \
	  awk '/^## / { desc=substr($$0,4); next } \
	       /^[a-zA-Z_-]+:/ { sub(/:.*/,"",$$1); printf "  \033[36mmake %-10s\033[0m %s\n", $$1, desc }'

## обучить модель (артефакты → model/.../artifacts)
train:
	uv run python -m houseprice_model.train

## показать отчёт по метрикам последней обученной модели
metrics:
	@test -f $(ARTIFACTS)/metrics.json || \
	  { echo "Нет $(ARTIFACTS)/metrics.json — сначала обучи модель: make train"; exit 1; }
	@jq -r '\
	  def r2: .*10000 | round / 10000;\
	  def pct1: .*1000 | round / 10;\
	  def bar(frac): (frac*20 | round) as $$n | ("█"*$$n) + ("░"*(20-$$n));\
	  "=== House Price Model — отчёт по метрикам ===",\
	  "",\
	  "Версия модели : \(.model_version)   ·   обучена \(.trained_at[0:10])   ·   коммит \(.git_commit)",\
	  "Датасет       : \(.dataset.n_rows) строк (hash \(.dataset.hash))",\
	  "Окружение     : py \(.python) · sklearn \(.sklearn) · pandas \(.pandas) · numpy \(.numpy)",\
	  "",\
	  "── Качество модели (доля, чем выше тем лучше) ──",\
	  "R²    \(.metrics.in_sample.r2 | pct1)% \(.metrics.in_sample.r2 | bar(.))  объяснённая дисперсия",\
	  "",\
	  "── Ошибка предсказания (чем ниже тем лучше) ──",\
	  "MAPE  \(.metrics.in_sample.mape | pct1)%  средняя относительная ошибка цены",\
	  "MAE   $$\(.metrics.in_sample.mae | round)   средняя ошибка цены, USD",\
	  "RMSE  $$\(.metrics.in_sample.rmse | round)   ср.-кв. ошибка цены, USD (штрафует крупные промахи)",\
	  "",\
	  "── RMSLE: наша модель vs baseline (лог-единицы, чем ниже тем лучше) ──",\
	  (.baseline.cv_rmsle) as $$base |\
	  "CV (честная) \(.cv_rmsle.mean | r2)  \(.cv_rmsle.mean | bar(./$$base))  ± \(.cv_rmsle.std | r2)",\
	  "Baseline     \($$base | r2)  \($$base | bar(1.0))",\
	  "In-sample    \(.metrics.in_sample.rmsle | r2)  \(.metrics.in_sample.rmsle | bar(./$$base))  (оптимистично, на обучающих)",\
	  "",\
	  "Lift: +\(.baseline.lift_pct | .*10 | round / 10)% над baseline ✅"\
	' $(ARTIFACTS)/metrics.json

## предсказать Kaggle test.csv → submission.csv (нужен HOUSEPRICE_TEST=путь)
submit:
	uv run python scripts/predict_test.py

## сравнить метрики двух последних версий модели (metrics-*.json)
compare:
	@files=$$(ls $(ARTIFACTS)/metrics-*.json 2>/dev/null | sort -V); \
	n=$$(echo "$$files" | grep -c .); \
	if [ "$$n" -lt 2 ]; then \
	  echo "Нужно ≥2 версий metrics-*.json, сейчас: $$n."; \
	  echo "Обучи новую версию (bump VERSION → make train) и повтори."; \
	  exit 1; \
	fi; \
	old=$$(echo "$$files" | tail -2 | head -1); \
	new=$$(echo "$$files" | tail -1); \
	echo "=== Сравнение версий: $$(jq -r .model_version $$old)  →  $$(jq -r .model_version $$new) ==="; \
	echo ""; \
	printf "%-14s %12s %12s %12s\n" "Метрика" "$$(jq -r .model_version $$old)" "$$(jq -r .model_version $$new)" "Δ (%)"; \
	printf "%-14s %12s %12s %12s\n" "─────────────" "──────────" "──────────" "──────────"; \
	cmp() { \
	  o=$$(jq -r "$$2" $$old); nw=$$(jq -r "$$2" $$new); \
	  d=$$(echo "$$o $$nw" | awk '{ if ($$1==0) print "—"; else printf "%+.1f%%", ($$2-$$1)/$$1*100 }'); \
	  printf "%-14s %12s %12s %12s\n" "$$1" "$$o" "$$nw" "$$d"; \
	}; \
	cmp "CV RMSLE"  ".cv_rmsle.mean         | .*10000|round/10000"; \
	cmp "OOF RMSLE"  ".cv_rmsle.oof         | .*10000|round/10000"; \
	cmp "R²"         ".metrics.in_sample.r2 | .*10000|round/10000"; \
	cmp "MAE (USD)"  ".metrics.in_sample.mae | round"; \
	cmp "RMSE (USD)" ".metrics.in_sample.rmse | round"; \
	cmp "Lift %"     ".baseline.lift_pct    | .*10|round/10"; \
	echo ""; \
	echo "Для RMSLE/MAE/RMSE меньше = лучше (отрицательная Δ — улучшение)."

## собрать docker-образ API (обучит модель, если нужно)
image: train
	rm -rf model_artifacts && mkdir model_artifacts
	cp $(ARTIFACTS)/houseprice_model-*.joblib $(ARTIFACTS)/input_schema.json $(ARTIFACTS)/metrics.json model_artifacts/
	docker compose build api

## поднять API локально (http://localhost:8001/docs)
serve: image
	docker compose up -d api
	@echo "API: http://localhost:8001/docs"

# Временный публичный URL (*.trycloudflare.com) — живёт, пока работает команда.
# Для постоянного URL нужен свой домен в Cloudflare: тогда named tunnel
# через compose-профиль demo (TUNNEL_TOKEN) — см. compose.yaml.
## временный публичный URL через cloudflared
demo: serve
	@echo "Поднимаю временный публичный URL (Ctrl+C — остановить туннель):"
	cloudflared tunnel --url http://localhost:8001

## остановить контейнеры
stop:
	docker compose --profile demo down

## полный сброс: остановить контейнеры, удалить образ и скопированные артефакты
clean: stop
	rm -rf model_artifacts
	-docker compose rm -f api
	-docker image rm houseprice-mlops-api 2>/dev/null || true
	@echo "Очищено: model_artifacts/, контейнер и образ api удалены."
	@echo "Обученные модели в $(ARTIFACTS)/ НЕ тронуты (make train не нужен заново)."
