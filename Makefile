ARTIFACTS = model/houseprice_model/artifacts

.PHONY: train image serve demo stop

train:
	uv run python -m houseprice_model.train

image: train
	rm -rf model_artifacts && mkdir model_artifacts
	cp $(ARTIFACTS)/houseprice_model-*.joblib $(ARTIFACTS)/input_schema.json $(ARTIFACTS)/metrics.json model_artifacts/
	docker compose build api

serve: image
	docker compose up -d api
	@echo "API: http://localhost:8001/docs"

# Временный публичный URL (*.trycloudflare.com) — живёт, пока работает команда.
# Для постоянного URL нужен свой домен в Cloudflare: тогда named tunnel
# через compose-профиль demo (TUNNEL_TOKEN) — см. compose.yaml.
demo: serve
	@echo "Поднимаю временный публичный URL (Ctrl+C — остановить туннель):"
	cloudflared tunnel --url http://localhost:8001

stop:
	docker compose --profile demo down
