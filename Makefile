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

demo: image
	TUNNEL_TOKEN=$$(security find-generic-password -s agents-system -a houseprice-tunnel -w) \
		docker compose --profile demo up -d
	@echo "Туннель поднят; постоянный URL — в дашборде Cloudflare Zero Trust"

stop:
	docker compose --profile demo down
