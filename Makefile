.PHONY: data baseline bilstm finetune eval app test
data:
	python -m src.data_loader --input data/raw/hinglish_sentiment.csv
baseline:
	python -m src.baseline
bilstm:
	python -m src.bilstm
finetune:
	python -m src.transformer_finetune
eval:
	@echo "Phase 5 has not been implemented yet."
app:
	@echo "Phase 6 has not been implemented yet."
test:
	pytest -q
