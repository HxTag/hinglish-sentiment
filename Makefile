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
	python -m src.evaluate
	python -m src.error_analysis
app:
	streamlit run app/streamlit_app.py
test:
	pytest -q
