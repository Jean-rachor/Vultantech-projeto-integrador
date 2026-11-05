FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY src /app/src
COPY Data /app/Data
COPY Docs/Entregaveis/ML/vulcantech_pipeline.joblib /app/Docs/Entregaveis/ML/vulcantech_pipeline.joblib
COPY Docs/Exemplos/predicao_sensores_base.json /app/Docs/Exemplos/predicao_sensores_base.json
COPY Vulcantech_imagens_soldas /app/Vulcantech_imagens_soldas
ENV PYTHONPATH=/app/src VULCANTECH_DB=/app/vulcantech.sqlite3 VULCANTECH_MODEL=/app/Docs/Entregaveis/ML/vulcantech_pipeline.joblib
EXPOSE 8000
CMD ["uvicorn", "main:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8000"]
