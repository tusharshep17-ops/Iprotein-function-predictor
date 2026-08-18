FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home appuser

COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./app.py
RUN pip install --no-cache-dir ".[app]"

USER appuser
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

