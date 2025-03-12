FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY models/ ./models/

COPY model.py .
COPY model_arch.py .
COPY app.py .

COPY API.log .
COPY __init__.py


EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
