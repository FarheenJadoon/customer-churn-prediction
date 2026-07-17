FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY xgboost-2.1.1-py3-none-manylinux_2_28_x86_64.whl .
COPY scipy-1.17.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl .

RUN pip install --no-cache-dir --no-deps xgboost-2.1.1-py3-none-manylinux_2_28_x86_64.whl
RUN pip install --no-cache-dir --no-deps scipy-1.17.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
RUN pip install --no-cache-dir --prefer-binary --default-timeout=1000 --retries 5 -r requirements.txt

COPY app ./app
COPY churn_model.pkl .
COPY scaler.pkl .
COPY model_columns.json .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]