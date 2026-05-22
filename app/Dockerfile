FROM python:3.12-slim

WORKDIR /employee-salary-prediction

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python train.py

EXPOSE 5000
CMD ["python", "app.py"]
