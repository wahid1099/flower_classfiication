FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.enableCORS=false"]
