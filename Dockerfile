FROM python:3.12-slim

WORKDIR /app

COPY . .

VOLUME [ "/app/files" ]

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "main.py"]