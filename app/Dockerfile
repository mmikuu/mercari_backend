FROM python:3.8

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

CMD python3 ./app/main.py

RUN pip install sqlite3
RUN sqlite3 test.db