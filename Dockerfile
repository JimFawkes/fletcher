FROM python:3.6
WORKDIR /usr/src/app

RUN mkdir fletcher logs data
COPY requirements.txt .
COPY download_models.sh .
RUN python3 -m pip install -r requirements.txt

COPY fletcher/ fletcher/

RUN ./download_models.sh

CMD ["python", "fletcher", "--help"]
