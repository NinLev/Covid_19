FROM python:3.8.6-buster
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY models /models
COPY Covid_19 /Covid_19
COPY api /api
COPY batch-606-covid-19-5d766c13ace0.json /batch-606-covid-19-5d766c13ace0.json
COPY .env /.env
COPY Procfile /Procfile
COPY setup.sh /setup.sh
COPY Makefile /Makefile
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT