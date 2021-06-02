FROM python:3.8.6-buster
#COPY model.joblib /model.joblib 
COPY Covid_19 /Covid_19
COPY api /api
COPY requirements.txt /requirements.txt
COPY batch-606-covid-19-5d766c13ace0.json /credentials.json
COPY .env /.env
COPY Procfile /Procfile
COPY setup.sh /setup.sh
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT