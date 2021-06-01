from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import io, os
import tempfile


EXPERIMENT_NAME = 'COVID_CT_Scan_Predict'
BUCKET_NAME = 'bucket-covid-19' 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
                )

def get_files_from_storage(file_names):
    #get files from google storage
    images = []
    if file_names:
        for file_name in file_names:
            client = storage.Client('[batch-606-covid-19]')
            bucket = client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(file_name)
            images.append(blob.download_to_filename(file_name))
        return images 
    else:
        return None
    
def upload_files_to_storage(files):
    if files:
        for file in files:
            client = storage.Client('[batch-606-covid-19]')
            bucket = client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(file_name)
            with tempfile.NamedTemporaryFile() as temp:
                blob.download_to_filename(temp.name)


@app.get("/")
def index():
    return {"greeting": "Hello"}

@app.get("/predict")
def predict(file_names):
    file_names = file_names.rsplit(',')
    CT_images = get_files_from_storage(file_names)
    diagnosis = len(CT_images) 
    return {"diagnosis":diagnosis}


