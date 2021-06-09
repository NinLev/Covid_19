from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
import io, os
import tempfile
from os.path import join, dirname
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import cv2
import joblib
import pandas as pd


# ...
env_path = join(dirname(dirname(__file__)),'.env') # ../.env
load_dotenv(dotenv_path=env_path)
C19_API_KEY = os.getenv('C19_API_KEY')

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
            storage_client = storage.Client.from_service_account_json(C19_API_KEY)
            bucket = storage_client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(file_name)
            images.append(blob)
        return images 
    else:
        return None
# preprocessing an image - grayscale and size
def preproc_before_pred(img):
    img=cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    #img=cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(200,200))
    return img

def decode_prediction(prediction):
    res = ''
    if prediction[0][0]==1:
        return 'Non-Informative'
    if prediction[0][1]==1:
        return 'Positive'
    if prediction[0][2]==1:
        return 'Negative'
        

@app.get("/")
def index():
    return {"greeting": "Hello"}


@app.get("/predict")
def predict(file_names):
    loaded_model = tf.keras.models.load_model('models/model_labeled_ct_15epochs') # load model
    file_names = file_names.rsplit(',')  # split file names string to list of file names 
    CT_images = get_files_from_storage(file_names)  # get images from cloud storage
    CT_images_preproc = []
    for blob in CT_images:
        blob.download_to_filename('temp1.jpeg')
        CT_images_preproc.append(np.array([preproc_before_pred('temp1.jpeg')])[:,:,:,np.newaxis])

    results = {}
    print(CT_images_preproc)
    # loop that goes over the client files, predicts and round result to determine class
    for image, file_name in zip(CT_images_preproc, file_names):
        diagnosis = loaded_model.predict(image) 
        rounded_diagnosis = np.rint(diagnosis)
        decode_diagnosis = decode_prediction(rounded_diagnosis)         
        results[f'{file_name}'] =  decode_diagnosis
    return {'result':decode_diagnosis} #returns a dictionary with file names as keys and one hot coded list as class


class Patient(BaseModel):
    NEP: float
    Age: float
    LDH: float
    NE: float
    LYP: float
    LY: float
    ALB: float
    EOP: float
    EO: float
    ALG: float
    CA: float
    MOP: float
    INR: float
    BUN: float
    TBIL: float
    WBC: float
    DD: float


@app.post("/predictcf/")
def predictcf(patient: Patient):
    # staging_pipeline = joblib.load('../notebooks/staging_pipeline.pkl')
    patient_dict = patient.dict()
    # transformed = staging_pipeline.transform(df_to_predict)
    trained_model = joblib.load('notebooks/cf_model.pkl')
    
    prediction = trained_model.predict(pd.DataFrame(patient_dict,index=[0]))
    return {'result':prediction.tolist()[0]} 