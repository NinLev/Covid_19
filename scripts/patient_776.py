# -*- coding: utf-8 -*-
"""Patient 776

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CjxhXWFw-YyIHZJ0S5Y9kKQNAp5-6npQ
"""

!pip install pydicom

from google.cloud import storage
import cv2
import pydicom 
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt

bucket_name = 'wagon-data-606-hitz'

def list_blobs_with_prefix(bucket_name, prefix='data/Patient 776', postfix='.jpg',  delimiter=None):

      imgs_one_patient = []
      if postfix == '.jpg':
        storage_client = storage.Client.from_service_account_json('wagon-bootcamp-312423-bde5b1b38bca.json')
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=None)
        print("Blobs jpg:")
        for blob in blobs:
            
            if blob.name.endswith(postfix):
              print(blob.name)
              image = cv2.imdecode(np.asarray(bytearray(blob.download_as_string())), 0)
              print(image.shape)
              imgs_one_patient.append(image)

      elif postfix == '.dcm':
        storage_client = storage.Client.from_service_account_json('wagon-bootcamp-312423-bde5b1b38bca.json')
        blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=None)
        print("Blobs dcm:")
        for blob in blobs:
            
            if blob.name.endswith(postfix):
              print(blob.name)
              image = cv2.imdecode(np.asarray(bytearray(blob.download_as_string())), 0)
              imgs_one_patient.append(image)
              #dicom_bucket= blob.name
              #dicom_local=blob.name
              #select bucket file
              #blob = bucket.blob(dicom_bucket)
              #download that file and name it 'local.joblib'
              #blob.download_to_filename(dicom_local)
              #load that file from local file
              #model = keras.models.load_model(model_local)

              #image = pydicom.dcmread(fname)
              #image = cv2.imdecode(dicom_local)
              #print(image.shape)
              #imgs_one_patient.append(image)
              #pydicom.dcmread(dicom_local)   
                    
      return imgs_one_patient

def preprocess(imgs):
    """method that pre-process the data"""
    print('Preprocessing data')
    # Resizing the images
    #imgs = cv2.resize(imgs,(512,512))

    # Scaling the images
    imgs = imgs / 255.

    return imgs

def get_data(folder=None, postfix=None):
    """method to get the data from google cloud bucket"""
    
    bucket = storage.Client.from_service_account_json('wagon-bootcamp-312423-bde5b1b38bca.json').get_bucket(bucket_name)    
    imgs_one_patient = list_blobs_with_prefix(bucket_name, delimiter='/', prefix=folder, postfix=postfix) 
    imgs_one_patient = preprocess(np.array(imgs_one_patient)) 
       
    return np.expand_dims(imgs_one_patient, axis=-1)

def get_model(prefix=None, postfix=None):
    """method to get the pretrained model from google cloud bucket"""
    
    bucket = storage.Client.from_service_account_json('wagon-bootcamp-312423-bde5b1b38bca.json').get_bucket(bucket_name)    
    print(bucket)
    storage_client = storage.Client.from_service_account_json('wagon-bootcamp-312423-bde5b1b38bca.json')

    model_bucket='models/cnn_baseline/v1/model_complete_model_complete.h5'
    model_local='model_complete_model_complete.h5'
    #select bucket file
    blob = bucket.blob(model_bucket)
    #download that file and name it 'local.joblib'
    blob.download_to_filename(model_local)
    #load that file from local file
    model = keras.models.load_model(model_local)

    #blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=None)
    #print("Blobs:")
    #for blob in blobs:
    #    print(blob.name)
    #    if blob.name.endswith(postfix):
    #      model = keras.models.load_model(blob.download_as_string())
    #      #model = blob.download_as_string()
    model.summary()
                     
    return model

#test model load
model = get_model(prefix='models/cnn_baseline/v1', postfix='.h5')

def get_clinical_features(BUCKET_NAME=None, BUCKET_TRAIN_DATA_PATH=None):
    """method to get the all patients clinical features data (or a portion of it) from google cloud bucket"""
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
    return df

#df_cf = get_clinical_features(BUCKET_NAME=BUCKET_NAME, BUCKET_TRAIN_DATA_PATH=None)

#chris
X = get_data(folder='data/Patient 776', postfix='.jpg')

X.shape

y_prob = model.predict(X)
y_df = pd.DataFrame(y_prob, columns=['nCT', 'pCT', 'NiCT'])
y_df.head(n=10)

y_df['class'] = np.argmax(y_prob, axis=1)
y_df.head(n=10)

# Get only pCT slices
y_df = y_df[y_df['class'].isin([1])] 
y_df

# Sort pCT according to the probability in descending order and take the top 10
y_df = y_df.sort_values(by=['pCT'], ascending=False)[0:10]

# Get those slices
def get_top_ten_pCT_slices(imgs=None):

      return imgs[y_df.index, :, :, :]

X_top_ten = get_top_ten_pCT_slices(imgs=X)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr=None):
    fig, axes = plt.subplots(2, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( np.squeeze(images_arr), axes):
        ax.imshow(img, cmap='gray')
        ax.axis('on')
    plt.tight_layout()
    plt.show()

plotImages(images_arr = X_top_ten)

































































