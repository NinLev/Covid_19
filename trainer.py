from google.cloud import storage
import numpy as np
import joblib
import cv2
from google.cloud import storage
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping


### GCP configuration - - - - - - - - - - - - - - - - - - -
# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -
# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -
#BUCKET_NAME = 'wagon-data-606-hitz'  #-- See MAKEFILE --
BUCKET_NAME = 'bucket-covid-19' 

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -
# train data file location
BUCKET_TRAIN_DATA_PATH = 'data/'  #-- Maybe include in MAKEFILE --

##### Training  - - - - - - - - - - - - - - - - - - - - - -
# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -
# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'cnn_baseline'
# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -
# not required here

def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #bucket = storage.Client.from_service_account_json('wagon-bootcamp-312423-bde5b1b38bca.json').get_bucket(BUCKET_NAME)
    bucket = storage.Client.from_service_account_json('wagon-bootcamp-312423-bde5b1b38bca.json').get_bucket(BUCKET_NAME)
    
    def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):

        #storage_client = storage.Client.from_service_account_json('wagon-bootcamp-312423-bde5b1b38bca.json')
        storage_client = storage.Client.from_service_account_json('wagon-bootcamp-312423-bde5b1b38bca.json')

        blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=None)

          print("Blobs:")

        images = []
        for blob in blobs:
            print(blob.name)
            if blob.name.endswith(".jpg"):
            image = cv2.imdecode(np.asarray(bytearray(blob.download_as_string())), 0)
            print(image.shape)         
            images.append(image)  

        return images

    imgs = list_blobs_with_prefix(bucket_name, f'data/labeled_CT_data/{label}', '/')

          
    return imgs
    
def compute_labels(value, shape, dtype):
    print('Computing labels')
    return np.full(shape, value, dtype=dtype)

def preprocess(imgs, labels):
    """method that pre-process the data"""
    print('Preprocessing data')
    # Resize images
    imgs=cv2.resize(imgs,(512,512))
    # Normalize images
    imgs = imgs / 255.
    # One-hot-encode classes
    labels = to_categorical(labels, 3)
    # Divide into training and test set
    imgs_train, imgs_test, labels_train, labels_test = train_test_split(imgs, labels, test_size=0.3, random_state=42)

    return imgs_train, imgs_test, labels_train, labels_test

def initialize_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(512, 512, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2, 2), activation='relu', padding='same'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    
    return model  

def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model      
    
def train_model(X_train, y_train):
    """method that trains the model"""
    print('Training data')
    es = EarlyStopping(patience=50, verbose=2)
    history = model.fit(X_train, y_train, 
                validation_split=0.3,
                callbacks=[es], 
                epochs=100, 
                batch_size=64,
                verbose = 2)

    return history


STORAGE_LOCATION = 'models/cnn_baseline/v1/model.joblib'

def upload_model_to_gcp():
    print('Uploading model to GCP')
    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename('model.joblib')

def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""
    
    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == '__main__':
    # get training data from GCP bucket
    X_nCT = get_data(label='nCT')
    X_nCT = np.expand_dims(X_nCT, axis=-1)
    X_pCT = get_data(label='pCT')
    X_pCT = np.expand_dims(X_pCT, axis=-1)
    X_NiCT = get_data(label='NiCT')
    X_NiCT = np.expand_dims(X_NiCT, axis=-1)

    # Compute labels
    labels_nCT = compute_labels(0, (np.shape(X_nCT)[0], 1), int)
    labels_pCT = compute_labels(1, (np.shape(X_pCT)[0], 1), int) 
    labels_NiCT = compute_labels(2, (np.shape(X_NiCT)[0], 1), int) 

    # preprocess data
    X_nCT_train, X_nCT_test, y_nCT_train, y_nCT_test = preprocess(X_nCT, labels_nCT)
    X_pCT_train, X_pCT_test, y_pCT_train, y_pCT_test = preprocess(X_pCT, labels_pCT)
    X_NiCT_train, X_NiCT_test, y_NiCT_train, y_NiCT_test = preprocess(X_NiCT, labels_NiCT)

    # Stack pCT, nCT, and NiCT them
    X_train = np.concatenate((X_nCT_train, X_pCT_train, X_NiCT_train), axis=0)
    print(f'Dimensions of X_train: {np.shape(X_train)}')
    y_train = np.concatenate((y_nCT_train, y_pCT_train, y_NiCT_train), axis=0)
    print(f'Dimensions of y_train: {np.shape(y_train)}')

    # Initialize model
    model = initialize_model()

    # Compile model
    model = compile_model(model)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    reg = train_model(X_train, y_train)

    # Evaluate model

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(reg)