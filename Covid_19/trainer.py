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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from os.path import join, dirname
import dotenv
from dotenv import load_dotenv

import os
import glob

from PIL import Image



# ...
#env_path = join(dirname(dirname(__file__)),'.env') # ../.env
#load_dotenv(dotenv_path=env_path)
#C19_API_KEY =  'batch-606-covid-19-5d766c13ace0.json' #os.getenv('C19_API_KEY')


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

def save_file_to_gcp(filename,file):
    BUCKET_NAME = "bucket-covid-19"
    #BUCKET_NAME='bucket-covid-19-predictions'
    #storage_location = "models/random_forest_model.joblib"
    #local_model_filename = "model.joblib"
    client = storage.Client('batch-606-covid-19')
    #client = storage.Client('Covid19')

    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_filename(file)


### GCP AI Platform - - - - - - - - - - - - - - - - - - - -
# not required here

def get_data(label):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #bucket = storage.Client.from_service_account_json('wagon-bootcamp-312423-bde5b1b38bca.json').get_bucket(BUCKET_NAME)
    #bucket = storage.Client.from_service_account_json(C19_API_KEY).get_bucket(BUCKET_NAME)
    print('###### get_data')
    
    def list_blobs_with_prefix(BUCKET_NAME, prefix, delimiter=None):
        print('###### list blobs prefix')
        #storage_client = storage.Client.from_service_account_json('wagon-bootcamp-312423-bde5b1b38bca.json')
        print()

        client = storage.Client()

        #storage_client = storage.Client.from_service_account_json(C19_API_KEY)
        
        bucket = client.bucket(BUCKET_NAME)        
        blobs = client.list_blobs(BUCKET_NAME, prefix=prefix, delimiter=None)

        print("Blobs:")

        images = []
        
        # for blob in blobs:
        #     #print(blob.name)
        #     if blob.name.endswith(".jpg"):
        #         image = cv2.imdecode(np.asarray(bytearray(blob.download_as_string())), 0)
        #         #print(image.shape)         
        #         images.append(image)  
                
        ct = 0
    
        for blob in blobs:
            while ct < 2500:
                #print(blob.name)
                if blob.name.endswith(".jpg"):
                    #blob.download_to_filename('downloaded_image.png')
                    image = cv2.imdecode(np.asarray(bytearray(blob.download_as_string())), 0)
                    #image = Image.open("downloaded_image.png") #.convert('RGB')
                    #print(image.shape)         
                    images.append(image)
                    #images.append(np.array(image))

                ct +=1
        
        return images

    imgs = list_blobs_with_prefix(BUCKET_NAME, f'data/labeled_CT_data/{label}', '/')
 
    #save_file_to_gcp('test', imgs)
    print('Getting data completed')      
    return imgs
    
def compute_labels(value, shape, dtype):
    print('Computing labels')
    return np.full(shape, value, dtype=dtype)

def preprocess(imgs, labels):
    """method that pre-process the data"""
    print('Preprocessing data')
    print(len(imgs))
    # Resize images
    #imgs=cv2.resize(imgs,(512,512))
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
    
def data_augmentation_model(X_train, y_train, X_test, y_test, model):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False,
        zoom_range=(0.8, 1.2),
        )
    
    datagen.fit(X_train)
    
    X_augmented_iterator = datagen.flow(X_train, shuffle=False, batch_size=1)
    
    # The model
    model_aug = initialize_model()
    model_aug = compile_model(model_aug)
    
    # The data generator
    X_tr = X_train[:40000]
    y_tr = y_train[:40000]
    X_val = X_train[40000:]
    y_val = y_train[40000:]
    train_flow = datagen.flow(X_tr, y_tr, batch_size=64)
    
    # The early stopping criterion
    es = EarlyStopping(patience=3)
    
    # The fit
    history_aug = model_aug.fit(train_flow,
                            epochs=50, 
                            callbacks=[es], 
                            validation_data=(X_val, y_val))
    
    #res_1 = model.evaluate(X_test, y_test, verbose=0)
    res_2 = model_aug.evaluate(X_test, y_test)
    
    #print(f'Accuracy without data augmentation {res_1[1]*100:.2f}%')
    print(f'Accuracy with data augmentation {res_2[1]*100:.2f}%')
    
    return model_aug      
    
def train_model(model, X_train, y_train):
    """method that trains the model"""
    print('Training data')
    es = EarlyStopping(patience=5, verbose=2)
    history = model.fit(X_train, y_train, 
                validation_split=0.3,
                callbacks=[es], 
                epochs=100, 
                batch_size=64,
                verbose = 2)

    return history


STORAGE_LOCATION = 'models/cnn_baseline/v1'

def upload_model_to_gcp():
    print('Uploading model to GCP')
    #client = storage.Client()
    #client = storage.Client.from_service_account_json(C19_API_KEY) #.get_bucket(BUCKET_NAME)

    # client = storage.Client()
    # bucket = client.get_bucket(BUCKET_NAME)
    # blob = bucket.blob(STORAGE_LOCATION)

    # #blob.upload_from_filename('model.joblib')
    # blob.upload_from_filename('model')
    
    upload_from_directory('model_2500', BUCKET_NAME, 'model_2500')  #creates model_complete under Covid-19 , then create models
    
def upload_from_directory(directory_path: str, dest_bucket_name: str, dest_blob_name: str):
    client = storage.Client()
    rel_paths = glob.glob(directory_path + '/**', recursive=True)
    bucket = client.get_bucket(dest_bucket_name)
    for local_file in rel_paths:
        remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)

def save_model(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""
    
    # saving the trained model to disk is mandatory to then beeing able to upload it to storage
    # Implement here
    #model.save('model_2000')
    model.save('./model_2500/model_2500.h5')  # creates a HDF5 file 'my_model.h5' locally
    #joblib.dump(reg, 'model.joblib')
    #print("saved model.joblib locally")

    # Implement here
    upload_model_to_gcp()
    print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


if __name__ == '__main__':
    # get training data from GCP bucket
    
    print('We are in the main function')

    #client = storage.Client.from_service_account_json(C19_API_KEY) #.get_bucket(BUCKET_NAME)
    #open('test.txt', 'w+').close()
    #save_file_to_gcp('test.txt','test.txt')
    
    
    X_nCT = get_data('nCT')
    print(len(X_nCT))
    X_nCT = np.expand_dims(X_nCT, axis=-1)
    print(len(X_nCT))
    X_pCT = get_data('pCT')
    print(len(X_pCT))
    X_pCT = np.expand_dims(X_pCT, axis=-1)
    print(len(X_pCT))
    X_NiCT = get_data('NiCT')
    print(len(X_NiCT))
    X_NiCT = np.expand_dims(X_NiCT, axis=-1)
    print(len(X_NiCT))


    # Compute labels
    labels_nCT = compute_labels(0, (np.shape(X_nCT)[0], 1), int)
    labels_pCT = compute_labels(1, (np.shape(X_pCT)[0], 1), int) 
    labels_NiCT = compute_labels(2, (np.shape(X_NiCT)[0], 1), int)
    
    print(labels_nCT.shape, labels_pCT.shape, labels_NiCT.shape)
    
    # preprocess data
    X_nCT_train, X_nCT_test, y_nCT_train, y_nCT_test = preprocess(X_nCT, labels_nCT)
    X_pCT_train, X_pCT_test, y_pCT_train, y_pCT_test = preprocess(X_pCT, labels_pCT)
    X_NiCT_train, X_NiCT_test, y_NiCT_train, y_NiCT_test = preprocess(X_NiCT, labels_NiCT)
    
    print(X_nCT_train.shape, X_nCT_test.shape, X_pCT_train.shape, X_pCT_test.shape, X_NiCT_train.shape, X_NiCT_test.shape)

    # Stack pCT, nCT, and NiCT them
    X_train = np.concatenate((X_nCT_train, X_pCT_train, X_NiCT_train), axis=0)
    print(f'Dimensions of X_train: {np.shape(X_train)}')
    y_train = np.concatenate((y_nCT_train, y_pCT_train, y_NiCT_train), axis=0)
    print(f'Dimensions of y_train: {np.shape(y_train)}')

    # augmentation
    #data_augmentation_model(    )


    # Initialize model
    print('initializing model')
    model = initialize_model()

    # Compile model
    print('compiling model')
    model = compile_model(model)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    print('train model')
    reg = train_model(model, X_train, y_train)

    # Evaluate model

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    print('saving model')
    save_model(reg)