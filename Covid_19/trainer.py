from google.cloud import storage
import pandas as pd
from sklearn import linear_model
import numpy as np
import joblib



### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'bucket-covid-19-predictions'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/Patient 501/CT/'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'models'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def get_data():
    """ function used in order to get the training data (or a portion of it) from Cloud Storage """
        
    from google.cloud import storage

    import numpy as np
    import cv2
    bucket = storage.Client().get_bucket(BUCKET_NAME)
    def load_data(bucket_name):
        bucket = storage.Client().get_bucket(bucket_name)

        return np.array(
            cv2.imdecode(
                np.asarray(bytearray(blob.download_as_string()), dtype=np.uint8), 0
            ).flatten()
            for blob in bucket.list_blobs()
            if blob.name.endswith(".jpg")
        )

    #X = load_data(f"gs://{BUCKET_NAME}")    
    return bucket
       
def preprocess():
    """ function that pre-processes the data """

    pass 

def train_model(X_train, y_train):
    """ function that trains the model """
    pass

def save_model(reg):
    """ method that saves the model into a .joblib file and uploads it on Google Storage /models folder """
    pass

if __name__ == '__main__':
    """ runs a training """
    target_list = get_data()
    print(target_list)