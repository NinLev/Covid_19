from google.cloud import storage
from os.path import join, dirname
from dotenv import load_dotenv
import os





# ...
env_path = join(dirname(dirname(__file__)),'.env') # ../.env
load_dotenv(dotenv_path=env_path)
C19_API_KEY = os.getenv('C19_API_KEY')


# step1: creating a funtion that uploads one image to google drive
# Function 1: 

def upload_to_bucket(blob_name, file):
    """ Upload data to a bucket"""
    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(C19_API_KEY)
    #print(buckets = list(storage_client.list_buckets())
    bucket_name="bucket-covid-19"
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(file)
    #returns a public url
    return True

#'../batch-606-covid-19-5d766c13ace0.json'
#upload_to_bucket("image1",)


#client = storage.Client('[batch-606-covid-19]')
#bucket = client.get_bucket(BUCKET_NAME)
#blob = bucket.blob()

# def save_model_to_gcp():
#     BUCKET_NAME = "le-wagon-data"
#     storage_location = "models/random_forest_model.joblib"
#     local_model_filename = "model.joblib"
#     client = storage.Client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(storage_location)
#     blob.upload_from_filename(local_model_filename)

# step2: trying on multiple images with for loop

















