# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements_model_training.txt

check_code:
	@flake8 scripts/* Covid_19/*.py

black:
	@black scripts/* Covid_19/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr Covid_19-*.dist-info
	@rm -fr Covid_19.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


##---------------------------------------------------------
##			Setting up Google Cloud project
##---------------------------------------------------------

# project id - replace with your GCP project id

#Eitans project
PROJECT_ID="batch-606-covid-19" 

#Chris project
#PROJECT_ID="wagon-bootcamp-312423"

#Alis project
#PROJECT_ID="batch-606-covid-19-315710"

#Ninas project
#PROJECT_ID="covid19-315509"

# bucket name - replace with your GCP bucket name

#Eitan Bucket
BUCKET_NAME=bucket-covid-19

#Chris Bucket
#BUCKET_NAME=wagon-data-606-hitz

#Alis Bucket
#BUCKET_NAME=bucket-covid-19-ali

#Ninas Bucket
## BUCKET_NAME=bucket-covid-19-predictions

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations

#Eitans, Alis, and Ninas region 
REGION=europe-west1

#Chris region
#REGION=europe-west6

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}
run_api:
	uvicorn api.fast:app --reload

##==================================================
##				Uploading dataset to the cloud
##==================================================

# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
# replace with your local path to the `dataset.csv,jpeg` and make sure to put the path between quotes
#LOCAL_PATH="<path to file or folder>"

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
#BUCKET_FOLDER=data
BUCKET_FOLDER=upload

# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = trainings

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

upload_data:
    # @gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

##==========================================================
## 						streamlit 
##==========================================================
streamlit:
	@streamlit run Covid_19/web_page.py


##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=Covid_19
FILENAME=trainer
RUNTIME_VERSION=2.4
PYTHON_VERSION = 3.7

#FRAMEWORK=scikit-learn


##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=covid_19_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

cp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs \
		--scale-tier STANDARD_1



# --scale-tier custom \
# --master-machine-type n1-highcpu-16 \
# --worker-machine-type n1-highcpu-16 \
# --parameter-server-machine-type n1-highmem-8 \
# --worker-count 2 \
# --parameter-server-count 3
