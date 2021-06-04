#from numpy.lib.type_check import imag
import streamlit as st
import matplotlib.pyplot as plt
import random
from PIL import Image
#from google.cloud import storage
from cloud_storage import upload_to_bucket
#from Covid_19.cloud_storage import save_file_to_gcp
import requests







'''
# Project Covid19 front
'''


st.markdown('''
'' Here is our first prototype interface for our dear Project Covid_19 Thanks:*Nina,Eitan,Cyril,Chris,Ali* ''
the structure of model:
- Dataset (cleaning and preprocessing)
- Model building
- GCP: Docker image, API
- Interface and online services
''')
st.markdown(''' ''')
st.markdown(''' ''')
st.markdown(''' ''')
st.markdown(''' This Website will help clinicians to upload CT lung scans & key clinical features to get a fast confirmation or rejection of a Covid-19 pneumonia diagnosis as well as a mortality likelihood per patient
''')

# col1, col2 = st.beta_columns(2)
# image1 ="streamlit_images/firstpic.jpg"
# image2 ="streamlit_images/secondpic.png"
image3 ="streamlit_images/thirdpic.png"
image4 ="streamlit_images/wide_pic.jpg"

# original = Image.open(image1)
# col1.header("Original")
# col1.image(original, use_column_width=True)

# grayscale = Image.open(image2)
# col2.header("Grayscale")
# col2.image(grayscale, use_column_width=True)

# # thirdy = Image.open(image3)
# # col3.header("Graphs")
# # col3.image(thirdy, use_column_width=True)
st.markdown(''' ''')
st.markdown(''' ''')
st.markdown(''' ''')

if st.button('SHOW ME YOUR RESEARCH INFO'):
    # print is visible in server output, not in the page
    print('button clicked!')
    st.write('I was clicked 🎉')
    image = Image.open(image4)
    st.image(image, caption='CNN', use_column_width=True)
else:
    st.write('feel free to look 🎉')
    
    
if st.button('Results of our Model training'):
    # print is visible in server output, not in the page
    print('button clicked!')
    st.write('Pretty high')
    image = Image.open(image3)
    st.image(image, caption='CNN', use_column_width=True)
else:
    st.write("You wo'nt bt dissappointed")
    

# image = Image.open(image4)
# st.image(image, caption='CNN', use_column_width=True)
#st.markdown('<img src="./firstpic.jpg"/>', unsafe_allow_html=True)



uploaded_files = st.file_uploader("Please Uploade your CT-scans for prediction",accept_multiple_files = True)

#st.image(uploaded_files, caption="**YOU DONT HAVE CORONA**")

#cv2.imwrite('scan1.png', uploaded_file)
def load_image(image_file):
    img= Image.open(image_file)
    return img
## =================================================================================================
#                       step2: trying on multiple images with for loop
## ==================================================================================================

counter = random.randrange(10000)
list_of_names = []

for uploaded_file in uploaded_files:
   if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
                                                                                                                                #st.write(file_details)
                                                                                                                                #img = load_image(uploaded_file)
                                                                                                                                #st.image(img)
    counter +=1
    image_name = f"image_number({counter}).jpg"
    list_of_names.append(image_name)
    upload_to_bucket(image_name,uploaded_file)
    st.image(uploaded_file)
    st.success("file is amazing")
    
                                                                                                                                #    with open(os.path.join("data",uploaded_file.name), "wb") as f:
                                                                                                                                #        f.write(uploaded_file.getbuffer())
                                                                                                                                #st.success("file saved")

# =================================================================================================
#                   step3: filenames of the images (empty list that can be appended)
#=================================================================================================
st.write("File Names:",list_of_names)

# step4: importing the function to streamlit page
# CHECK 


# step5: testing on multiple images
# CHECK






















#with open(os.path.join("data",uploaded_file.name),"wb") as f:
      #  f.write(uploaded_file.getbuffer())
















#UPLOAD FUNCTION: 
# def upload_to_bucket(blob_name, file):
#     """ Upload data to a bucket"""
#     # Explicitly use service account credentials by specifying the private key
#     # file.
#     storage_client = storage.Client.from_service_account_json(
#         '../batch-606-covid-19-5d766c13ace0.json')
#     #print(buckets = list(storage_client.list_buckets())
#     bucket_name="bucket-covid-19"
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#     blob.upload_from_file(file)
#     #returns a public url
#     return True


#if uploaded_file is not None:
    #uploaded_file = Image.open(uploaded_file)
#    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
#   st.write("")
#   st.write("Classifying...")
#    #st.write(type(uploaded_file))
 #   label = machine_classification(uploaded_file,'model1.h5')
 #   my_bar = st.progress(0)
 #   for percent_complete in range(100):
 #       time.sleep(0.1)
 #       my_bar.progress(percent_complete + 1)
 #   if label == 0:
  #      st.subheader('RESULT :')
 #       t = "<div>As per our AI Engine - There is a chance that it is a<span class='highlight'> <span class='bold'> benign</span> </span> melanoma!</div>"
 #       st.markdown(t, unsafe_allow_html=True)
 #   else:
  #      st.subheader('RESULT :')
 #       t = "<div>As per our AI Engine - There is a chance that it is a<span class='highlight'> <span class='bold'> Malignant</span> </span> melanoma!</div>"
 #       st.markdown(t, unsafe_allow_html=True)
