# Covid19 diagnosis and prediction

This project is designed to improve diagnosis of Covid-19 pneumonia & predict the disease severity and risk factor through automated analysis of CT-scan images and clinical indices.
This tool uses two different but equally important models to predict the severity of the disease on one patient.


## Model 1- Disease classification using CT scans (deep learning): 

By uploading the chest images and feeding it to the model, it is going to tell us if the patient has indication of sickness 
Using deep learning model allows users to upload CT-scan images of a patient’s lungs and receive an immediate and accurate diagnosis of their COVID-19 status: Negative / Positive Mild / Positive Severe. 

Model description:

The first Convolutional Neuronal Network model takes labelled CT slices with 3 classes as an input and learns the pattern within these CT scans to distinguish CT scans with or without Covid-19 features. Additionally, the network also learns how to identify CT scans, which shows non-informative patterns. Therefore, the model is able to label new CT scans with one of three labels:
•	Positive CT scan with Covid-19 features;
•	Negative CT scan with no Covid-19 features;
•	Non-informative CT scans.
To be able to have a high accuracy in the predictions, we use a sequential network architecture composed of several alternating layers of Conv2D, MaxPooling2D, and Dropout blocks, followed by a Flatten layer, a Dense layer, a Dropout layer, and a final Dense layer with three neurons to perform the classification of each CT scan into one of the three classes.
After finding out that the patient is positive for covid 19, we can use the second model to assess the Risk level

## Model 2 - Clinical features/Mortality risk evaluation (machine learning):

For a confirmed Positive patient this machine learning model allows users to upload a set of clinical indices and receive an evaluated risk factor for that patient: High Risk / Low Risk.
This way doctors can have a full analysis of a patient’s condition and treat them accordingly if they needed extra attention.  
The project is designed to produce a tool that is meant to be used by medical personnel that often work in hospitals under stressful conditions, especially at times of crisis and pandemics. So, we are hoping that this quick two step process could reduce their work load significantly. The tool is currently deployed online and can be accessed by anyone.
For further questions or potential collaboration feel free to contact us on one of the following addresses:

Ali: leesalem@gmail.com
Cyril: cyrilaubry22@gmail.com
Eitan: eitanir.is@gmail.com
Nina: nina.l.ams@gmail.com
