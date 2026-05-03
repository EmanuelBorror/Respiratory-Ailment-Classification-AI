# Respiratory-Ailment-Classification-AI
A machine learning model trained on .wav files of asthma, COPD, Bronchial, pneumonia, and healthy breathing to propose probable diagnoses.  

See the dataset_download.py file to download the dataset for training with this script. 

Utilizes the Asthma Detection Dataset Version 2 from kaggle.com: https://www.kaggle.com/datasets/mohammedtawfikmusaed/asthma-detection-dataset-version-2/data

This ML model utilizes a series of convolutional layers to identify the features of MFCC spectrograms to diagnose respiratory illness from breathing patterns. 
The model utilizes CNN, BiGRU, and Attention to reduce the effects of unequal samples in the dataset.   

