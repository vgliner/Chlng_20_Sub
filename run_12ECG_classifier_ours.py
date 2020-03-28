#!/usr/bin/env python
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as tvtf
import transforms as tf
from transforms import *
import torchvision
import timeit
import os
import models




def run_12ECG_classifier(data,header_data,classes,model):

    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # Use your classifier here to obtain a label and score for each class. 
    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1,-1)
    label = model.predict(feats_reshape)
    score = model.predict_proba(feats_reshape)

    current_label[label] = 1

    for i in range(num_classes):
        current_score[i] = np.array(score[0][i])

    return current_label, current_score
    

def load_12ECG_model():
    # load the model from disk 
    loaded_model=[]
    for cntr in range(9):
        uploaded_model=Upload_specific_trained_Deep_Net_model(cntr)
        loaded_model.append()

    return loaded_model



if __name__ == '__main__':
    print('Testing the file run_12_ECG_classifier_ours.py')
    print('Testing model loader...')
    loaded_model= load_12ECG_model()
    print('Model loaded...')

