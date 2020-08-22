#!/usr/bin/env python

import numpy as np, os, sys
import joblib
from get_12ECG_features import get_12ECG_features
from models import *
import random
import torch
import pandas as pd
from Utils import *


def augmentation_algorithm(record =  None,record_length=3e4):
    current_record_length = record.shape[1]
    if current_record_length == record_length:
        return record
    if current_record_length <= record_length:  # record is shorter than maximal length or similar
        new_sample = np.zeros((12, record_length))
        index_for_pasting = random.sample(range(record_length - current_record_length), 1)
        new_sample[:, index_for_pasting[0]:index_for_pasting[0] + current_record_length] = record
    else:  # record is longer than maximal length
        index_for_pasting = random.sample(range(current_record_length - record_length), 1)
        new_sample = record[:, index_for_pasting[0]:index_for_pasting[0] + record_length]
    return new_sample


def normalization(record):
    sample = record
    for i, strip in enumerate(sample):
        max_ = np.max(strip)
        min_ = np.min(strip)
        if max_ - min_ == 0:
            sample[i] = strip
        else:
            sample[i] = (strip - min_) / (max_ - min_)
    return sample


def run_12ECG_classifier(data,header_data,loaded_model):

    #########################################
    device = 'cpu'
    sample = np.array(data)
    sample = augmentation_algorithm(sample,record_length=30000)
    sample = normalization(sample)    
    batch_size = 1
    x = torch.from_numpy(sample).float().unsqueeze(0).to(device)
    out = loaded_model[0](x).reshape((batch_size, -1))
    out = loaded_model[1](out)
    out = torch.nn.functional.softmax(out,dim=1)
    current_score = out.detach().cpu().numpy().squeeze()
    current_label = current_score> 0.5
    current_label = current_label.astype(np.int)
    classes = loaded_model[2]['Code'].tolist()
    classes = map(str,classes)
    ######################################### EXAMPLE
    # Use your classifier here to obtain a label and score for each class.
    """
    model = loaded_model['model']
    imputer = loaded_model['imputer']
    classes = loaded_model['classes']

    features=np.asarray(get_12ECG_features(data,header_data))
    feats_reshape = features.reshape(1, -1)
    feats_reshape = imputer.transform(feats_reshape)
    current_label = model.predict(feats_reshape)[0]
    current_label=current_label.astype(int)
    current_score = model.predict_proba(feats_reshape)
    current_score=np.asarray(current_score)
    current_score=current_score[:,0,1]
    """
    ###########################################  END OF EXAMPLE
    return current_label, current_score,classes

def load_12ECG_model_legacy(input_directory):
    # load the model from disk 
    f_out='finalized_model.sav'
    filename = os.path.join(input_directory,f_out)

    loaded_model = joblib.load(filename)

    return loaded_model


def load_12ECG_model(input_directory):
    # load the model from disk 
    target_dir= os.path.dirname(os.path.abspath(__file__))
    join(target_dir, os.path.join(target_dir,'classifier.pt'), int(100e6))

    in_channels = 12 
    input_length = 3e4
    num_of_classes = 27
    fc_hidden_dims = [2**4]*2
    hidden_channels = [2**(4 + i) for i in range(10)]
    kernel_lengths = [3]*10
    dropout = None
    stride = 2
    dilation = 1
    batch_norm = True        
    encoder = ConvNet1d(in_channels, hidden_channels, kernel_lengths, dropout, stride, dilation, batch_norm)

    in_dim = encoder.out_dim(input_length)
    classifier = SimpleFFN(in_dim, num_of_classes, fc_hidden_dims)  
    f_out='classifier.pt'
    f = os.path.join(input_directory,f_out)      
    saved_state = torch.load(f, map_location='cpu')
    encoder.load_state_dict(saved_state['model_0_state'])
    classifier.load_state_dict(saved_state['model_1_state'])    
    labels = pd.read_csv(os.path.join(input_directory,'classes_lookup_table.csv'))
    loaded_model = [encoder,classifier,labels]
    return loaded_model


if __name__ == '__main__':
    print('Start ')
    loaded_model = load_12ECG_model(os.getcwd())
    print('End ')
