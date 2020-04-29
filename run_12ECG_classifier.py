#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
import torch
import models
import training
import torch.optim as optim
import torch.nn as nn
from training import Ecg12LeadNetTrainerBinary
from torch.utils.data import DataLoader


def correct_corrupted_record(corrupted_record,num_of_seconds):
    corrected_record=[]
    total_num_of_seconds=0.0
    num_of_times_to_concatenate = int(np.ceil(10/num_of_seconds))
    for concat_iter in range(num_of_times_to_concatenate):
        total_num_of_seconds+=num_of_seconds
        if concat_iter==0:
            corrected_record=corrupted_record
        else:
            corrected_record=np.concatenate((corrected_record,corrected_record),axis=1)
    return (corrected_record, num_of_seconds)


def split_data_to_batches(data,header_data, to_normalize=True):
    split_data=[]
    split_header=header_data[0].split(" ")
    sample_rate=float(split_header[2])
    record_length=int(split_header[3])
    num_of_seconds=record_length//sample_rate
    if num_of_seconds<10:
        data, num_of_seconds = correct_corrupted_record(data,num_of_seconds)
    number_of_split_records=num_of_seconds//2.5
    for cntr in range(int(number_of_split_records)-1):
        short_records=data[:,int(cntr*2.5*sample_rate):int((cntr+1)*2.5*sample_rate)]
        NANS_cntr_short=np.count_nonzero(np.isnan(short_records))
        if NANS_cntr_short:
            print('Found NAN value in short records')
        norm_factor=(np.amax(short_records,1)-np.amin(short_records,1))
        if to_normalize:
            norm_factor = np.where(norm_factor==0.0,1.0,norm_factor)
            short_records=np.asarray(short_records)/ norm_factor[:,np.newaxis]
        try:
            long_record=data[1,int(cntr//4*sample_rate*2.5):int((cntr//4+4)*sample_rate*2.5)]
            NANS_cntr_long=np.count_nonzero(np.isnan(long_record))
            if NANS_cntr_long:
                print('Found NAN value in long record')
        except:
            long_record=data[1,int((cntr//4-4)*sample_rate*2.5):int((cntr//4)*sample_rate*2.5)]
        long_record=np.asarray(long_record)
        if to_normalize:
            long_record=long_record/(max(long_record)-min(long_record))
        record=(short_records, long_record)
        split_data.append(record)
    return split_data


def run_12ECG_classifier(data,header_data,classes,model):
    lookup_list_from_Chinese_Challenge=[1,2,3,0,5,6,4,7,8]   #Chinese Challenge does not correspond to this one
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    split_records=split_data_to_batches(data,header_data)

    # Use your classifier here to obtain a label and score for each class. 
    # features=np.asarray(get_12ECG_features(data,header_data))
    # feats_reshape = features.reshape(1,-1)
    # label = model.predict(feats_reshape)
    # score = model.predict_proba(feats_reshape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # current_label[label] = 1
    batch_size = len(split_records)
    num_epochs=0
    classification_results=[]
    for split_record in(split_records):
        classification_result=[]
        for ii,sl in enumerate(split_record[0]):
            split_record[0][ii]=np.nan_to_num(split_record[0][ii], nan=0.0)
        x = (torch.from_numpy(split_record[0]).float().unsqueeze(0).to(device),torch.from_numpy(split_record[1]).float().unsqueeze(0).unsqueeze(0).to(device))
        for cntr in range(9):
            classification_result.append((model[cntr].forward(x)).data.cpu().numpy())
        Nans=np.argwhere(np.isnan(classification_result))
        if len(Nans)==0:
            classification_results.append(classification_result)
    classification_results=np.asarray(classification_results)
    # classification_results=np.mean(classification_results,axis=0)
    classification_results=np.max(classification_results,axis=0)
    current_label=np.asarray(classification_results>0,dtype=int).squeeze()
    if sum(current_label)==0:
        current_label[np.argmax(classification_results)]=1
    current_score=np.squeeze(classification_results)
    current_score= 1/(1 + np.exp(-current_score)) # Sigmoid -> to bring to (0,1)
    # current_label=current_label[lookup_list_from_Chinese_Challenge]
    return current_label, current_score

def load_12ECG_model():
    number_of_categories= 9
    # load the model from disk 
    kernel_size =17
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    short_hidden_channels = [16, 32, 64, 128, 256, 512]
    long_hidden_channels = [4, 8, 16, 32, 64, 128, 256, 512]
    short_kernel_lengths = [kernel_size]*6
    long_kernel_lengths = [kernel_size]*8
    # which tricks to use: dropout, stride, batch normalization and dilation
    short_dropout = 0.5
    long_dropout = 0.5
    short_stride = 2
    long_stride = 2
    short_dilation = 1
    long_dilation = 1
    short_batch_norm = True
    long_batch_norm = True
    # enter input length here
    short_input_length = 1250
    long_input_length = 5000
    # FC net structure:
    # num of hidden units in every FC layer
    fc_hidden_dims = [128]
    # num of output classess
    num_of_classes = 2 
    model=[]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """  Original order: Normal, AF, I-AVB, LBBB, RBBB, PAC, PVC, STD, STE
            New order: AF, I-AVB, LBBB,	Normal,	PAC, PVC, RBBB,	STD, STE
    """
    lookup_list_from_Chinese_Challenge=[1,2,3,0,5,6,4,7,8]   #

    for categ in range(number_of_categories):   
        model.append( models.Ecg12LeadNet(short_hidden_channels, long_hidden_channels,
                                short_kernel_lengths, long_kernel_lengths,
                                fc_hidden_dims,
                                short_dropout, long_dropout,
                                short_stride, long_stride,
                                short_dilation, long_dilation,
                                short_batch_norm, long_batch_norm,
                                short_input_length, long_input_length,
                                num_of_classes).to(device) )
        model[categ].train(False)  # set evaluation (test) mode        
        print('Uploaded model')
        print(model[categ])
        checkpoint_filename='Ecg12LeadNetDigitizedToClass__'+str(lookup_list_from_Chinese_Challenge[categ])+'.pt'
        print(f'*** Loading checkpoint file {checkpoint_filename}')
        saved_state = torch.load(checkpoint_filename,
                                    map_location=device)
        model[categ].load_state_dict(saved_state['model_state'])            
    return model
