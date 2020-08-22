from torch.utils.data import Dataset
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import random
from scipy.io import loadmat
import Utils
from scipy import interpolate
from scipy import signal
import csv
from scipy.signal import butter, lfilter, freqz
import re
from glob import glob
import time
import pickle


"""
It contains annotations about 6 different ECGs abnormalities:
- 1st degree AV block (1dAVb);
- right bundle branch block (RBBB);
- left bundle branch block (LBBB);
- sinus bradycardia (SB);
- atrial fibrillation (AF); 
- sinus tachycardia (ST).

Notation of multiclass_to_binary_type: 
[-1] Return multiclass [0] I-AVB, [1] RBBB, [2] LBBB, [3] SB, [4] AF, [5] ST
"""

PRINT_FLAG = False


class ECG_Multilead_Dataset_Brazilian_records(Dataset):
    def __init__(self, root_dir=None, transform=None, multiclass=False,
                 binary_class_type=1, apply_aurmentation=True, random_augmentation=True,
                 augmentation_method=None, record_length=60, to_normalize=True, Uploading_method='HDD',
                 old_format= False):
        #                record_length [sec]
        #   Uploading_method = 'HDD'\'RAM'\'cache'
        super().__init__()
        self.data = []
        self.samples = None
        self.root_dir = root_dir
        self.transform = transform
        self.multiclass = multiclass
        self.binary_class_type = binary_class_type
        self.apply_aurmentation = apply_aurmentation
        self.random_augmentation = random_augmentation
        self.augmentation_method = augmentation_method
        self.database_length = 0
        self.data_mutual_sample_rate = 500
        self.record_length = record_length * self.data_mutual_sample_rate
        self.to_normalize = to_normalize
        self.Uploading_method = Uploading_method
        self.brazilian_database_path = None
        self.brazilian_annotations_path = None
        self.sample_rate = 400
        self.maximal_length = self.sample_rate * self.record_length

        if not multiclass:
            assert binary_class_type >= 0, 'Class selection is mandatory for single class classification'

        if self.root_dir is None:
            paths = Utils.read_config_file()
            self.brazilian_database_path = paths[1]
            self.brazilian_annotations_path = paths[2]
            self.brazilian_annotations_dict_path = paths[3]

        else:
            self.brazilian_database_path = self.root_dir + dataset_filename

        self.f = h5py.File(self.brazilian_database_path, "r")
        self.data_ids = np.array(self.f['id_exam'])
        self.data = self.f['signal']
        start = time.process_time()
        self.annotations = pd.read_csv(self.brazilian_annotations_path)
        end = time.process_time()
        print(f'Uploading annotations took {end-start} sec.')
        start = time.process_time()

        # Convert Data Frame to Dictionary (set_index method allows any column to be used as index)
        with open(self.brazilian_annotations_dict_path, 'rb') as handle:
            self.annotations_dict = pickle.load(handle)
        #self.annotations_dict = self.annotations.set_index('id_exam').transpose().to_dict(orient='dict')
        end = time.process_time()
        print(f'Uploading annotations dictionary took {end-start} sec.')
        print('finished')

        self.loaded_data = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if idx not in self.loaded_data.keys():
            sample = self.data[idx]
            data_id = self.data_ids[idx]
            sample = np.transpose(sample)
            annotation = self.annotations_dict[data_id]
            annotation = list(annotation.values())[3:]
            sample = (sample, annotation)
        else:
            sample = self.loaded_data[idx]

        if self.to_normalize:
            sample = self.normalization(sample)

        if self.binary_class_type >= 0 and not self.multiclass:
            sample[1] = sample[1][int(self.binary_class_type)]

        if self.multiclass:
            sample[1] = np.stack(sample[1])

        if self.Uploading_method == 'cache' and idx not in self.loaded_data.keys():
            self.loaded_data[idx] = sample

        if self.apply_aurmentation:
            sample = self.augmentation_algorithm(sample)

        return sample

    def find_annotations(self, id_to_find):
        a= list(self.annotations['id_exam']).index(id_to_find)
        return list(self.annotations.iloc[a].values[4:])

    @staticmethod
    def plot(sample):
        item_to_plot = sample[0]
        fig, axes = plt.subplots(nrows=6, ncols=2)
        fig.suptitle(np.array2string(sample[1]), fontsize=14)
        titles = ['Lead1', 'Lead2', 'Lead3', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        b = item_to_plot
        for ax, cntr in zip(axes.flatten(), range(12)):
            ax.plot(b[cntr, :], linewidth=1.0)
            ax.set(title=titles[cntr])
        plt.plot()
        plt.show()
        return

    @staticmethod
    def plot_one_strip(one_strip):
        item_to_plot = one_strip
        plt.plot(item_to_plot)
        plt.show()
        return


    def augmentation_algorithm(self, record):
        current_record_length = record[0].shape[1]
        if current_record_length == self.record_length:
            return record
        if current_record_length <= self.record_length:  # record is shorter than maximal length or similar
            new_sample = np.zeros((12, self.record_length))
            index_for_pasting = random.sample(range(self.record_length - current_record_length), 1)
            new_sample[:, index_for_pasting[0]:index_for_pasting[0] + current_record_length] = record[0]
        else:  # record is longer than maximal length
            index_for_pasting = random.sample(range(current_record_length - self.record_length), 1)
            new_sample = record[0][:, index_for_pasting[0]:index_for_pasting[0] + self.record_length]
        return [new_sample, record[1]]

    @staticmethod
    def normalization(record):
        sample = record[0]
        for i, strip in enumerate(sample):
            max_ = np.max(strip)
            min_ = np.min(strip)
            if max_ - min_ == 0:
                sample[i] = strip
            else:
                sample[i] = (strip - min_) / (max_ - min_)
        return [sample, record[1]]                


def test_Brazilian_db_dataloader():
    print('Testing Brazilian  database')
    ds = ECG_Multilead_Dataset_Brazilian_records()
    start = time.process_time()
    for record_counter in range(len(ds)):
        ds_record = ds[record_counter]
        # ds.plot(ds_record)
        if record_counter %10000 ==0:
            stop = time.process_time()
            print(f'Loaded record # {record_counter}, time : {stop-start}')
    print('Finished testing')


if __name__ == "__main__":
    test_Brazilian_db_dataloader()
