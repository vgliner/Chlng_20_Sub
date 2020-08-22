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


"""
Notation of multiclass_to_binary_type: 
[-1] Return multiclass [0] Normal, [1] AF, [2] I-AVB, [3] LBBB, [4] RBBB, [5] PAC, [6] PVC, [7] STD, [8] STE
"""

PRINT_FLAG = False


class ECG_Multilead_Dataset_long_records(Dataset):
    def __init__(self, root_dir=None, transform=None, multiclass=False,
                 binary_class_type=1, apply_aurmentation=True, random_augmentation=True,
                 augmentation_method=None, record_length=60, to_normalize=True, Uploading_method='HDD',
                 old_format= False):
        #                record_length [sec]
        #   Uploading_method = 'HDD'\'RAM'\'Cache'
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
        # Chinese database constructor
        # self.chinese_db_path='Database_for_augmentation.hdf5'
        # self.chinese_file=h5py.File(self.chinese_db_path, 'r')
        # self.chinese_keys=self.chinese_file.keys()        
        # self.chinese_db_length=self.chinese_file['length_chinese_db']
        # self.chinese_sample_rate = 500
        # self.chinese_maximal_length= self.chinese_sample_rate * self.record_length
        # All in database
        self.old_format = old_format
        if self.old_format:
            dataset_filename= 'All_in_dataset.hdf5'
        else:
            dataset_filename= 'All_in_dataset_new.hdf5'

        if self.root_dir is None:
            paths = Utils.read_config_file()
            self.all_in_database_path = os.path.join(paths[0], dataset_filename)
        else:
            self.all_in_database_path = self.root_dir + dataset_filename

        self.all_in_sample_rate = 500
        self.all_in_maximal_length = self.all_in_sample_rate * self.record_length
        self.all_in_file = h5py.File(self.all_in_database_path, 'r')
        self.all_in_db_length = self.all_in_file['lengths'][()]
        self.all_in_keys = self.all_in_file.keys()
        self.statistics = self.all_in_file['statistics'][()]
        self.Uploaded_data = {}
        if self.Uploading_method == 'RAM':
            try:
                from tqdm import tqdm
                for idx in tqdm(range(np.sum(self.all_in_db_length, dtype=int))):
                    n1 = self.all_in_file[str(idx + 1) + '_d']
                    n2 = self.all_in_file[str(idx + 1) + '_c']
                    sample = [np.array(n1), np.array(n2)]
                    self.Uploaded_data[str(idx)] = sample
            except ImportError:
                for idx in range(np.sum(self.all_in_db_length, dtype=int)):
                    n1 = self.all_in_file[str(idx + 1) + '_d']
                    n2 = self.all_in_file[str(idx + 1) + '_c']
                    sample = [np.array(n1), np.array(n2)]
                    self.Uploaded_data[str(idx)] = sample
            print(f' {np.sum(self.all_in_db_length, dtype=int)} data records were uploaded to RAM')

    def __len__(self):
        # self.database_length = int(np.array(self.chinese_db_length))
        self.database_length = np.sum(self.all_in_db_length, dtype=int)
        return self.database_length

    def __getitem__(self, idx):
        # n1= self.chinese_file[str(idx)+'_d']
        # n2= self.chinese_file[str(idx)+'_c']
        if self.Uploading_method == 'HDD':
            n1 = self.all_in_file[str(idx + 1) + '_d']
            n2 = self.all_in_file[str(idx + 1) + '_c']
            sample = [np.array(n1), np.array(n2)]
        elif self.Uploading_method == 'RAM':
            sample = self.Uploaded_data[str(idx)]
        else:  # CACHE
            if str(idx) not in self.Uploaded_data.keys():
                n1 = self.all_in_file[str(idx + 1) + '_d']
                n2 = self.all_in_file[str(idx + 1) + '_c']
                sample = [np.array(n1), np.array(n2)]
                self.Uploaded_data[str(idx)] = sample
            else:
                sample = self.Uploaded_data[str(idx)]

        if self.apply_aurmentation:
            sample = self.augmentation_algorithm(sample)
        if self.to_normalize:
            sample = self.normalization(sample)
        if self.binary_class_type >= 0:
            sample[1] = sample[1][int(self.binary_class_type)]
        return sample

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


def test_dataloader():
    print('Testing  merge with Russian database')
    # target_path = r'C:\Users\noam\Desktop\ch2020' + '\\'
    ECG_dataset_test = ECG_Multilead_Dataset_long_records(transform=None, multiclass=True,  # target_path,
                                                          binary_class_type=1, random_augmentation=True,
                                                          augmentation_method=None, record_length=60,
                                                          Uploading_method='Cache')
    for i in range(1, len(ECG_dataset_test) // 20 + 1):
        testing1 = ECG_dataset_test[i]
        # ECG_dataset_test.plot(testing1)
        print(f'{i}')
    for i in range(1, len(ECG_dataset_test) // 20 + 1):
        testing1 = ECG_dataset_test[i]
        # ECG_dataset_test.plot(testing1)
        print(f'{i}')
    print(f'Type of record: {type(testing1)}')
    print(f'Database length is : {len(ECG_dataset_test)}')


def Chinese_database_creator():
    Database_path = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Databases\Chineese_database\Original' + '\\'
    files_in_folder = os.listdir(Database_path)
    raw_records_processed = 0
    db_ref_path = Database_path + 'REFERENCE.csv'
    with h5py.File("Database_for_augmentation" + ".hdf5", "w") as f:
        for file in files_in_folder:
            if file.endswith('.mat'):
                print(f'Parsing {file}')
                raw_records_processed += 1
                mat_contents = sio.loadmat(Database_path + file)
                b = mat_contents['ECG']['data'].item()
                sex = mat_contents['ECG']['sex'].item().item()
                classification = upload_classification(db_ref_path, file)
                dset = f.create_dataset(str(int(file[-8:-4])) + '_d', data=b)
                dset = f.create_dataset(str(int(file[-8:-4])) + '_c', data=classification)
        dset = f.create_dataset('length_chinese_db', data=raw_records_processed)
    print(f'Database created, {raw_records_processed} records uploaded')


def Challenge_database_creator(database_name, desired_sample_rate = 500, is_old_version= True):
    Databases_list = [
    r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Databases\Chineese_database\Original_with_codes\Training_WFDB' + '\\',
    r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Databases\Chineese_database\Original_Addition\Training_2' + '\\',
    r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Databases\StPetersburg INCART\PhysioNetChallenge2020_Training_StPetersburg' + '\\',
    r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Databases\PhysioNetChallenge2020_Training_PTB\Training_PTB' + '\\',
    r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Databases\PhysioNetChallenge2020_PTB-XL\WFDB' + '\\',
    r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Databases\PhysioNetChallenge2020_Training_E\WFDB' + '\\'
    ]
    Database_path = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Databases\Chineese_database\Original_Addition\Training_2' + '\\'
    Database_path = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Databases\StPetersburg INCART\PhysioNetChallenge2020_Training_StPetersburg' + '\\'

    if is_old_version:
        statistics = np.zeros((7, 9))
        lengths = np.zeros(7)
    else:
        statistics = np.zeros((6, 27))
        lengths = np.zeros(6)
    number_of_processed_files = 0

    with h5py.File(database_name + ".hdf5", "w") as fl:
        for database_num, database in enumerate(Databases_list):
            Database_path = database
            input_files = []
            for f in os.listdir(Database_path):
                if os.path.isfile(os.path.join(Database_path, f)) and not f.lower().startswith(
                        '.') and f.lower().endswith('mat'):
                    print(f'Processing: {f}')
                    input_files.append(f)
                    if is_old_version:
                        classification = get_classes_codes(Database_path, f)
                    else:
                        classification = get_classes_codes_new_version(Database_path, f)
                    sample_rate=  get_sample_rate(Database_path, f)                
                    statistics[database_num, :] += classification
                    number_of_processed_files += 1
                    data, header_data = load_challenge_data(Database_path + f)
                    mat_contents = sio.loadmat(Database_path + f)
                    b = np.array(mat_contents['val'], dtype=float)
                    if not (desired_sample_rate == sample_rate):
                        print('Resampling')
                        b = [ECG_resampling(b_ , desired_sample_rate=500, current_sample_rate=sample_rate) for b_ in b]
                        b = np.array(b, dtype= float)
                    # b = mat_contents['ECG']['data'].item()
                    # sex= mat_contents['ECG']['sex'].item().item()
                    dset = fl.create_dataset(str(number_of_processed_files) + '_d', data=b)
                    dset = fl.create_dataset(str(number_of_processed_files) + '_c', data=classification)
                    lengths[database_num] += 1

        if is_old_version:
            data,classifications = Parse_russian_DB()
            for cl_num, cl in enumerate(classifications):
                if np.sum(cl):
                    number_of_processed_files += 1
                    dset = fl.create_dataset(str(number_of_processed_files) + '_d', data=data[cl_num])
                    dset = fl.create_dataset(str(number_of_processed_files) + '_c', data=cl)   
                    lengths[6] += 1
                    statistics[6, :] += cl                
        dset = fl.create_dataset('lengths', data=lengths)
        dset = fl.create_dataset('statistics', data=statistics)
        
    print('Finished conversion')


def get_sample_rate(Database_path, input_file):
    f = input_file
    g = f.replace('.mat', '.hea')
    input_file = os.path.join(Database_path, g)
    with open(input_file, 'r') as f:
        for line_num, lines in enumerate(f):
            if line_num== 0 :
                x = lines.split()
    return int(x[2])
    

def get_classes_codes(Database_path, input_file):
    # [0] Normal, [1] AF, [2] I-AVB, [3] LBBB, [4] RBBB, [5] PAC, [6] PVC, [7] STD, [8] STE
    classes = []
    f = input_file
    g = f.replace('.mat', '.hea')
    input_file = os.path.join(Database_path, g)
    with open(input_file, 'r') as f:
        for lines in f:
            if lines.startswith('#Dx'):
                print(lines)
                tmp = lines.split(': ')[1].split(',')
                for c in tmp:
                    classes.append(c.strip())
    classification = np.zeros((9,))
    for t in tmp:
        if '426783006' in t:
            classification[0] = 1.0
        if ('164889003' in t) or ('195080001' in t):
            classification[1] = 1.0
            classification[0] = 0.0
        if '270492004' in t:
            classification[2] = 1.0
            classification[0] = 0.0
        if '164909002' in t:
            classification[3] = 1.0
            classification[0] = 0.0
        if '59118001' in t:
            classification[4] = 1.0
            classification[0] = 0.0
        if '284470004' in t:
            classification[5] = 1.0
            classification[0] = 0.0
        if ("164884008" in t) or ('427172004' in t):
            classification[6] = 1.0
            classification[0] = 0.0
        if '429622005' in t:
            classification[7] = 1.0
            classification[0] = 0.0
        if '164931005' in t:
            classification[8] = 1.0
            classification[0] = 0.0
    return classification

def get_classes_codes_new_version(Database_path, input_file):
    # 0	1st degree av block
    # 1	atrial fibrillation
    # 2	atrial flutter
    # 3	bradycardia
    # 4	complete right bundle branch block
    # 5	incomplete right bundle branch block
    # 6	left anterior fascicular block
    # 7	left axis deviation
    # 8	left bundle branch block
    # 9	low qrs voltages
    # 10	nonspecific intraventricular conduction disorder
    # 11	pacing rhythm
    # 12	premature atrial contraction
    # 13	premature ventricular contractions
    # 14	prolonged pr interval
    # 15	prolonged qt interval
    # 16	qwave abnormal
    # 17	right axis deviation
    # 18	right bundle branch block
    # 19	sinus arrhythmia
    # 20	sinus bradycardia
    # 21	sinus rhythm
    # 22	sinus tachycardia
    # 23	supraventricular premature beats
    # 24	t wave abnormal
    # 25	t wave inversion
    # 26	ventricular premature beats


    classes = []
    f = input_file
    g = f.replace('.mat', '.hea')
    input_file = os.path.join(Database_path, g)
    with open(input_file, 'r') as f:
        for lines in f:
            if lines.startswith('#Dx'):
                print(lines)
                tmp = lines.split(': ')[1].split(',')
                for c in tmp:
                    classes.append(c.strip())
    classification = np.zeros((27,))
    for t in tmp:
        if '426783006' in t:
            classification[21] = 1.0
        if ('270492004' in t):
            classification[0] = 1.0
            classification[21] = 0.0
        if '164889003' in t:
            classification[1] = 1.0
            classification[21] = 0.0
        if '164890007' in t:
            classification[2] = 1.0
            classification[21] = 0.0
        if '426627000' in t:
            classification[3] = 1.0
            classification[21] = 0.0
        if ('713427006' in t) or ('59118001' in t):
            classification[4] = 1.0
            classification[21] = 0.0
        if '713426002' in t:
            classification[5] = 1.0
            classification[21] = 0.0         
        if '445118002' in t:
            classification[6] = 1.0
            classification[21] = 0.0   
        if '39732003' in t:
            classification[7] = 1.0
            classification[21] = 0.0   
        if '164909002' in t:
            classification[8] = 1.0
            classification[21] = 0.0 
        if '251146004' in t:
            classification[9] = 1.0
            classification[21] = 0.0 
        if '698252002' in t:
            classification[10] = 1.0
            classification[21] = 0.0 
        if '10370003' in t:
            classification[11] = 1.0
            classification[21] = 0.0 
        if ('284470004' in t) or ('63593006' in t):
            classification[12] = 1.0
            classification[21] = 0.0 
        if ('427172004' in t) or ('17338001' in t):
            classification[13] = 1.0
            classification[21] = 0.0             
        if '164947007' in t:
            classification[14] = 1.0
            classification[21] = 0.0     
        if '111975006' in t:
            classification[15] = 1.0
            classification[21] = 0.0    
        if '164917005' in t:
            classification[16] = 1.0
            classification[21] = 0.0        
        if '47665007' in t:
            classification[17] = 1.0
            classification[21] = 0.0     
        if ('59118001' in t) or ('713427006' in t):
            classification[18] = 1.0
            classification[21] = 0.0               
        if '427393009' in t:
            classification[19] = 1.0
            classification[21] = 0.0      
        if '426177001' in t:
            classification[20] = 1.0
            classification[21] = 0.0      
        if '427084000' in t:
            classification[22] = 1.0
            classification[21] = 0.0          
        if ('63593006' in t) or ('284470004' in t):
            classification[23] = 1.0
            classification[21] = 0.0    
        if '164934002' in t:
            classification[24] = 1.0
            classification[21] = 0.0 
        if '59931005' in t:
            classification[25] = 1.0
            classification[21] = 0.0    
        if ('17338001' in t) or ('427172004' in t):
            classification[26] = 1.0
            classification[21] = 0.0                                                                                                                  
    return classification


def get_classes(input_directory, files):
    classes = set()
    for f in files:
        g = f.replace('.mat', '.hea')
        input_file = os.path.join(input_directory, g)
        with open(input_file, 'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())

    return sorted(classes)


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat', '.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file, 'r') as f:
        header_data = f.readlines()
    return data, header_data


def upload_classification(DB_ref_path, required_entry):
    # print(DB_ref_path)
    data = pd.read_csv(DB_ref_path)
    data.head()
    _entries = data.Recording.to_list()
    _entry_number_in_list = _entries.index(required_entry[0:5])
    _values = data.values[_entry_number_in_list, :]
    classification = np.zeros(9)
    for val in range(1, 4):
        if _values[val] < 10:
            classification[int(_values[val]) - 1] = 1
    return classification

def ECG_resampling(ECG_signal, desired_sample_rate=500, current_sample_rate=250):
    import wfdb.processing
    signal_to_interpolate=ECG_signal
    # x = np.arange(0, len(signal_to_interpolate))
    y = signal_to_interpolate
    # # ECG_Multilead_Dataset_long_records().plot_one_strip(y)
    # plt.plot(y)
    # plt.show()    
    interpolated = wfdb.processing.resample_sig(y, current_sample_rate, desired_sample_rate)
    interpolated_before=interpolated
    order = 10
    if current_sample_rate < desired_sample_rate:
        A= butter_lowpass_filter(interpolated[0], current_sample_rate/2.5,desired_sample_rate, order)    
        return A
    # f = interpolate.interp1d(x, y)
    # xnew = np.arange(0, len(signal_to_interpolate)-np.finfo(float).eps, 0.1)  #current_sample_rate/desired_sample_rate
    # ynew = f(xnew)   # use interpolation function returned by `interp1d`
    # ECG_Multilead_Dataset_long_records().plot_one_strip(ynew)

    # ECG_Multilead_Dataset_long_records().plot_one_strip(y)
    # ECG_Multilead_Dataset_long_records().plot_one_strip(interpolated_before[0])
    # ECG_Multilead_Dataset_long_records().plot_one_strip(interpolated[0])    
    return interpolated[0]


def interpolation_tester():
    print('Testing interpolation')
    ECG_dataset_test = ECG_Multilead_Dataset_long_records(transform=None, multiclass=False,  # target_path,
                                                          binary_class_type=1, random_augmentation=True,
                                                          augmentation_method=None, record_length=60,
                                                          Uploading_method='Cache')    
    testing1 = ECG_dataset_test[0]
    ECG_resampling(testing1, desired_sample_rate=500, current_sample_rate=250)

def Parse_russian_DB():
    Russian_database_path=[r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Databases\Russian\Only_10000\Only_10000\files'+ '\\',
    r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Databases\Russians_ver2\2020_03_26'+'\\']
    sum_vector=np.zeros(11)
    data=[]
    classifications=[]
    for db in Russian_database_path:
        # List files in folder 
        Database_path = Russian_database_path
        db_sample_rate = 200 # Hz
        # Filter requirements.
        order = 10
        fs = db_sample_rate       # sample rate, Hz
        cutoff = db_sample_rate/2  # desired cutoff frequency of the filter, Hz
        for path, subdirs, files in os.walk(db):
            for f in files:
            # f='17239.csv'
                if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('csv'):  
                    print(f'So far: {sum_vector}') 
                    print(f'Parsing : {f}')
                    ## PARSE DATA
                    record=[] 
                    with open(os.path.join(path, f), newline='\n') as csvfile:
                        reader = csv.reader(csvfile, delimiter=';')
                        for ii,row in enumerate(reader):
                            if ii == 0: # Skip header
                                continue
                            record.append([int(i) for i in row])
                    record = np.array(record)
                    record = np.transpose(record)
                    record = record[1:,]
                    print('Resampling')
                    b = [butter_lowpass_filter(ECG_resampling(b_ , desired_sample_rate=500, current_sample_rate=db_sample_rate), db_sample_rate/2.5,
                        500, order) for b_ in record]            
                    # b = [ECG_resampling(b_ , desired_sample_rate=500, current_sample_rate=db_sample_rate) for b_ in record]
                    b = np.array(b, dtype= float)      
                    # y = butter_lowpass_filter(b[1], db_sample_rate/2, 500, order)            
                    # ECG_Multilead_Dataset_long_records().plot_one_strip(record[0])
                    # ECG_Multilead_Dataset_long_records().plot_one_strip(b[0])  
                    data.append(b)
                    ## PARSE DATA
                    new_file = f.replace('.csv', '_info.txt')
                    classification = np.zeros(9)
                    with open(os.path.join(path, new_file), encoding = 'utf-8', mode = 'r') as my_file: 
                        K=my_file.read()
                        Normal_str=['ЭКГ без существенных отклонений от нормы','существенных отклонений от нормы']
                        if if_one_of_substrings_in_string(K, Normal_str):
                            #('Синусовый ритм' in K) or K.find('Cинусовый ритм.')>0 or K.find('Синусовный ритм.')>0 or \
                            #  K.lower().find('ритм синусовый') or \
                            classification[0] = 1.0
                            print('Sinus rhythm')

                        AFIB_str=['Фибриляция предсердий','трепетание предсердий','фибриляция желудочков','фибриляция  предсердий','трепетане предсердий',
                        'фибриляция предсерлдий','трепетания предсердий','фибриляция предсержий','фибриляция предсердипй','фибриляция-трепетание предсердия',
                        'фибрилляция предсердий']
                        if if_one_of_substrings_in_string(K, AFIB_str):
                            classification[1] = 1.0
                            classification[0] = 0.0
                            sum_vector[1]+=1
                            print('AFIB')

                        IAVB_str=['A-V блокада 1 степени','a-v блокада 1 степени','a-v  блокада 1 ст','av i степени','av блокада','a-v  бло4када',
                        'a-v i степени','a-v блокада i степени','a-v блокада 1 степени','a-v  i степени','avl','аv блокада i степени',
                        'a-v - блокада 1 ст.','a-v блокада 1 ст','av блокда i степени','na-v блокада 1 ст.','av  блокада i степени',
                        'a-v блокада 1 1 степени','a-v - блокада 1 ст.','a-v блокаду 1 степени','av блокадой i степени','a-v блокаду 1 степени',
                        'a-v  блокада i степени','a-v блокада 11']
                        if if_one_of_substrings_in_string(K, IAVB_str):
                            classification[2] = 1.0   
                            classification[0] = 0.0
                            sum_vector[2]+=1
                            print('I-AVB')

                        LBBB_str=['блокада левой','левой ножки','блокада  левой','левой н.п.гиса','левой \r\nножки',
                        'левой \r\nножки','левой \r\nножки','левой \r\nножки п гиса']
                        if if_one_of_substrings_in_string(K, LBBB_str):
                            classification[3] = 1.0   
                            classification[0] = 0.0
                            sum_vector[3]+=1
                            print('LBBB')    

                        RBBB_str=['блокада правой','правой ножки','блокада праовый','блокада праов йножки п. гиса']
                        if if_one_of_substrings_in_string(K, RBBB_str):
                            classification[4] = 1.0 
                            classification[0] = 0.0
                            sum_vector[4]+=1
                            print('RBBB')       

                        PAC_str=['эктопический предсердный ритм','эктопический предсердный ритм.','Предсердный  эктопический',
                        'эктопический  ускоренный предсердный','Предсердный эктопический','наджелудочковые экстрасистолы','желудочковые экстрасистолы',
                        'желудочковые','желудочковая экстрасистола','предсердные экстрасистолы']
                        if if_one_of_substrings_in_string(K, PAC_str):
                            classification[5] = 1.0
                            classification[0] = 0.0
                            sum_vector[5]+=1
                            print('PAC')

                        PVC_str=['Нижнепредсердный эктопический ритм','Нижнепредсердный эктопический ритм.', 'эктопический  нижнепредсердный',
                        'Нижнепредсердный ритм']
                        if if_one_of_substrings_in_string(K, PVC_str):
                            classification[6] = 1.0   
                            classification[0] = 0.0
                            sum_vector[6]+=1
                            print('PVC')

                        STD_str=['депрессия сегмента st','депрессия \r\nсегментов st','депресия сегмента st','депрессия st',
                        'депрессия st']
                        if if_one_of_substrings_in_string(K, STD_str):
                            classification[7] = 1.0   
                            classification[0] = 0.0
                            sum_vector[7]+=1
                            print('STE')

                        STE_str=['элевация st','элевация сегмента st','подъем сегмента st','подъемов сегмента st','подъем сегментов \r\nst',
                        'подъем сегментов \r\nst','подъем сегментов st','элевция st','подъемом st','подъем st','подъем сегментов \r\nst',
                        'подъем  st','подъем  st']
                        if if_one_of_substrings_in_string(K, STE_str):
                            classification[8] = 1.0   
                            classification[0] = 0.0
                            sum_vector[8]+=1
                            print('STE')

                        if np.sum(classification) == 0:
                            with open('Unrecognized_log.txt', encoding = 'utf-8', mode = 'a+') as unrecognized_log:
                                unrecognized_log.write(f'{new_file} : ')
                                unrecognized_log.write(K[K.lower().find('result'):K.lower().find('sex')].lower()) 
                                unrecognized_log.write('\n') 

                            sum_vector[9]+=1

                        sum_vector[10]+=1
                        if classification[0]:
                            sum_vector[0]+=1                    
                            with open('Suspected_as_normals.txt', encoding = 'utf-8', mode = 'a+') as unrecognized_log:
                                unrecognized_log.write(f'{new_file} : ')
                                unrecognized_log.write(K[K.lower().find('result'):K.lower().find('sex')].lower()) 
                                unrecognized_log.write('\n') 
                        print(K) 
                        print(classification)
                        classifications.append(classification)    
    print('Done')                       
    return (data,classifications)          

def if_one_of_substrings_in_string(base_string, list_of_substrings):
    for substring in list_of_substrings:
        if base_string.lower().find(substring.lower())>0:
            return True
    return False
                    
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



if __name__ == "__main__":
    # print('Starting dataloader test')
    # Chinese_database_creator()    
    # test_dataloader()
    # Challenge_database_creator('All_in_dataset_new',is_old_version= False)
    interpolation_tester()
    # Parse_russian_DB()
