import os
import pickle
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from random import shuffle, randrange, choices
from nilearn import image, maskers, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from scipy.io import loadmat
import re
def read_netts(file_path):
    with open(file_path, 'r') as file:

        netts_list=[]
        for line in file:
           
        # 处理每一行
           ts_list = str(line).split()
           ts_float = [float(value) for value in ts_list]
           netts_list.append(ts_float)
        netts_matrix = np.array(netts_list)
        return netts_matrix.T
    
def read_sc(file_path):

    sc_dict = loadmat(file_path)
    sc_matrix = sc_dict['dataTable']
    percentile_values = np.percentile(sc_matrix, 85, axis=1)
    sc_result = np.where(sc_matrix >= percentile_values[:, np.newaxis], 1, 0)
    return sc_result

def is_in_sc(dir_path, file_name):
    subject_id = file_name.split('_')[1]
    sc_file_name = subject_id + '.mat'
    file_path = os.path.join(dir_path, sc_file_name)
    return os.path.isfile(file_path), file_path


class DatasetABIDE(torch.utils.data.Dataset):
    def __init__(self, sourcedir, dynamic_length=None, k_fold=None, task='label'):
        super().__init__()
        self.filename = 'abide-246-rest'
        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.task = task
        if os.path.isfile(os.path.join(sourcedir, 'processed', f'{self.filename}.pth')):
            self.timeseries_list, self.label_list = torch.load(os.path.join(sourcedir, 'processed',f'{self.filename}.pth'))
        else:
            self.timeseries_list = []
            self.label_list = []
        
            netts_list = [f for f in os.listdir(os.path.join(sourcedir, 'ABIDE')) if f.endswith('netts')]
            netts_list.sort()
            ad_data = pd.read_csv(os.path.join(sourcedir, 'abide_phenotypic_848.csv'))
            for subject in tqdm(netts_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):
                subject_path = os.path.join(sourcedir, 'ABIDE', subject)
                subject_id = subject.split('_')[1] + '_' + subject.split('_')[2]
                if subject_id not in ad_data['SUB_ID'].values:
                    continue
                label = ad_data[ad_data['SUB_ID'] == subject_id][self.task].values[0].astype(int)
                gender = ad_data[ad_data['SUB_ID'] == subject_id]['gender'].values[0].astype(int)
                timeseries = read_netts(subject_path)
                if timeseries.shape[1] != 246:
                    continue
                
                self.timeseries_list.append(timeseries)
                label -= 1 
                self.label_list.append(label)
            torch.save((self.timeseries_list, self.label_list), os.path.join(sourcedir, 'processed', f'{self.filename}.pth'))
        if type(k_fold) is type(None):
            k_fold = 0
        if k_fold >1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None
        self.folds = list(range(k_fold))
        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None

    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.timeseries_list, self.label_list))[fold]
        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False

    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.dynamic_length is None:
            sampling_init = randrange(len(timeseries)-self.dynamic_length)
            timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]
        label = self.label_list[self.fold_idx[idx]]
        
        return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label)}




class DatasetADHD(torch.utils.data.Dataset):
    def __init__(self, sourcedir, dynamic_length=None, k_fold=None, task='label'):
        super().__init__()
        self.filename = 'adhd-246-rest'
        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.task = task
        if os.path.isfile(os.path.join(sourcedir, 'processed', f'{self.filename}.pth')):
            self.timeseries_list, self.label_list = torch.load(os.path.join(sourcedir, 'processed',f'{self.filename}.pth'))
        else:
            self.timeseries_list = []
            self.label_list = []
            sub_list = []
            shape_list = []
            netts_list = [f for f in os.listdir(os.path.join(sourcedir, 'ADHD')) if f.endswith('netts')]
            netts_list.sort()
            ad_data = pd.read_csv(os.path.join(sourcedir, 'adhd_phenotypic_575_site_new.csv'))
            for subject in tqdm(netts_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):
                subject_path = os.path.join(sourcedir, 'ADHD', subject)
                subject_id = subject.split('_')[1]
                if subject_id not in ad_data['SUB_ID'].values:
                    continue
                label = ad_data[ad_data['SUB_ID'] == subject_id][self.task].values[0].astype(int)
                timeseries = read_netts(subject_path)
                if timeseries.shape[1] != 246:
                    sub_list.append(subject_id)
                    shape_list.append(timeseries.shape)
                    continue
                self.timeseries_list.append(timeseries)
                self.label_list.append(label)
            torch.save((self.timeseries_list, self.label_list), os.path.join(sourcedir, 'processed', f'{self.filename}.pth'))
        if type(k_fold) is type(None):
            k_fold = 0
        if k_fold >1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None
        self.folds = list(range(k_fold))
        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None

    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.timeseries_list, self.label_list))[fold]
        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False

    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.dynamic_length is None:
            sampling_init = randrange(len(timeseries)-self.dynamic_length)
            timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]
        label = self.label_list[self.fold_idx[idx]]
       
        return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label)}



class DatasetABIDEII(torch.utils.data.Dataset):
    def __init__(self, sourcedir, dynamic_length=None, k_fold=None, task='dx_group'):
        super().__init__()
        self.filename = 'abide-246-snf_15'
        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.task = task
        if os.path.isfile(os.path.join(sourcedir, 'processed', f'{self.filename}.pth')):
            self.timeseries_list, self.sc_list, self.label_list = torch.load(os.path.join(sourcedir, 'processed',f'{self.filename}.pth'))
        else:
            self.timeseries_list = []
            self.label_list = []
            self.sc_list = []
            sub_list = []
            shape_list = []
            netts_list = [f for f in os.listdir(os.path.join(sourcedir, 'ABIDE_II','ABIDE_REST')) if f.endswith('netts')]
            netts_list.sort()
            ab_data = pd.read_excel(os.path.join(sourcedir, 'ABIDE_II', 'ABIDE_2_participants.xlsx'))
            for subject in tqdm(netts_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):
                dir_path = os.path.join(sourcedir, 'ABIDE_II', 'ABIDE_SNF')
                exist, sc_path = is_in_sc(dir_path,subject)
                if exist:
                    subject_path = os.path.join(sourcedir, 'ABIDE_II','ABIDE_REST', subject)
                    subject_id = re.search(r'\d+', subject.split('_')[1]).group()
                    label = ab_data[ab_data['participant_id'] == int(subject_id)][self.task].values[0].astype(int)
                    timeseries = read_netts(subject_path)
                    if timeseries.shape[1] != 246:
                        sub_list.append(subject_id)
                        shape_list.append(timeseries.shape)
                        continue
                    sc_matrix = read_sc(sc_path)
                    self.timeseries_list.append(timeseries)
                    label -= 1
                    self.label_list.append(label)
                    self.sc_list.append(sc_matrix)
            print(self.label_list.count(0))
            print(self.label_list.count(1))
            torch.save((self.timeseries_list, self.sc_list, self.label_list), os.path.join(sourcedir, 'processed', f'{self.filename}.pth'))
            # torch.save((self.timeseries_list, self.label_list), os.path.join(sourcedir, 'processed', f'{self.filename}.pth'))
        if type(k_fold) is type(None):
            k_fold = 0
        if k_fold >1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None
        self.folds = list(range(k_fold))
        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None

    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.timeseries_list, self.label_list))[fold]
        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False

    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        sc_matrix = self.sc_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.dynamic_length is None:
            if self.train:
                sampling_init = randrange(len(timeseries)-self.dynamic_length)
                timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]
        label = self.label_list[self.fold_idx[idx]]
       
        return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'sc_matrix': torch.tensor(sc_matrix, dtype=torch.float32), 'label': torch.tensor(label)}
        # return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label)}
    
class DatasetDS246(torch.utils.data.Dataset):
    def __init__(self, sourcedir, dynamic_length=None, k_fold=None, task='label',ds=1):
        super().__init__()
        self.filename = 'ds-246-'+ str(ds) + '-snf'
        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.task = task
        if os.path.isfile(os.path.join(sourcedir, 'processed', f'{self.filename}.pth')):
            self.timeseries_list, self.sc_list, self.label_list = torch.load(os.path.join(sourcedir, 'processed',f'{self.filename}.pth'))
        else:
            self.timeseries_list = []
            self.label_list = []
            self.sc_list = []
            sub_list = []
            shape_list = []
            netts_list = [f for f in os.listdir(os.path.join(sourcedir, 'DS00030')) if f.endswith('netts')]
            netts_list.sort()
            ds_data = pd.read_csv(os.path.join(sourcedir, 'ds0030_phenotypic_261.csv'))
            for subject in tqdm(netts_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):
                # dir_path = os.path.join(sourcedir, 'DS00030_SC', 'DS00030_CNP','CNP_SC')
                dir_path = os.path.join(sourcedir, 'DS00030_SC','DS00030_CNP','CNP_SC')
                exist, sc_path = is_in_sc(dir_path,subject)
                if exist:
                    subject_path = os.path.join(sourcedir, 'DS00030', subject)
                    subject_id = subject.split('_')[1]
                    label = ds_data[ds_data['SUB_ID'] == subject_id][self.task].values[0].astype(int)
                    timeseries = read_netts(subject_path)
                    if timeseries.shape[1] != 246:
                        # sub_list.append(subject_id)
                        # shape_list.append(timeseries.shape)
                        continue
                    if label == 0 or label == ds:
                        sc_matrix = read_sc(sc_path)
                        self.timeseries_list.append(timeseries)
                        if label==1:
                            print(subject_id)
                        if label>1:
                            label = 1
                        sub_list.append(subject_id)
                        self.label_list.append(label)
                        self.sc_list.append(sc_matrix)

            print(self.label_list.count(0))
            print(self.label_list.count(1))
            print(sub_list)
            torch.save((self.timeseries_list, self.sc_list, self.label_list), os.path.join(sourcedir, 'processed', f'{self.filename}.pth'))
            # torch.save((self.timeseries_list, self.label_list), os.path.join(sourcedir, 'processed', f'{self.filename}.pth'))
        if type(k_fold) is type(None):
            k_fold = 0
        if k_fold >1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None
        self.folds = list(range(k_fold))
        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None

    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.timeseries_list, self.label_list))[fold]
        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False

    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        sc_matrix = self.sc_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.dynamic_length is None:
            if self.train:
                sampling_init = randrange(len(timeseries)-self.dynamic_length)
                timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]
        label = self.label_list[self.fold_idx[idx]]
       

        return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'sc_matrix': torch.tensor(sc_matrix, dtype=torch.float32), 'label': torch.tensor(label)}
        # return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label)}
    
class DatasetSC246(torch.utils.data.Dataset):
    def __init__(self, sourcedir, dynamic_length=None, k_fold=None):
        super().__init__()
        self.filename = 'sc-246'
        task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATION': 232, 'SOCIAL': 274, 'WM': 405}
        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.task_list = list(task_timepoints.keys())
        self.task_list.sort()
        print(self.task_list)

        if os.path.isfile(os.path.join(sourcedir, 'processed', f'{self.filename}.pth')):
            self.timeseries_list, self.sc_list, self.label_list = torch.load(os.path.join(sourcedir, 'processed',f'{self.filename}.pth'))
        else:
            self.timeseries_list = []
            self.label_list = []
            self.sc_list = []
            for task in self.task_list:
                netts_list = [f for f in os.listdir(os.path.join(sourcedir, 'TASK', task)) if f.endswith('netts')]
                netts_list.sort()
                for subject in tqdm(netts_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):
                    dir_path = os.path.join(sourcedir, 'HCP_SC')
                    exist, sc_path = is_in_sc(dir_path,subject)
                    if exist:
                       subject_path = os.path.join(sourcedir, 'TASK', task, subject)
                       timeseries = read_netts(subject_path)
                       sc_matrix = read_sc(sc_path)
                       if not len(timeseries)==task_timepoints[task]:
                          
                          print(f"subject: {subject} "+ f"task: {task} "+f"short timeseries: {len(timeseries)}")
                          continue
                       self.timeseries_list.append(timeseries)
                       self.label_list.append(task)
                       self.sc_list.append(sc_matrix)
            torch.save((self.timeseries_list, self.sc_list, self.label_list), os.path.join(sourcedir, 'processed', f'{self.filename}.pth'))
        if type(k_fold) is type(None):
            k_fold = 0
        if k_fold > 1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None
        self.folds = list(range(k_fold))
        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None

    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.timeseries_list, self.label_list))[fold]
        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False

    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        sc_matrix = self.sc_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.dynamic_length is None:
            if self.train:
                sampling_init = randrange(len(timeseries)-self.dynamic_length)
                timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]
        task = self.label_list[self.fold_idx[idx]]
        for task_idx, _task in enumerate(self.task_list):
            if task == _task:
                label = task_idx

        return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'sc_matrix': torch.tensor(sc_matrix, dtype=torch.float32), 'label': torch.tensor(label)}
    
class DatasetHCP246(torch.utils.data.Dataset):

    def __init__(self, sourcedir, dynamic_length=None, k_fold=None):
        super().__init__()
        self.filename = 'hcp-246'
        task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATION': 232, 'SOCIAL': 274, 'WM': 405}
        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.task_list = list(task_timepoints.keys())
        self.task_list.sort()
        print(self.task_list)

        if os.path.isfile(os.path.join(sourcedir, 'processed', f'{self.filename}.pth')):
            self.timeseries_list, self.label_list = torch.load(os.path.join(sourcedir, 'processed',f'{self.filename}.pth'))
        else:
            self.timeseries_list = []
            self.label_list = []
            for task in self.task_list:
                netts_list = [f for f in os.listdir(os.path.join(sourcedir, 'TASK', task)) if f.endswith('netts')]
                netts_list.sort()
                for subject in tqdm(netts_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):
                    subject_path = os.path.join(sourcedir, 'TASK', task, subject)
                    timeseries = read_netts(subject_path)
                    if not len(timeseries)==task_timepoints[task]:
                        print(f"subject: {subject} "+ f"task: {task} "+f"short timeseries: {len(timeseries)}")
                        continue
                    self.timeseries_list.append(timeseries)
                    self.label_list.append(task)
            torch.save((self.timeseries_list, self.label_list), os.path.join(sourcedir, 'processed', f'{self.filename}.pth'))
        if type(k_fold) is type(None):
            k_fold = 0
        if k_fold > 1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None
        self.folds = list(range(k_fold))
        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None

    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)


    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        train_idx, test_idx = list(self.k_fold.split(self.timeseries_list, self.label_list))[fold]
        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False

    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.dynamic_length is None:
            if self.train:
                sampling_init = randrange(len(timeseries)-self.dynamic_length)
                timeseries = timeseries[sampling_init:sampling_init+self.dynamic_length]
        task = self.label_list[self.fold_idx[idx]]

        for task_idx, _task in enumerate(self.task_list):
            if task == _task:
                label = task_idx

        return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label)}


