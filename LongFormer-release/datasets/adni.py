import torch
import torch.utils.data

import os

import random

import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F

class ADNIDataset:
    def __init__(self, csv_file, args):
        self.args = args
        print('Constructing dataset...')
        self.getPatientsInfo(csv_file)

    def getPatientsInfo(self,csv_path):
        if not os.path.exists(csv_path):
            raise ValueError("{} dir not found".format(csv_path))

        self._paths = []
        self._months = []
        self._labels = []
        self._ptids = []
        self.long_info = {}
        with open(csv_path) as f:
            for line in f:
                if 'ADNI' in csv_path:
                    img_path, month, label = line.split('\n')[0].split(',')
                    if self.args.n_times == 2:
                        ptid = img_path.split('/')[-3]
                        scan_time = img_path.split('/')[-2]
                        opflow_path = '.../voxelmorph_out/' + ptid + '_' + scan_time + '_flow.nii.gz'
                        if not os.path.exists(opflow_path):
                            continue

                else:
                    img_path, label = line.split('\n')[0].split(',')
                    month = 'bl'
                patient_id = img_path.split('/')[-3]
                label_map = {
                    'Nondemented': 0,
                    'Demented': 1,
                    'CN': 0,
                    'AD': 1,
                    'MCI': 2
                }
                label = label_map[label]
                if label not in [0,1]:
                    continue
                if month == 'bl':
                    month = 0
                else:
                    month = int(month[1:])
                self._ptids.append(patient_id)
                self._paths.append(img_path)
                self._months.append(month)
                self._labels.append(int(label))
                
                if patient_id not in self.long_info.keys():
                    self.long_info[patient_id] = {}
                self.long_info[patient_id][month] = img_path

        print("Constructing adni patients (size: {}) from {}".format(len(self._paths), csv_path))

    def __len__(self):
        return len(self._paths)
    
    def pad_img(self, img, size=224):
        '''pad img to square.
        '''
        x, y, z = img.shape
        img = img.unsqueeze(0).unsqueeze(0) # BCHWD
        max_size = max(x, y, z)
        new_size = (int(size*x/max_size), int(size*y/max_size), int(size*z/max_size))
        img = F.interpolate(img,size=new_size,mode='trilinear',align_corners=True)

        x,y,z = new_size
        new_im = torch.zeros((1,1,size,size,size))
        x_min = int((size - x) / 2)
        x_max = x_min + x
        y_min = int((size - y) / 2)
        y_max = y_min + y
        z_min = int((size - z) / 2)
        z_max = z_min + z
        new_im[:,:,x_min:x_max,y_min:y_max,z_min:z_max] = img
        
        return new_im.squeeze(0)
    
    def pad_img_3d(self, img, size=224):
        '''pad img to square.
        '''
        # import pdb;pdb.set_trace()
        img = img.permute(3,0,1,2)
        _, x, y, z = img.shape
        img = img.unsqueeze(0) # BCHWD
        max_size = max(x, y, z)
        new_size = (int(size*x/max_size), int(size*y/max_size), int(size*z/max_size))
        img = F.interpolate(img,size=new_size,mode='trilinear',align_corners=True)

        x,y,z = new_size
        new_im = torch.zeros((1,3,size,size,size))
        x_min = int((size - x) / 2)
        x_max = x_min + x
        y_min = int((size - y) / 2)
        y_max = y_min + y
        z_min = int((size - z) / 2)
        z_max = z_min + z
        new_im[:,:,x_min:x_max,y_min:y_max,z_min:z_max] = img
        
        return new_im.squeeze(0)
    
    def norm_img(self, img):
        return (img - img.min())/(img.max() - img.min())
    
    def preprocess(self, path):
        img = torch.FloatTensor(sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(float))
        img = self.norm_img(img)
        img = self.pad_img(img) 
        return img
    
    def preprocess_3dim(self, path):
        img = torch.FloatTensor(sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(float))
        img = self.norm_img(img)
        img = self.pad_img_3d(img) 
        return img


    def __getitem__(self, idx):

       
        label = self._labels[idx]
        path = self._paths[idx].replace('data2','data')
        month = self._months[idx]
        ptid = self._ptids[idx]
        img = self.preprocess(path)
        img_indicators = [1]

        if self.args.n_times == 2:
            scan_time = path.split('/')[-2]
            opflow_path = '.../voxelmorph_out/' + ptid + '_' + scan_time + '_flow.nii.gz'
            if os.path.exists(opflow_path):
                img_opflow = self.preprocess_3dim(opflow_path)
                img_indicators.append(1)
            else:
                img_opflow = torch.zeros((3,img.shape[-3],img.shape[-2],img.shape[-1])).to(img.dtype)
                img_indicators.append(0)
            
        img_indicators = torch.Tensor(img_indicators)
        return img, img_opflow, label, img_indicators, path
