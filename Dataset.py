import torch
from PIL import Image, ImageOps
import os
from scipy.signal import stft
import numpy as np

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

class PlaceCrop(object):

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths))
        imgs = imgs.convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data

# for bhm example
def MinMaxNorm(x_train):
    x_max = np.max(x_train)
    x_min = np.min(x_train)
    x_train = (x_train-x_min)/(x_max - x_min)
    return x_train

def load_bhm_feature(filePath,v_num,b1b2,fea_name):
    labels = np.genfromtxt(filePath+'label.csv', delimiter=',', dtype='f')
    data1 = np.genfromtxt(filePath+'1.csv', delimiter=',', dtype='f').T
    data2 = np.genfromtxt(filePath+'2.csv', delimiter=',', dtype='f').T
    data3 = np.genfromtxt(filePath+'3.csv', delimiter=',', dtype='f').T
    data4 = np.genfromtxt(filePath+'4.csv', delimiter=',', dtype='f').T
    leng = 4000
    n_sample = 480

    if fea_name == 'stft':
        f,t,Zxx = stft(data1[1,:],nperseg=128)
        fea_bw=np.zeros((3600,np.shape(Zxx)[0],\
                        np.shape(Zxx)[1]))
        fea_fw=np.zeros((3600,np.shape(Zxx)[0],\
                        np.shape(Zxx)[1]))
        fea_bc=np.zeros((3600,np.shape(Zxx)[0],\
                        np.shape(Zxx)[1]))
        fea_fc=np.zeros((3600,np.shape(Zxx)[0],\
                        np.shape(Zxx)[1]))
        for i in range(3600):
            _,_,tmp=stft(data1[i,:],nperseg=128)
            fea_bw[i,:,:] = MinMaxNorm(abs(tmp))
            _,_,tmp=stft(data2[i,:],nperseg=128)
            fea_fw[i,:,:] = MinMaxNorm(abs(tmp))
            _,_,tmp=stft(data3[i,:],nperseg=128)
            fea_bc[i,:,:] = MinMaxNorm(abs(tmp))
            _,_,tmp=stft(data4[i,:],nperseg=128)
            fea_fc[i,:,:] = MinMaxNorm(abs(tmp))
        x_dann = np.zeros((n_sample*2,np.shape(Zxx)[0],\
                        np.shape(Zxx)[1],4))

    feas = [fea_bw,fea_fw,fea_bc,fea_fc]
    n=0
    for fea_ in feas:
        b1vnl2 = (np.floor(labels/100)==1200+v_num)&(labels%(1200+v_num)>0)
        b1vnl4 = (np.floor(labels/100)==1400+v_num)
        b1vnl6 = (np.floor(labels/100)==1600+v_num)&(labels%(1600+v_num)>0)  
        idx_tmp1 = np.logical_or.reduce((b1vnl2,b1vnl4,b1vnl6))
        feas_bb1 = fea_[idx_tmp1]
        b2vnl2 = (np.floor(labels/100)==2200+v_num)&(labels%(2200+v_num)>0)
        b2vnl4 = (np.floor(labels/100)==2400+v_num)
        b2vnl6 = (np.floor(labels/100)==2600+v_num)&(labels%(2600+v_num)>0)
        idx_tmp2 = np.logical_or.reduce((b2vnl2,b2vnl4,b2vnl6))
        feas_bb2 = fea_[idx_tmp2]
        
        feas_bb1_0 = feas_bb1[120:150,:,:]
        feas_bb2_0 = feas_bb2[120:150,:,:]
        if b1b2:
            feas_bb = np.concatenate((feas_bb1,feas_bb1_0,feas_bb1_0,feas_bb1_0,\
                                    feas_bb2,feas_bb2_0,feas_bb2_0,feas_bb2_0))
        else:
            feas_bb = np.concatenate((feas_bb2,feas_bb2_0,feas_bb2_0,feas_bb2_0,\
                                    feas_bb1,feas_bb1_0,feas_bb1_0,feas_bb1_0))     
        
        x_dann[:,:,:,n]=feas_bb
        n = n+1
        idx_tmp = np.logical_or.reduce((idx_tmp1,idx_tmp2))
        label_bb = labels[idx_tmp]

    label_l = np.zeros((n_sample,1))
    label_s = np.zeros((n_sample,1))
    damage_ls = np.zeros((150,1))
    for i in range(0,5,1):
        damage_ls[30*i:30*(i+1),0]=i
    label_s[0:120,0] = damage_ls[30:].squeeze()
    label_s[120:270,0] = damage_ls.squeeze()
    label_s[270:390,0] = damage_ls[30:].squeeze()
    label_l[0:120,0] = 1
    label_l[120:150,0] = 0
    label_l[150:270,0] = 2
    label_l[270:390,0] = 3
    label_s = label_s.squeeze()
    label_l = label_l.squeeze()
    label_d = label_l.copy()
    label_d[label_d!=0] = 1
    label_flatten = np.zeros((n_sample,1))
    for i in range(0,13,1):
        label_flatten[i*30:(i+1)*30,0]=i

    return x_dann, label_l, label_s, label_d, label_flatten, label_bb, damage_ls

class Dataset_bhm(torch.utils.data.Dataset):
    def __init__(self, data, label, resize_size=64, crop_size=60,\
                 is_train = True):
        n = len(data)
        self.data = data
        self.label = label
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    # Get one sample
    def __getitem__(self, index):  
        labels = int(self.label[index])
        img = self.data[index]
        if not self.is_train:
            img = img + np.random.randn(65,64,4) * 0.01
        else:
            img = img
        img = torch.tensor(img).float().transpose(0,2)

        return img, torch.tensor(labels).long()
