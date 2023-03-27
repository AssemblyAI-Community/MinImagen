# importing necessary packages 
from PIL import Image
from skimage import io, transform
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from itertools import compress
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from minimagen.t5 import t5_encode_text



dataset_path = "/home/kareemelgohary/Downloads/Sample_data/CholecT50_sample-20221208T171010Z-001/CholecT50_sample"
data_path = os.path.join(dataset_path, 'data')
triplet_path = os.path.join(dataset_path, 'triplet')
dict_path = os.path.join(dataset_path, 'dict')
video_names = os.listdir(data_path)
video_path= os.path.join(data_path,video_names[0])
print("Dataset paths successfully defined!")
# print(triplet_path)
# print(video_names)

# Function return the videos pathes 
def video_pathes(data_path,video_names):
    """
    Defines simple function to return video pathes as List
    Args:
        String the path of the data
        List of Video names
    Returns:
        List of pathes for each video folder.
    """
    list_video_pathes= []
    for video_name in video_names:
        list_video_pathes.append(os.path.join(data_path,video_name))
    return sorted(list_video_pathes)

vp= video_pathes(data_path,video_names)
fram=sorted(os.listdir(vp[0]))
# print(fram[0])

# Create function to get the pathes of the frames from the videos in sequentiol orders.
def video_frames(video_pathes):
    """
    Defines simple function to return the frames of the videos as List in sequential way
    Args:
        List of pathes of the videos
    Return:
        List of Frames
    """
    frames_pathes_list=[]
    for video_name in video_pathes:
        frames = sorted(os.listdir(video_name))
        for frame in frames:
            frames_pathes_list.append(os.path.join(video_name,frame))
    return frames_pathes_list

# print(video_frames(vp)[0:6])  
video_frames_pathes = video_frames(vp)  

# Create dictionary mapping triplet ids to readable label

with open(os.path.join(dict_path, 'triplet.txt'), 'r') as f:
  triplet_info = f.readlines()
  triplet_dict = {}
  for l in triplet_info:
    triplet_id, triplet_label = l.split(':')
    triplet_dict[int(triplet_id)] = triplet_label.rstrip()
# print(triplet_dict)


# Create Functoin to get the tripplet 
def Tripplite_label(triplet_path,video_names):
    data_set={}
    for video_name in video_names:
        with open(os.path.join(triplet_path, video_name + '.txt'), mode='r') as infile:
            reader = csv.reader(infile)

            for line in reader:
                line = np.array(line, np.int64)
                frame_id, triplet_label = line[0], line[1:]
                image_path = os.path.join(data_path, video_name, "%06d.png" %frame_id)
                image = np.array(Image.open(image_path), np.float32) / 255.0
                indices = list(compress(range(len(triplet_label)), triplet_label))

                
                data_set.update({image_path:indices})
    return data_set

# print(Tripplite_label(triplet_path,video_names))
data_set= Tripplite_label(triplet_path,video_names)



# Create function to generate the image pathes and its triplets.
def mapping(data_set,triplet_dict):
    """
    Function to create dictionary consists of the the image paths and its labels in English
    Args:
        Data set (Dictionary) the pathes and its binary labels
        triplet (Dictionary) the indx and the english words of triplet 
    Return:
        Dictionary contain the image pathes and its triplet. the triplet could be more than one list 
    """
    data = {}
    for i in data_set:
        indeces_singel_list = data_set[i]
        if  len(indeces_singel_list)!=0:
            label = []
            for indx in indeces_singel_list:
                label.append([triplet_dict[indx]])
            data.update({i:label})
        pass
    return data
     



# print(mapping(data_set,triplet_dict))
# d= mapping(data_set,triplet_dict)



# # create json object from dictionary
# json = json.dumps(d)
# # open file for writing, "w" 
# f = open("dict.json","w")
# # write json object to file
# f.write(json)
# # close file
# f.close()

# print(list(d.items())[5][1][0])
# example= list(d.items())[5][1][0]
# str_example= ' '.join(example).replace(',', ' ')
# # str_example = str_example.replace(',', ' ')
# print(str_example)

# Creating Dataset Class
class MinimagenDatasetNew(Dataset):
    """Triplet DataSet """
    def __init__(self,dect_dataset,encoder_name,max_length,transform=None):
        """
        Args:
            Dictionary file contain the image pathes and its triplet
        """
        self.Triplet_data = list(dect_dataset.items())
        self.encoder_name = encoder_name
        self.max_length = max_length
        self.transform = transform
    def __len__(self):
        return len(self.Triplet_data)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.Triplet_data[idx][0]
        description = self.Triplet_data[idx][1][0] 
        str_description = ' '.join(description).replace(',', ' ')
        image = io.imread(img_name)
        # convert_tensor = transforms.ToTensor()
        # converted_image= convert_tensor(image)

        enc, msk = t5_encode_text([str_description], self.encoder_name, self.max_length)

        sample = {'image': image, 'encoding': enc, 'mask': msk}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):
    """ Rescale the image in sample to given size"""
    """
    Args:
        output_size (tuple or int)If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """


    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        image = sample['image']
        enc = sample['encoding']
        msk= sample['mask']
        h,w=image.shape[:2]
        if isinstance(self.output_size,int):
            if h>w:
                new_h,new_w = self.output_size * h/w , self.output_size
            else:
                new_h,new_w = self.output_size,self.output_size * w/h 
        else:
            new_h , new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': image, 'encoding': enc, 'mask': msk}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self,sample):
        image = sample['image']
        enc = sample['encoding']
        msk= sample['mask']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'encoding': enc, 'mask': msk}








with open('/home/kareemelgohary/Desktop/minImagen/MinImagen/dict.json') as f:
    d = json.load(f)
T_dataset=MinimagenDatasetNew(d,"t5_base",28,transform=transforms.Compose([Rescale(256),ToTensor()]))
print("script ends here")
# fig = plt.figure()
for i in range(len(T_dataset)):
    sample = T_dataset[i]
    print(i, sample['image'].shape,sample['encoding'].shape,sample['mask'].shape)
    if i ==7:
        break


