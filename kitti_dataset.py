import os
import time
import random
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import tqdm as tq
import torchvision.transforms as transforms

class KittiDataset(Dataset):
    def __init__(self, path, transform, is_train=True, flow=False, segmentation=False, next_frame=False):
        super().__init__()
        self.path = path
        self.flow = flow
        self.segmentation = segmentation
        self.transform = transform
        self.next_frame = next_frame
        self.is_train = is_train
        self.images, self.labels, self.flows = self.get_images_and_labels_filename(path)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def get_images_and_labels_filename(self, path):
        # get full path of every png file in the path recursively
        # skip if it is in a label folder at any level
        images = []
        labels = []
        flows = []
        for root, dirs, files in os.walk(path):
            for file in files:
                full_path = os.path.join(root, file)
                flow_path = full_path.replace('train', 'flow_v6')
                if full_path.endswith(".png") and "label" not in full_path \
                        and "flow" not in full_path and "right" not in full_path and "val" not in full_path:

                    label_path = self.get_train_label_path(full_path)
                    if (not self.segmentation or os.path.exists(label_path)) and (not self.flow or os.path.exists(flow_path)):
                        images.append(full_path)
                        labels.append(label_path)
                        flows.append(flow_path)

        return np.array(images), np.array(labels), np.array(flows)
    
    def get_label_key(self, file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        # split by _ and remove leading zeros
        filename_split = filename.split('_')
        filename_split[1] = filename_split[1].lstrip('0')
        key = '_'.join(filename_split)
        return key
    
    def get_all_labels(self):
        labels = {}
        for root, dirs, files in os.walk(self.path):
            for file in files:
                full_path = os.path.join(root, file)
                if full_path.endswith(".png") and "label" in full_path:
                    key = self.get_label_key(full_path)
                    labels[key] = full_path
        return labels
                    
    def next_image_path(self, im_path, change):
        # filename with no extension
        filename = os.path.basename(im_path).split('.')[0]
        if self.is_train:
            filename_int = int(os.path.splitext(filename)[0])
            # fill leading zeros to the same string length as filename
            next_filename = str(filename_int + change).zfill(len(filename))
            next_im_path = im_path.replace(filename, next_filename)
        else:
            # format is 3_0000040.png and 3_0000041.png
            filename_int = int(filename.split('_')[1])
            next_filename = str(filename_int + change).zfill(len(filename.split('_')[1]))
            next_im_path = im_path.replace(filename.split('_')[1], next_filename)
        return next_im_path
    
    def get_train_label_path(self, im_path):
        label_path = str(im_path).replace('left', 'label').replace('right', 'label')
        label = os.path.splitext(os.path.basename(label_path))[0]
        filename = label.lstrip('0')
        return label_path.replace(label, filename)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        random_seed = int(time.time())
        random_state = torch.get_rng_state()

        outputs = []
        image = Image.open(self.images[idx])
        image = self.transform(image)
        image = self.normalize(image)
        outputs.append(image)

        if self.next_frame:
            next_image = Image.open(self.next_image_path(self.images[idx], 2))
            next_image = self.transform(next_image)
            next_image = self.normalize(next_image)
            outputs.append(next_image)

        if self.flow:
            flow = Image.open(self.flows[idx])
            flow = self.transform(flow)
            flow = torch.div(flow, 255.0)
            outputs.append(flow)

        # 3x2x600
        imu_dummy = torch.zeros(3, 2, 100)
        outputs.append(imu_dummy)
        

        if self.segmentation:
            label = Image.open(self.labels[idx])
            label1 = label.point(lambda p: p == 151 and 255).convert('1')
            # if the value is 128, set to 1
            label2 = label.point(lambda p: (p == 150) and 255).convert('1')
            # if the value is 0, set to 1
            label3 = label.point(lambda p: (p == 0 or p == 255) and 255).convert('1')
            
            torch.set_rng_state(random_state)
            label1 = self.transform(label1)
            torch.set_rng_state(random_state)
            label2 = self.transform(label2)
            torch.set_rng_state(random_state)
            label3 = self.transform(label3)
            torch.set_rng_state(random_state)

            label_joined = torch.cat((label1, label2, label3), dim=0)

            outputs.append(label_joined)

        return tuple(outputs)
    
    def compute_num_of_classes(self):
        # set
        classes = set()
        for label in self.labels:
            label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
            classes.update(np.unique(label))
        self.classes = classes
        self.num_classes = len(classes)