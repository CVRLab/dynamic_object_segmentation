import os
import time
import random
import numpy as np
import pandas as pd
import cv2
import torch
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathos.multiprocessing import ProcessingPool as Pool

import quaternion
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

# get every imu0_aligned.csv file and get the min and max values for each column
def get_imu_normalization_parameters(root):
    imu_files = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name == 'imu0_aligned.csv':
                imu_files.append(os.path.join(path, name))
    imu_files.sort()

    # all_imu = pd.DataFrame(columns=["timestamp", "wx", "ax", "wy", "ay", "wz", "az"])
    all_imu = pd.DataFrame()
    for path in imu_files:
        imu = pd.read_csv(path, skiprows=1, header=None)
        imu.columns = ["timestamp", "wx", "wy", "wz", "ax", "ay", "az"]
        # change to wx, ax, wy, ay, wz, az
        imu = imu[["timestamp", "wx", "ax", "wy", "ay", "wz", "az"]]
        all_imu = pd.concat([all_imu, imu], ignore_index=True)
    
    # get min and max values for each column
    min_values = all_imu.min()
    max_values = all_imu.max()

    # return as float numpy arrays with no timestamp
    return min_values[1:].values.astype(np.float32), max_values[1:].values.astype(np.float32)

imu_mins, imu_maxs = get_imu_normalization_parameters("/home/thiago/Workspace/motion-segmentation/datasets/WHUVID")

class WhuvidDataset(Dataset):
    def __init__(self, root_path, sequences, transform, segmentation, flow, use_gdino=False, use_bbox=False):
        self.segmentation = segmentation
        self.flow = flow
        self.use_gdino = use_gdino
        self.use_bbox = use_bbox
        self.images = []
        self.masks = []
        self.flows = []
        self.imu = []
        self.poses = []
        self.bounding_boxes = []
        self.transform = transform
        self.width = 1280
        self.height = 720
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.frame_diff = 10
        self.imu_max_size = self.frame_diff * 10

        # same as above, but parallelized
        with Pool(24) as p:
            files = p.map(lambda seq: self.get_all_data(os.path.join(root_path, seq), "cam0"), sequences)
        for f in files:
            self.images += f[0]
            self.masks += f[1]
            self.flows += f[2]
            self.imu += f[3]
            self.poses += f[4]
            self.bounding_boxes += f[5]
            print(f"Sequence has {len(f[0])} images")

    
    def image_list_from_file(self, root_path):
        path = os.path.join(root_path, "other_files/image.txt")
        with open(path) as f:
            lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        # remove first 3 (header)
        lines = lines[3:]
        # get only the filename (after cam0/)
        lines = [os.path.basename(line) for line in lines]
        return lines
    
    def to_list_of_boxes(self, bb_raw):
        bb_of_cars = [(x['relative_coordinates'], x['confidence'], x['name']) for x in bb_raw if x['name'] != "label"]

        bb_of_cars_pixel = [{'center_x': x['center_x'] * self.width,
                             'center_y': x['center_y'] * self.height,
                             'width': x['width'] * self.width,
                             'height': x['height'] * self.height,
                             'confidence': confidence,
                             'name': name
                             } for x, confidence, name in bb_of_cars]
        boxes = []
        for bb in bb_of_cars_pixel:
            x0, y0 = bb['center_x'] - bb['width'] / 2, bb['center_y'] - bb['height'] / 2
            x1, y1 = x0 + bb['width'], y0 + bb['height']
            confidence = bb['confidence']
            name = bb['name']
            boxes.append([x0, y0, x1, y1, confidence, name])
        return boxes

    def get_bounding_boxes(self, path):
        if self.use_gdino:
            file_path = os.path.join(path, "other_files/gdino.json")
        else:
            file_path = os.path.join(path, "other_files/BoundingBox.json")

        with open(file_path) as f:
            json_file = json.load(f)

            bounding_boxes_raw = {os.path.basename(x['filename']): x['objects'] for x in json_file}
            
            # bounding_boxes.append(self.to_list_of_boxes(bounding_boxes_raw[file_path]))
            return {k: self.to_list_of_boxes(v) for k, v in bounding_boxes_raw.items()}

    # imu_read is a numpy array of 10x6
    # self.imu_maxs and self.imu_mins are lists of 6
    def imu_normalize(self, imu_read):
        imu_read = imu_read.copy()
        for i in range(6):
            imu_read[:, i] = (imu_read[:, i] - imu_mins[i]) / (imu_maxs[i] - imu_mins[i])
        # change from 10x3 to 3x2x10, where 3 is the axis (x, y, z) and 2 is the sensor (gyro, accel)
        imu_read = np.transpose(imu_read, (1, 0))
        imu_read = np.reshape(imu_read, (3, 2, self.imu_max_size))
        return imu_read
    
    def read_imu(self, root_path):
        # in root_path/other_files/imu0.txt
        # as csv with timestamp, wx, wy, wz, ax, ay, az
        # skip first line
        imu_path = os.path.join(root_path, "other_files/imu0_aligned.csv")
        imu = pd.read_csv(imu_path, skiprows=1, header=None)
        imu.columns = ["timestamp", "wx", "wy", "wz", "ax", "ay", "az"]
        # change to wx, ax, wy, ay, wz, az
        imu = imu[["timestamp", "wx", "ax", "wy", "ay", "wz", "az"]]
        # sort by timestamp
        imu = imu.sort_values(by=["timestamp"])
        return imu
    
    # return imu data between start and end timestamps using binary search
    def find_imu_binary_search(self, imu, start, end):
        # find index of start and end
        start_index = imu["timestamp"].searchsorted(start)
        end_index = imu["timestamp"].searchsorted(end)
        # return imu data between start and end timestamps
        imu_range = imu.iloc[start_index:end_index]
        # turn into numpy array of 10x6, without timestamp, padded with 0 and cropped to 10
        imu_range = imu_range.drop(columns=["timestamp"]).to_numpy()
        # crop to max_size
        imu_range = imu_range[:self.imu_max_size]
        # pad with 0 if less than max_size
        imu_range = np.pad(imu_range, ((0, self.imu_max_size - imu_range.shape[0]), (0, 0)))
        return self.imu_normalize(imu_range)
    
    # return imu data between start and end timestamps
    def find_imu(self, imu, start, end):
        imu_range = imu[(imu["timestamp"] >= start) & (imu["timestamp"] < end)]
        # turn into numpy array of 10x6, without timestamp, padded with 0 and cropped to 10
        imu_range = imu_range.drop(columns=["timestamp"]).to_numpy()
        # crop to max_size
        imu_range = imu_range[:self.imu_max_size]
        # pad with 0 if less than max_size
        imu_range = np.pad(imu_range, ((0, self.imu_max_size - imu_range.shape[0]), (0, 0)))
        return self.imu_normalize(imu_range)
    
    def find_imu_with_cache(self, imu, start, end, dir):
        filename = f"{start}_{end}.npy"
        filepath = os.path.join(dir, filename)
        if os.path.exists(filepath):
            return np.load(filepath)
        imu_range = self.find_imu_binary_search(imu, start, end)
        np.save(filepath, imu_range)
        return imu_range
    
    # split between data before and after timestamp
    # imu is a pandas dataframe with timestamp, wx, wy, wz, ax, ay, az
    def split_imu_data(self, imu, timestamp):
        for i, t in enumerate(imu["timestamp"]):
            if t > timestamp:
                return imu.iloc[:i], imu.iloc[i:]
        return imu, pd.DataFrame()
    
    def imu_format(self, imu):
        imu = imu.drop(columns=["timestamp"]).to_numpy()
        imu = imu[:10]
        imu = np.pad(imu, ((0, 10 - imu.shape[0]), (0, 0)))
        return self.imu_normalize(imu)
    
    # in groundtruthTUM_100hz.txt inside other_files
    # file start example:
    # # ground truth trajectory of 100hz
    # # folder: WHG0XX
    # # timeval ENU_x ENU_y ENU_z qx qy qz qw
    # 1605327910.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.984411 0.175882
    # 1605327910.010000 0.000000 0.000000 0.000000 0.000000 0.000000 0.984411 0.175882
    # 1605327910.020000 0.000000 0.000000 0.000000 0.000000 0.000000 0.984411 0.175882
    def read_ground_truth_pose(self, root_path):
        gt_path = os.path.join(root_path, "other_files/groundtruthTUM_100hz.txt")
        gt = pd.read_csv(gt_path, skiprows=3, header=None, delimiter=" ", float_precision='round_trip')
        gt.columns = ["timestamp", "ENU_x", "ENU_y", "ENU_z", "qx", "qy", "qz", "qw"]
        # find duplicated timestamps and keep only the first
        # gt = gt.drop_duplicates(subset=["timestamp"], keep="first")
        # print root_path if there are duplicated timestamps
        if len(gt) != len(gt.drop_duplicates(subset=["timestamp"], keep="first")):
            print(root_path + " has duplicated timestamps")
        gt = gt.set_index("timestamp").to_dict(orient="index")
        # convert to int
        gt = {int(k * 100): v for k, v in gt.items()}
        return gt
    
    def get_pose_of_image(self, gt, timestamp):
        # unit conversion, 1605328317999816061 -> 160532831799 (rounded)
        timestamp = int(timestamp / 10000000)
        return gt[timestamp]
    
    def calib_imu(self, imu_path, gt_path):
        imu = np.genfromtxt(imu_path, delimiter=",", skip_header=1)

        imu[:, 0] = imu[:, 0] / 1e9

        # read ground truth to numpy
        gt = np.genfromtxt(gt_path, delimiter=" ", skip_header=2)

        # reduce IMU to the same amout of ground truth
        imu_2 = np.zeros((gt.shape[0], 7))
        for i in range(gt.shape[0]):
            idx = np.argmin(np.abs(imu[:, 0] - gt[i, 0]))
            imu_2[i, :] = imu[idx, :]
        imu = imu_2

        # ground truth quaternion starting from identity
        first_q = quaternion.quaternion(gt[0, 7], gt[0, 4], gt[0, 5], gt[0, 6])
        gt_q = [first_q]
        for i in range(1, gt.shape[0]):
            qx, qy, qz, qw = gt[i, 4:]
            q = quaternion.quaternion(qw, qx, qy, qz)
            gt_q.append(q)


        # gt_q as [ts, qw, qx, qy, qz]
        qt_np = np.array([[gt[i, 0], *gt_q[i].components] for i in range(len(gt_q))])
        # change from xyzw to wxyz
        qt_np = qt_np[:, [0, 4, 1, 2, 3]]

        # Function to integrate the gyroscope data to estimate rotation
        def integrate_gyro(timestamps, gyroscope_data, bias):
            num_samples = len(timestamps)
            estimated_rotations = np.zeros((num_samples, 4))
            estimated_rotations[0] = [1, 0, 0, 0]  # Initial rotation (identity quaternion)

            for i in range(1, num_samples):
                dt = timestamps[i] - timestamps[i - 1]
                wx, wy, wz = gyroscope_data[i] - bias
                omega = np.array([wx, wy, wz])
                delta_q = R.from_rotvec(omega * dt).as_quat()
                estimated_rotations[i] = (R.from_quat(estimated_rotations[i - 1]) * R.from_quat(delta_q)).as_quat()

            return estimated_rotations

        # Cost function to minimize
        def cost_function(bias, timestamps, gyroscope_data, ground_truth):
            estimated_rotations = integrate_gyro(timestamps, gyroscope_data, bias)
            diff = R.from_quat(ground_truth).inv() * R.from_quat(estimated_rotations)
            angle_diff = np.linalg.norm(diff.as_rotvec(), axis=1)
            return np.sum(angle_diff**2)
        
        timestamps = imu[:, 0]
        gyroscope_data = imu[:, 1:4]
        ground_truth = qt_np[:, 1:]

        # Initial guess for the bias
        initial_bias = np.array([0, 0, 0])

        # Optimize the bias
        result = minimize(cost_function, initial_bias, args=(timestamps, gyroscope_data, ground_truth))
        optimal_bias = result.x
        return optimal_bias
    
    def get_all_data(self, root_path, cam):
        images = []
        masks = []
        flows = []
        imu = []
        poses = []
        bounding_boxes = []

        imu_path = os.path.join(root_path, "other_files/imu0_aligned.csv")
        gt_path = os.path.join(root_path, "other_files/groundtruthTUM_100hz.txt")
        # imu_bias = self.calib_imu(imu_path, gt_path)

        image_list = self.image_list_from_file(root_path)
        bounding_boxes_dict = self.get_bounding_boxes(root_path)
        # temp removal to avoid errors
        # ground_truth_pose = self.read_ground_truth_pose(root_path)
        images_path = os.path.join(root_path, cam)
        masks_path = os.path.join(root_path, cam + "_masks_ann")
        imu_split_path = os.path.join(root_path, "imu_split")
        if not os.path.exists(imu_split_path):
            os.makedirs(imu_split_path)
        imu_data = self.read_imu(root_path)

        
        for i, image_name in enumerate(image_list):
            image_path = os.path.join(images_path, image_name)
            label_path = os.path.join(masks_path, image_name)
            flow_path = os.path.join(root_path, "flow_v6", cam, image_name)
            next_image_path = image_list[i+self.frame_diff] if i < len(image_list) - self.frame_diff else None
            bbox = bounding_boxes_dict[image_name] if image_name in bounding_boxes_dict else None

            if os.path.exists(image_path) and (not self.segmentation or os.path.exists(label_path)) \
                and next_image_path is not None and (not self.flow or os.path.exists(flow_path)) \
                and (not self.use_bbox or bbox is not None):
                # and os.path.exists(flow_path)
                timestamp = int(os.path.splitext(image_name)[0])
                next_timestamp = int(os.path.splitext(os.path.basename(next_image_path))[0])
                imu_range = self.find_imu_with_cache(imu_data, timestamp, next_timestamp, imu_split_path)
                imu.append(imu_range)
                images.append(image_path)
                masks.append(label_path)
                flows.append(flow_path)
                # poses.append(pose)
                poses.append([0, 0, 0, 0, 0, 0, 1])
                bounding_boxes.append(bbox)

        return images, masks, flows, imu, poses, bounding_boxes
         
    def __getitem__(self, idx):
        outputs = []
        
        image = Image.open(self.images[idx])
        random_state = torch.get_rng_state()
        image = self.transform(image)
        image = self.normalize(image)


        outputs.append(image)

        if self.flow:
            flow = Image.open(self.flows[idx])
            # flow = cv2.imread(self.flows[idx])
            torch.set_rng_state(random_state)
            flow = self.transform(flow)
            # from 0-255 to 0-1
            # flow = torch.tensor(flow, dtype=torch.float32) / 255.0
            flow = torch.div(flow, 255.0)
            outputs.append(flow)
        else:
            outputs.append(None)

        imu = self.imu[idx]

        imu = torch.from_numpy(imu)
        outputs.append(imu)

        if self.segmentation:

            label = Image.open(self.masks[idx])


            # if the value is 255, set to 1
            label1 = label.point(lambda p: p >= 255 and 255).convert('1')
            # if the value is 128, set to 1
            label2 = label.point(lambda p: p == 128 and 255).convert('1')
            # if the value is 0, set to 1
            label3 = label.point(lambda p: p == 0 and 255).convert('1')

            torch.set_rng_state(random_state)
            label1 = self.transform(label1)
            torch.set_rng_state(random_state)
            label2 = self.transform(label2)
            torch.set_rng_state(random_state)
            label3 = self.transform(label3)

            label_joined = torch.cat((label1, label2, label3), dim=0)

            outputs.append(label_joined)
        
        if self.use_bbox:
            bbox = self.bounding_boxes[idx]
            bbox = [[x[0], x[1], x[2], x[3]] for x in bbox]
            outputs.append(bbox)

        return tuple(outputs)
    
    def __len__(self):
        return len(self.images)
    
    def split_train_test(self):
        train_idx = []
        test_idx = []
        for i in range(len(self.images)):
            if i % 5 == 0:
                test_idx.append(i)
            else:
                train_idx.append(i)
        return train_idx, test_idx
        