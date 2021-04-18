import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
import itertools
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from dataloaders import transforms
from torchvision.transforms import RandomCrop
from dataloaders.pose_estimator import get_pose_pnp

input_options = ['d', 'rgb', 'rgbd', 'g', 'gd']

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)


def load_calib():
    """
    Temporarily hardcoding the calibration matrix using calib file from 2011_09_26
    """
    calib = open("dataloaders/calib_cam_to_cam.txt", "r")
    lines = calib.readlines()
    P_rect_line = lines[25]

    Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
    Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                      (3, 4)).astype(np.float32)
    K = Proj[:3, :3]  # camera matrix

    # note: we will take the center crop of the images during augmentation
    # that changes the optical centers, but not focal lengths
    K[0, 2] = K[
        0,
        2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
    K[1, 2] = K[
        1,
        2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
    return K


def get_paths_and_transform(split, args, trainvalsplit = 0.9):
    assert (args.use_d or args.use_rgb
            or args.use_g), 'no proper input selected'
        
    if split == 'val_selection':
        transform = test_transform
        glob_d = glob.glob(os.path.join(
            args.data_folder, 'Data',
            "depth_selection/val_selection_cropped/velodyne_raw/*.png"
        ))
        glob_gt = glob.glob(os.path.join(
            args.data_folder, 'Data',
            "depth_selection/val_selection_cropped/groundtruth_depth/*.png"))
        glob_rgb = glob.glob(os.path.join(
            args.data_folder,'Data',
            "depth_selection/val_selection_cropped/image/*.png"))

    elif split == "test_completion":
        transform = test_transform
        glob_d = glob.glob(os.path.join(
            args.data_folder, 'Data',
            "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
        ))
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = glob.glob(os.path.join(
            args.data_folder,'Data',
            "depth_selection/test_depth_completion_anonymous/image/*.png"))    
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob_d)
        paths_gt = sorted(glob_gt)
        if len(glob_rgb):
            paths_rgb = sorted(glob_rgb)
        else:            
            paths_rgb = [get_rgb_paths(p) for p in paths_gt]
    else:  
        # test only has d or rgb
        paths_rgb = sorted(glob_rgb)
        paths_gt = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob_d)

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0 and args.use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise (RuntimeError("Produced different sizes for datasets"))
    totlen = len(paths_rgb)
    if split == 'train':
        paths_0 = [paths_rgb[i] for i in range(0, int(round(trainvalsplit*totlen)), 1)]
        paths_1 = [paths_d[i] for i in range(0, int(round(trainvalsplit*totlen)), 1)]
        paths_2 = [paths_gt[i] for i in range(0, int(round(trainvalsplit*totlen)), 1)]
        paths = {"rgb": paths_0, 
                "d":  paths_1, 
                "gt": paths_2}
    elif split == 'val':
        paths = {"rgb": paths_rgb[int(round(trainvalsplit*totlen))+1:], 
                "d": paths_d[int(round(trainvalsplit*totlen))+1:], 
                "gt": paths_gt[int(round(trainvalsplit*totlen))+1:]}

    
                                 
    else:            
        paths = {"rgb": paths_rgb[0:], "d": paths_d[0:], "gt": paths_gt[0:]}
    return paths, transform


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    
    #assert np.max(depth_png) > 255, \
    #    "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


oheight, owidth, cwidth = 256, 1216, 1216


def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth


def train_transform(rgb, sparse, target, args, params = None):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip
    maxcrop_h, maxcrop_w = oheight, cwidth
    y_b = 0
    y_t = y_b + maxcrop_h
    h = oheight
    w = owidth
    x_l = np.random.randint(w - maxcrop_w + 1, size = ())
    x_r = x_l + maxcrop_w
    #do_flip = False
    transform_geometric = transforms.Compose([
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip), 
        transforms.RandomCrop([x_l, x_r, y_b, y_t])        
    ])
    if sparse is not None:
        sparse = transform_geometric(sparse)
    target = transform_geometric(target)

    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)        

    return rgb, sparse, target,


def val_transform(rgb, sparse, target, args):
    maxcrop_h, maxcrop_w = oheight, cwidth
    y_b = 0
    y_t = y_b + maxcrop_h
    h = oheight
    w = owidth
    x_l = np.random.randint(w - maxcrop_w + 1, size = ())
    x_r = x_l + maxcrop_w

    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
        transforms.RandomCrop([x_l, x_r, y_b, y_t])
        ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
    
    return rgb, sparse, target, 


def test_transform(rgb, sparse, target, args):
    maxcrop_h, maxcrop_w = oheight, cwidth
    y_b = 0
    y_t = y_b + maxcrop_h
    h = oheight
    w = owidth
    x_l = np.random.randint(w - maxcrop_w + 1, size = ())
    x_r = x_l + maxcrop_w

    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
        #transforms.RandomCrop([x_l, x_r, y_b, y_t])
        ])
    if rgb is not None:
        rgb = transform(rgb)
    if sparse is not None:
        sparse = transform(sparse)
    if target is not None:
        target = transform(target)
   
    return rgb, sparse, target, 


def no_transform(rgb, sparse, target, args):
    return rgb, sparse, target, 


class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """
    def __init__(self, split, args, trainvalsplit = 0.9):
        self.args = args
        self.split = split
        self.trainvalsplit = trainvalsplit
        paths, transform = get_paths_and_transform(split, args, trainvalsplit = trainvalsplit)
        self.paths = paths
        self.transform = transform
        self.K = load_calib()
        self.threshold_translation = 0.1
        
        
    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None and (self.args.use_rgb or self.args.use_g)) else None
        sparse = depth_read(self.paths['d'][index]) if \
            (self.paths['d'][index] is not None and self.args.use_d) else None
        target = depth_read(self.paths['gt'][index]) if \
            self.paths['gt'][index] is not None else None
        
        return rgb, sparse, target, 

    def __getitem__(self, index):
        rgb, sparse, target,  = self.__getraw__(index)
        orig_sz = np.shape(sparse)   
        rgb, sparse, target, = self.transform(rgb, sparse, target,
                                                       self.args)       
        
        
        
        if rgb is not None:
            rgb = rgb/255.
        
                    
        candidates = {"rgb": rgb, "d": sparse, "gt": target}
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths['gt'])


