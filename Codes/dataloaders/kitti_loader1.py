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

    #sparse_dep = '1x0_nSKips'
    sparse_dep = '4x0_nSKips'
    #sparse_dep = '0x0_nSKips_256'
    #sparse_dep = '8x0_nSKips'
    
        

    if split == "test":
        if args.val == "full":
            transform = test_transform
            glob_d = glob.glob(os.path.join(
                args.data_folder,
                'KITTI_Depth/KITTI_final_origcrop/val/*_sync/proj_depth', sparse_dep,'image_0[2,3]/*.png'
            ))
            glob_gt = glob.glob(os.path.join(
                args.data_folder,
                'KITTI_Depth/KITTI_final_origcrop/val/*_sync/proj_depth/groundtruth_256/image_0[2,3]/*.png'
            ))
            glob_rgb = glob.glob(os.path.join(
                args.data_folder,
                "KITTI_Depth/KITTI_final_origcrop/val/*_sync/color/image_0[2,3]/*.png")
                )
            def get_rgb_paths(p):
                ps = p.split('/')
                file_prepend = ('/').join(p.split('/')[4:8])
                file_postpend = ('/').join(p.split('/')[-2:])
                pnew = os.path.join(args.data_folder, file_prepend, 'color', file_postpend)            
                return pnew
        elif args.val in ["select", 'test_selectsplit']:
            if args.val == 'select':
                transform = test_transform
            else:                
                transform = test_transformsplit
            glob_d = glob.glob(os.path.join(
                args.data_folder,
                "KITTI_Depth/KITTI_final_origcrop/val_temp/val_shortened/proj_depth", sparse_dep, "*.png"))
            glob_gt = glob.glob(os.path.join(
                args.data_folder,
                "KITTI_Depth/KITTI_final_origcrop/val_temp/val_shortened/proj_depth/groundtruth_256/*.png")
            )
            glob_rgb = glob.glob(os.path.join(
                args.data_folder,
                "KITTI_Depth/KITTI_final_origcrop/val_temp/val_shortened/image/*.png")
                ) 
            def get_rgb_paths(p):
                return p.replace("proj_depth/groundtruth_256","image")        
        

        
    elif split == 'val_selection':
        transform = test_transform
        glob_d = glob.glob(os.path.join(
            args.data_folder, 'KITTI_Depth',
            "depth_selection/val_selection_cropped/velodyne_raw/*.png"
        ))
        glob_gt = glob.glob(os.path.join(
            args.data_folder, 'KITTI_Depth',
            "depth_selection/val_selection_cropped/groundtruth_depth/*.png"))
        glob_rgb = glob.glob(os.path.join(
            args.data_folder,'KITTI_Depth',
            "depth_selection/val_selection_cropped/image/*.png"))

    elif split == "test_completion":
        transform = test_transform
        glob_d = glob.glob(os.path.join(
            args.data_folder, 'KITTI_Depth',
            "depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
        ))
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = glob.glob(os.path.join(
            args.data_folder,'KITTI_Depth',
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

def depth2dc(depth, method = 'gaussian3hot', params = None, sigma = None):

    nchan = params['dce_nChannels'] + 2
    depthbins = np.arange(1, nchan)    
    dstep = np.float64(params['dce_dstep'])
    hgt, wid, __ = np.shape(depth)    
    depth = depth.flatten()/dstep
    depthind = np.round(depth)
    delta = depth - depthind
    goodpix = np.where(np.logical_and(depthind > 0, depthind <= params['dce_nChannels']))
    pix_indics = np.concatenate((goodpix[0], goodpix[0], goodpix[0]), axis = 0)
    depthbins = np.int64(np.concatenate((depthind[goodpix[0]] - 1, depthind[goodpix[0]], depthind[goodpix[0]]+1), 0))
    delta_filtered = delta[goodpix[0]]

    if method == '3hot':

        vals = np.concatenate(((0.5 - delta_filtered)/2, 0.5*np.ones(np.shape(delta_filtered)), (0.5 + delta_filtered)/2), axis = 0)

    elif method == 'gaussian3hot':
        sigma = 0.6*dstep
        delta1sq = np.exp( -np.power(depthind[goodpix[0]] - 1 - depth[goodpix[0]], 2) * np.power(dstep/sigma,2) / 2 )  #neighbor to peak bin
        delta2sq = np.exp( - np.power(depthind[goodpix[0]] - depth[goodpix[0]], 2) * np.power(dstep/sigma,2) / 2 )     #peak bin
        delta3sq = np.exp( -np.power(depthind[goodpix[0]] + 1 - depth[goodpix[0]], 2) * np.power(dstep/sigma, 2) / 2 )  #other neighbor to peak bin        
        
        normval = delta1sq + delta2sq + delta3sq  #normalize by sum of 3 Depth Coefficients
        delta1sq = delta1sq/normval
        delta2sq = delta2sq/normval
        delta3sq = delta3sq/normval

        vals = np.concatenate(( delta1sq, delta2sq, delta3sq), axis = 0)        

    else:
        ValueError('depth2dc method does not exist')

    dcc_fin = np.zeros((hgt*wid*nchan))
    ind = sub2ind((hgt*wid, nchan), pix_indics, depthbins)
    dcc_fin[ind] = vals
    dcc_fin = np.reshape(dcc_fin, (hgt , wid , nchan))

    return dcc_fin


def dc2depth(dcpred, spatialdim, params = None):

    nchan = params['dce_nChannels'] + 2
    dstep = params['dce_dstep']
    dcpred = np.reshape(dcpred, [spatialdim[0]*spatialdim[1], nchan])
    depthchan = np.float32(np.arange(0, nchan))*dstep
    
    dvals = np.ones((spatialdim[0]*spatialdim[1], 1),np.float32)*depthchan
    sumProbs = np.maximum(1e-5, np.sum( dcpred, 1 ) )  #avoid divide by zero
    depth = np.zeros((spatialdim[0]*spatialdim[1]))
    goodpix = np.where(sumProbs > 1e-5)
    depth[goodpix[0]] = np.sum( dvals[goodpix[0], :] * dcpred[goodpix[0], :], 1) / sumProbs[goodpix[0]]
    depth = np.reshape(depth, (spatialdim[0], spatialdim[1], 1))
    return depth

def dc2max3coeffdep(dcpred, spatialdim, params = None):
    nChannels = params['dce_nChannels'] + 2
    dstep = params['dce_dstep']
    dcpred = np.reshape(dcpred, (-1, nChannels ))
    max_bin = np.argmax(dcpred, 1)
    goodpixind = np.logical_and(max_bin > 0, max_bin <= (nChannels + 1))
    
    max_binfilt = max_bin[goodpixind]
    near_binfilt = max_binfilt - 1
    far_binfilt = max_binfilt + 1    
    
    max_wtfilt = dcpred[goodpixind, max_binfilt]
    near_wtfilt = dcpred[goodpixind, near_binfilt]
    far_wtfilt = dcpred[goodpixind, far_binfilt]

    depth_vals = (max_wtfilt*max_binfilt*dstep + near_wtfilt*near_binfilt*dstep + far_binfilt*far_wtfilt*dstep)/(max_wtfilt + near_wtfilt + far_wtfilt)

    depth = np.zeros((spatialdim[0]*spatialdim[1], 1), np.float32)
    depth[goodpixind] = depth_vals
    depth = np.reshape(depth, (spatialdim[0], spatialdim[1], 1))
    
    return depth

def full2sparse(depth, method = 'gaussian3hot', params = None):

    nchan = params['dce_nChannels'] + 2
    spatial_dim = depth.shape
    depthbins = np.arange(1, nchan)    
    dstep = np.float64(params['dce_dstep'])    
    depth = depth.flatten()/dstep
    depthind = np.round(depth)
    delta = depth - depthind
    goodpix = np.where(np.logical_and(depthind > 0, depthind <= params['dce_nChannels']))
    y, x = np.tile(np.asarray(ind2sub(spatial_dim, goodpix[0])), (1, 3)).astype(np.int32)

    pix_indics = np.concatenate((goodpix[0], goodpix[0], goodpix[0]), axis = 0)
    depthbins = np.int32(np.concatenate((depthind[goodpix[0]] - 1, depthind[goodpix[0]], depthind[goodpix[0]] + 1), 0))
    depthbins = depthbins[:, np.newaxis].T
    coords = np.vstack((y, x, depthbins)).T
    delta_filtered = delta[goodpix[0]]

    if method == '3hot':

        vals = np.concatenate(((0.5 - delta_filtered)/2, 0.5*np.ones(np.shape(delta_filtered)), (0.5 + delta_filtered)/2), axis = 0)

    elif method == 'gaussian3hot':
        sigma = 0.6*dstep
        delta1sq = np.exp( -np.power(depthind[goodpix[0]] - 1 - depth[goodpix[0]], 2) * np.power(dstep/sigma,2) / 2 )  #neighbor to peak bin
        delta2sq = np.exp( - np.power(depthind[goodpix[0]] - depth[goodpix[0]], 2) * np.power(dstep/sigma,2) / 2 )     #peak bin
        delta3sq = np.exp( -np.power(depthind[goodpix[0]] + 1 - depth[goodpix[0]], 2) * np.power(dstep/sigma, 2) / 2 )  #other neighbor to peak bin        
        
        normval = delta1sq + delta2sq + delta3sq  #normalize by sum of 3 Depth Coefficients
        delta1sq = delta1sq/normval
        delta2sq = delta2sq/normval
        delta3sq = delta3sq/normval

        vals = np.concatenate(( delta1sq, delta2sq, delta3sq), axis = 0)        

    else:
        ValueError('depth2dc method does not exist')    
    vals = vals[:, np.newaxis]
    return coords, vals


def full2sparse_nonuniform(depth_map, depthbins, binsigmas, params = None):

    sz = depth_map.shape()
    depth_map = np.flatten(depth_map)    
    N = sz[0]*sz[1]*sz[2]
    depthbins = depthbins.astype(np.float32)
    binsigmas = binsigmas.astype(np.float32)
    depthbins = np.ones((N, 1))*depthbins        
    binlen = depthbins.shape()[1]
    goodmask = depth_map > 0    
    validpix = np.where(goodmask)    
    deltas = np.pow(depthbins - depth_map[:, np.newaxis], 2)/(2*(binsigmas[np.newaxis, :]**2))
    gauss = np.exp(-deltas)        
    depth_mask = depth_map == 0
    depth_mask = depth_mask[:, np.newaxis]
    depth_mask = np.tile(depth_mask, (1, binlen))
    gauss[depth_mask] = 0
    
    mainind = np.argmax(gauss, 1)
    mainind = mainind*goodmask.astype(np.float32)
    mainind = mainind[:, np.newaxis]    
    closeind = mainind - 1
    farind = mainind + 1
    pixind = np.concatenate((validpix, validpix, validpix), 0)    
    depthbins = np.concatenate((closeind, mainind, farind), 0)    
    sparseind = sub2ind((N, binlen), pixind, depthbins)
    vals = gauss[sparseind]    
    y, x = np.tile(np.asarray(ind2sub((sz[0]*sz[1]), validpix)), (1, 3)).astype(np.int32)
    depthbins = depthbins[:, np.newaxis].T
    coords = np.vstack((y, x, depthbins)).T    

    vals = vals[:, np.newaxis]
    return coords, vals

def train_transform(rgb, sparse, target, rgb_near, args, params = None):
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
        if rgb_near is not None:
            rgb_near = transform_rgb(rgb_near)
        #sparse = drop_depth_measurements(sparse, 0.7)

    return rgb, sparse, target, rgb_near


def val_transform(rgb, sparse, target, rgb_near, args):
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
    if rgb_near is not None:
        rgb_near = transform(rgb_near)
    return rgb, sparse, target, rgb_near


def test_transform(rgb, sparse, target, rgb_near, args):
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
    if rgb_near is not None:
        rgb_near = transform(rgb_near)
    return rgb, sparse, target, rgb_near


def no_transform(rgb, sparse, target, rgb_near, args):
    return rgb, sparse, target, rgb_near


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img

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
        rgb_near = get_rgb_near(self.paths['rgb'][index], self.args) if \
            self.split == 'train' else None
        return rgb, sparse, target, rgb_near

    def __getitem__(self, index):
        rgb, sparse, target, rgb_near = self.__getraw__(index)
        orig_sz = np.shape(sparse)   
        rgb, sparse, target, rgb_near = self.transform(rgb, sparse, target,
                                                       rgb_near, self.args)
        
        
        
        #rgb, gray = handle_gray(rgb, self.args)
        rgb, gray = rgb, None
        if rgb is not None:
            rgb = rgb/255.
        if gray is not None:
            gray = gray/255.            
                    
        candidates = {"rgb": rgb, "d": sparse, "gt": target,  \
            "g":gray, "r_mat":r_mat, "t_vec":t_vec, "rgb_near":rgb_near, 'orig_sz': np.zeros(orig_sz)}
        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }

        return items

    def __len__(self):
        return len(self.paths['gt'])


