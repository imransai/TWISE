import argparse
import os
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt
import scipy.io as io

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from dataloaders.kitti_loader import load_calib, oheight, owidth, input_options, KittiDepth
from model import *
from metrics import AverageMeter, Result, accum_errorstat
import helper

from utils import *

params = {'depth_maxrange': 80.0,                                                
          'threshold': 100, 
          }

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w',
                    '--workers',
                    default=1,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--data-folder',
                    default='/home/imransai/Documents/Depth_SuperResolution/TWISE_Public/TWISE',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default= 'rgbd',
                    choices=input_options,
                    help='input: | '.join(input_options))
parser.add_argument(
    '--rank-metric',
    type=str,
    default='rmse',
    choices=[m for m in dir(Result()) if not m.startswith('_')],
    help='metrics for which best result is sbatch_datacted')
parser.add_argument(
    '-m',
    '--train-mode',
    type=str,
    default="dense",
    choices=["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
    help='dense | sparse | photo | sparse+photo | dense+photo')
parser.add_argument('--save_imageflag',
                    default='F', type=str,                   
                    help='save images for test')
parser.add_argument('--cpu', action="store_true", help='run on cpu')

args = parser.parse_args()
              
model_name = 'TWISE_gamma2.5'

modelcheckpt_dir = os.path.join('../pretrained_models', model_name)
modellog_dir = os.path.join('./log_dir', model_name)


# args.pretrained = not args.no_pretrained
args.result = modelcheckpt_dir
#args.use_rgb = ('rgb' in args.input)
args.use_rgb = True
args.use_d = 'd' in args.input
#args.use_g = 'g' in args.input
args.use_g = False
print(args)

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))


def iterate(mode, args, loader1, model, optimizer, logger, epoch, curr_step = None):
    
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]
    SAVE_MODEL_EVERY = 600
    error_binningflag = True
    if curr_step is None:
            curr_step = 0
    
    # switch to appropriate mode
    assert mode in ["train", "val", "test", "val_selection", "test_completion"], \
        "unsupported mode: {}".format(mode)
    
    model.eval()
    lr = 0
    t_total = 0        
        
    ndata_perepoch = round(len(loader1)/args.batch_size)        
    
    
    for i, batch_data in enumerate(loader1):

        curr_stepepoch = ndata_perepoch*epoch + curr_step
        batch_data = {
                key: val.to(device)
                for key, val in batch_data.items() if val is not None
            }

        start = time.time()
        data_time = time.time() - start                        

        
        if mode in  ['test_completion']:                
            save_path = os.path.join(args.data_folder, 'Eval_DepthModel', model_name)                
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            SAVE_SAMPLEINTERV = 1

            start = time.time()            
            color_img = batch_data['rgb']
            sparse_dep = batch_data['d']/params['depth_maxrange']                
            sparse_C = (sparse_dep > 0).float()
            valpred_dc, __, __ = model(sparse_dep, color_img)                
            pred_datavalid = smooth2chandep(valpred_dc, params = params, device = device)
            #pred_datavalid = hard2chandep(valpred_dc, params = params, device = device)
            #pred_datavalid =  F.relu(valpred_dc)*params['depth_maxrange']
            if not curr_step%SAVE_SAMPLEINTERV:
                pred_dep = pred_datavalid.detach().cpu().numpy()
                pred_dep = np.squeeze(pred_dep)
                pred_h, pred_w = np.shape(pred_dep)   
                zero_mtrx = np.zeros((352 - pred_h, owidth))
                pred_dep = np.concatenate((zero_mtrx, pred_dep))
                pred_dep_round = np.uint16(pred_dep*256)
                filename = '%010d.png'%curr_step
                imageio.imwrite(os.path.join(save_path, filename), pred_dep_round)
                print('Finished writing sample %010d.png\n'%curr_step)
            curr_step += 1                    
            avg = None
            is_best = None        

        elif mode in ['test', 'val_selection']:


            #save_path = os.path.join(args.data_folder, 'Val_DepthModel', model_name)                
            #save_path = os.path.join(args.data_folder, 'PredDep_AvgPoolgamma2.5_full')
            save_path = os.path.join('Save_ImageResults', model_name)
            if args.save_imageflag == 'T':
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                SAVE_SAMPLEINTERV = 1000
                data_anal = {}

            with torch.no_grad():
                start = time.time()
                gt_val = batch_data['gt']
                color_img = batch_data['rgb']
                sparse_dep = batch_data['d']/params['depth_maxrange']                
                
                t_start = time.time()
                valpred_dc, __, __ = model(sparse_dep, color_img)                
                #valpred_dc, __, __ = model(sparse_dep, color_img)                
                pred_datavalid = smooth2chandep(valpred_dc, params = params, device = device)
                t_total += time.time() - t_start
                
                if args.save_imageflag == 'T':
                    pred_dep = pred_datavalid.detach().cpu().numpy()
                    pred_dep = np.squeeze(pred_dep)
                    
                    split_chans = torch.split(valpred_dc, 1, 1) 
                    split_chans = list(split_chans)
                    dep_1stchan = split_chans[0].detach().cpu().numpy()
                    dep_2ndchan = split_chans[1].detach().cpu().numpy()
                    sigmoid_chan = torch.sigmoid(split_chans[2]).detach().cpu().numpy()
                    pred_dc = valpred_dc.detach().cpu().numpy()
                    sparse_dep = batch_data['d'].detach().cpu().numpy()
                    gt_dep = gt_val.detach().cpu().numpy()
                    color_img = batch_data['rgb'].detach().cpu().numpy()
                    data_anal['dep_1stchan'] = dep_1stchan
                    data_anal['dep_2ndchan'] = dep_2ndchan
                    data_anal['sigmoid_chan'] = sigmoid_chan
                    data_anal['color_img'] = color_img
                    data_anal['pred_dep'] = pred_dep                    
                    data_anal['pred_dc'] = pred_dc
                    data_anal['sparse_dep'] = sparse_dep
                    data_anal['gt_dep'] = gt_dep
                    io.savemat(os.path.join(save_path, 'sample_%0d.mat'%curr_step), data_anal)
                    
                gpu_time = time.time() - start
                result = Result()
                mini_batch_size = next(iter(batch_data.values())).size(0)
                result.evaluate(pred_datavalid.data, gt_val.data, 0)
                [
                    m.update(result, gpu_time, data_time, mini_batch_size)
                    for m in meters
                ]                
                
                logger.conditional_print(mode, i, epoch, lr, len(loader1),
                                    block_average_meter, average_meter)                    
                
                avg = logger.conditional_save_info(mode, average_meter, epoch)
                is_best = logger.rank_conditional_save_best(mode, avg, epoch)
            
                logger.conditional_summarize(mode, avg, is_best, i)                
                curr_step += 1

    
    return avg, is_best



def main():
    global args
    checkpoint = None
    is_eval = False
    args.curr_step = 0
    args_new = args        
    
    
    complete_modelname = os.path.join(modelcheckpt_dir, 'model_best.pth.tar')
    if os.path.isfile(complete_modelname):            
        print("=> loading checkpoint '{}' ... ".format(complete_modelname),
                end='')
        checkpoint = torch.load(complete_modelname, map_location=device)
        args = checkpoint['args']
        args.data_folder = args_new.data_folder
        args.save_imageflag = args_new.save_imageflag
        is_eval = True
        print("Completed.")
    else:
        print("No model found at '{}'".format(modelcheckpt_dir))
        return   
      
    
    print("=> creating model and optimizer ... ", end='')
    
    model = MultiRes_network_avgpool_diffspatialsizes().to(device)
    
    model_named_params = [
        p for _, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(model_named_params,
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    print("completed.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    model = torch.nn.DataParallel(model)

    # Data loading code
    print("=> creating data loaders ... ")
                
    val_dataset = KittiDepth("val_selection", args, trainvalsplit = 0.95)    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    # create backups and results folder
    logger = helper.logger(args, output_directory = modelcheckpt_dir)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")    

    if is_eval:
        print("=> starting model evaluation ...")
        result, is_best = iterate('val_selection', args, val_loader, model, None, logger,
                                  checkpoint['epoch'])
        return
    # main loop
    
        
    writer.close()                                  

if __name__ == '__main__':
    main()
