import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import csv
import pandas as pd
from scipy.special import softmax
from scipy.spatial import KDTree
from numpy.linalg import lstsq

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_seg', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
HOSTNAME = socket.gethostname()

def compute_scalar_stats(arr, name, add_std=True):
    out = {
        f"{name}_mean":   float(np.mean(arr)),
        f"{name}_q1":     float(np.percentile(arr, 25)),
        f"{name}_median": float(np.median(arr)),
        f"{name}_q3":     float(np.percentile(arr, 75)),
        f"{name}_min":    float(np.min(arr)),
        f"{name}_max":    float(np.max(arr)),
    }
    if add_std:
        out[f"{name}_std"] = float(np.std(arr))
    return out

def compute_metrics_per_step(gt_uvw, pred_uvw):
    """
    针对单个时间步计算：
      - GT / Pred 的 u,v,w mean/std
      - NMAE_u/v/w 与 NRMSE_u/v/w（分母为“全时段 GT 的 (max-min)”）
      - ASI/MSI/SI 的统计（含 std）
    """

    # ASI / MSI / SI
    gt_mag   = np.linalg.norm(gt_uvw, axis=1)
    pred_mag = np.linalg.norm(pred_uvw, axis=1)
    dotp     = np.sum(gt_uvw * pred_uvw, axis=1)
    cos_sim  = dotp / (gt_mag * pred_mag)
    cos_sim  = np.clip(cos_sim, -1.0, 1.0)
    asi      = 0.5 * (1.0 + cos_sim)

    # 归一化幅值差（以本步 GT 的 max|v| 做尺度）
    max_gt_mag = float(np.max(gt_mag))
    msi = 1.0 - np.abs((pred_mag / max_gt_mag) - (gt_mag / max_gt_mag))
    si  = asi * msi

    out = {}
    
    out.update(compute_scalar_stats(asi, "ASI", add_std=True))
    out.update(compute_scalar_stats(msi, "MSI", add_std=True))
    out.update(compute_scalar_stats(si,  "SI",  add_std=True))
    return out

def read_data(csv_file):
    """Reads a CSV file and extracts point cloud and velocity data."""
    x, y, z, u, v, w = [], [], [], [], [], []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, start=1):
            # Skip the first 6 header lines
            if line_num <= 6:
                continue

            x.append(float(row[0])*1000)
            y.append(float(row[1])*1000)
            z.append(float(row[2])*1000)
            u.append(float(row[5]))
            v.append(float(row[6]))
            w.append(float(row[7]))

    return np.array(x).reshape(-1,1), np.array(y).reshape(-1,1),np.array(z).reshape(-1,1),np.array(u).reshape(-1,1),np.array(v).reshape(-1,1),np.array(w).reshape(-1,1)
   
layers = [4] + 10*[250] + [4]

eqn_file = ('../DATA_Veinous_Sinus/p6/velp_step1.csv')
x_eqn, y_eqn, z_eqn, u_gt, v_gt, w_gt = read_data(eqn_file)
t_eqn = np.tile(0.0, [x_eqn.shape[0], 1])

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        t_eqns_pl, x_eqns_pl, y_eqns_pl, z_eqns_pl = [tf.placeholder(tf.float32, shape=(None, 1)) for _ in range(4)]   
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        modelseg = MODEL.get_model(t_eqn, x_eqn, y_eqn, z_eqn, layers, is_training_pl)
            
        # Get model and loss 
        # Data
        u_pred, v_pred, w_pred, p_pred = modelseg(t_eqns_pl, x_eqns_pl, y_eqns_pl, z_eqns_pl)  
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'t_eqns_pl': t_eqns_pl, 'x_eqns_pl': x_eqns_pl, 'y_eqns_pl': y_eqns_pl, 'z_eqns_pl': z_eqns_pl,
           'is_training_pl': is_training_pl,
           'u_pred': u_pred, 'v_pred': v_pred,'w_pred': w_pred, 'p_pred': p_pred}
        

    eval_one_epoch(sess, ops, num_votes)


def eval_one_epoch(sess, ops, num_votes=1, topk=1):

    is_training = False
    all_stats = []
    
    mina = pd.DataFrame(columns = ['u', 'v', 'w'])
    maxa = pd.DataFrame(columns = ['u', 'v', 'w'])
    errors = pd.DataFrame(columns = ['maeu', 'maev', 'maew'])
    
    i=1
    for time in np.arange(0,0.8,0.001): 
    
        # i = 30
        # time = i * 0.001
        
        eqn_file = (f'../DATA_Veinous_Sinus/p6/velp_step{i}.csv')
        x_eqn, y_eqn, z_eqn, u_gt, v_gt, w_gt = read_data(eqn_file)
        
        t_eqn = np.tile(time, [x_eqn.shape[0], 1])
        
        feed_dict = {ops['t_eqns_pl']: t_eqn.reshape(-1,1),
                     ops['x_eqns_pl']: x_eqn.reshape(-1,1),
                     ops['y_eqns_pl']: y_eqn.reshape(-1,1),
                     ops['z_eqns_pl']: z_eqn.reshape(-1,1),
                     ops['is_training_pl']: is_training}
        u_pred,v_pred,w_pred,p_pred = sess.run([ops['u_pred'],
                                                 ops['v_pred'],
                                                 ops['w_pred'],
                                                 ops['p_pred']], feed_dict=feed_dict)
                                                 
        errors.loc[i] = (np.mean(np.abs(u_gt-u_pred),axis=0),np.mean(np.abs(v_gt-v_pred),axis=0),np.mean(np.abs(w_gt-w_pred),axis=0))
        
        mina.loc[i] = (np.min(np.abs(u_gt),axis=0),np.min(np.abs(v_gt),axis=0),np.min(np.abs(w_gt),axis=0))
        maxa.loc[i] = (np.max(np.abs(u_gt),axis=0),np.max(np.abs(v_gt),axis=0),np.max(np.abs(w_gt),axis=0))
        print(errors.loc[i])
        
        gt_uvw = np.concatenate([u_gt, v_gt, w_gt],axis=-1)
        pred_uvw = np.concatenate([u_pred,v_pred,w_pred],axis=-1)
        
        stats = compute_metrics_per_step(gt_uvw, pred_uvw)
        stats["step"] = i
        stats["time"] = time
        all_stats.append(stats)
        print(all_stats)
        
        # result = pd.DataFrame(np.concatenate([np.array(x_eqn).reshape(-1,1), np.array(y_eqn).reshape(-1,1), np.array(z_eqn).reshape(-1,1), 
                                             # np.array(u_pred).reshape(-1,1), np.array(v_pred).reshape(-1,1), np.array(w_pred).reshape(-1,1), 
                                             # np.array(p_pred).reshape(-1,1)], axis=-1), 
                                             # columns=["x", "y", "z", "u", "v", "w", "p"])
        # result.to_csv(f'transient_results/velp_step{i}.csv', header=True, index=False)
        i = i+1
        
    
    minmin = np.min(np.array(mina),axis=0)
    maxmax = np.max(np.array(maxa),axis=0)
    print(minmin)
    print(maxmax)
    print(maxmax-minmin)
    dummy
    # print(errors)
    errors['maeu'] = errors['maeu']/(maxmax[0]-minmin[0])
    errors['maev'] = errors['maev']/(maxmax[1]-minmin[1])
    errors['maew'] = errors['maew']/(maxmax[2]-minmin[2])

    # print(errors)
    errors.to_csv("test_error_transient.csv",index=False)
    
    df_stats = pd.DataFrame(all_stats)
    df_stats.to_csv("stats_asi_msi_si.csv", index=False)

if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
