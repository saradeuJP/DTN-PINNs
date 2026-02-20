import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import csv
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import pandas as pd
import re

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_seg', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=120000, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=1000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# 文件夹路径
input_folder = "../DATA_Veinous_Sinus/p6_ds_0.7mm_fftnoise_time_ave"

# 匹配文件名中的 step 数字
pattern = re.compile(r"velp_step(\d+)\.csv")

# 获取并排序所有匹配的文件
file_list = []
for file in os.listdir(input_folder):
    match = pattern.match(file)
    if match:
        step_num = int(match.group(1))
        file_list.append((step_num, file))

# 按 step_num 排序
file_list.sort(key=lambda x: x[0])

x_list, y_list, z_list = [], [], []
u_list, v_list, w_list = [], [], []

for _, filename in file_list:
    df = pd.read_csv(os.path.join(input_folder, filename))

    x_list.append(df['x'].values.reshape(-1, 1)*1000)
    y_list.append(df['y'].values.reshape(-1, 1)*1000)
    z_list.append(df['z'].values.reshape(-1, 1)*1000)
    u_list.append(df['u'].values.reshape(-1, 1))
    v_list.append(df['v'].values.reshape(-1, 1))
    w_list.append(df['w'].values.reshape(-1, 1))

# 沿 axis=1 堆叠：结果是 [N, T]
x_data = np.hstack(x_list)
y_data = np.hstack(y_list)
z_data = np.hstack(z_list)
u_data = np.hstack(u_list)
v_data = np.hstack(v_list)
w_data = np.hstack(w_list)

# 生成对应时间数组：t_data 为 [T, 1]
T = len(file_list)
t_data = np.arange(T).reshape(1, -1) * 0.05
print(t_data)

t_data = np.tile(t_data, [x_data.shape[0], 1])

x_eqn = x_data[:,0].reshape(-1,1)
y_eqn = y_data[:,0].reshape(-1,1)
z_eqn = z_data[:,0].reshape(-1,1)

layers = [4] + 10*[250] + [4]

print(x_data.shape)
print(x_eqn.shape)
print(x_data.max()-x_data.min())
print(x_eqn.max()-x_eqn.min())
# dummy

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            t1_pl, x1_pl, y1_pl, z1_pl, u1_pl, v1_pl, w1_pl = [tf.placeholder(tf.float32, shape=(None, 1)) for _ in range(7)]
            t0_pl, x0_pl, y0_pl, z0_pl, u0_pl, v0_pl, w0_pl = [tf.placeholder(tf.float32, shape=(None, 1)) for _ in range(7)]
            t0_num_pl, t1_num_pl = [tf.placeholder(tf.float32, shape=(None, 1)) for _ in range(2)]
            
            x_eqns_pl, y_eqns_pl, z_eqns_pl = [tf.placeholder(tf.float32, shape=(None, 1)) for _ in range(3)]  
            # x_wall_pl, y_wall_pl, z_wall_pl = [tf.placeholder(tf.float32, shape=(None, 1)) for _ in range(3)]              
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            
            modelseg = MODEL.get_model(t_data, x_data, y_data, z_data, layers, is_training_pl, bn_decay=bn_decay)
            
            # Data
            # t = 0
            u0_datap, v0_datap, w0_datap, _ = modelseg(t0_pl, x0_pl, y0_pl, z0_pl) 
            
            eu0 = provider.mean_squared_error(u0_datap, u0_pl)
            ev0 = provider.mean_squared_error(v0_datap, v0_pl)
            ew0 = provider.mean_squared_error(w0_datap, w0_pl)
            
            # t = 1
            u1_datap, v1_datap, w1_datap, _ = modelseg(t1_pl, x1_pl, y1_pl, z1_pl) 
            
            eu1 = provider.mean_squared_error(u1_datap, u1_pl)
            ev1 = provider.mean_squared_error(v1_datap, v1_pl)
            ew1 = provider.mean_squared_error(w1_datap, w1_pl)
            
            e_data = (eu0 + ev0 + ew0) + (eu1 + ev1 + ew1)
            
            # Eqns

            Rey = 1.0/0.0036
            t_middle = tf.linspace(tf.squeeze(t0_num_pl), tf.squeeze(t1_num_pl), 6)[0:-1]
            
            e_eqn = 0.0
            for i in range(4):
                ti = tf.reshape(t_middle[i], [1, 1])  # ensure shape [1, 1]
                ti_next = tf.reshape(t_middle[i+1], [1, 1]) 
                
                [res1, res2, res3, res4] = MODEL.nPINNs_time(x_eqns_pl,y_eqns_pl,z_eqns_pl, 
                                                       tf.tile(ti, [tf.shape(x_eqns_pl)[0],1]),
                                                       tf.tile(ti_next, [tf.shape(x_eqns_pl)[0],1]),
                                                       Rey, modelseg)
                
                e_eqn += provider.mean_squared_error(res1, 0) + provider.mean_squared_error(res2, 0) \
                             + provider.mean_squared_error(res3, 0) + provider.mean_squared_error(res4, 0)
            
            
            # Wall
            
            # u_predw, v_predw, w_predw, p_predw = modelseg(x_wall_pl, y_wall_pl, z_wall_pl) 
            
            # euw = provider.mean_squared_error(u_predw, 0.0)
            # evw = provider.mean_squared_error(v_predw, 0.0)
            # eww = provider.mean_squared_error(w_predw, 0.0)
            
            # e_eqn = e1r + e2r + e3r + e4r # + euw + evw + eww

            loss = 20 * e_eqn + e_data
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        checkpoint_path = tf.train.latest_checkpoint(LOG_DIR)
        if checkpoint_path:
            print(f"Restoring from checkpoint: {checkpoint_path}")
            saver.restore(sess, checkpoint_path)
        else:
            print("No checkpoint found. Training from scratch.")
        

        ops = {'t0_pl': t0_pl, 'x0_pl': x0_pl, 'y0_pl': y0_pl, 'z0_pl': z0_pl,
               'u0_pl': u0_pl, 'v0_pl': v0_pl, 'w0_pl': w0_pl, 
               't1_pl': t1_pl, 'x1_pl': x1_pl, 'y1_pl': y1_pl, 'z1_pl': z1_pl,
               'u1_pl': u1_pl, 'v1_pl': v1_pl, 'w1_pl': w1_pl, 
               "t0_num_pl": t0_num_pl, "t1_num_pl": t1_num_pl,
               "ti":ti, "ti_next":ti_next,"t_middle":t_middle,
               'x_eqns_pl': x_eqns_pl, 'y_eqns_pl': y_eqns_pl, 'z_eqns_pl': z_eqns_pl,
               # 'x_wall_pl': x_wall_pl, 'y_wall_pl': y_wall_pl, 'z_wall_pl': z_wall_pl,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'e_eqn':e_eqn,
               'e_data':e_data,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        
        train_loss = []
        eval_loss = []
        
        for epoch in range(MAX_EPOCH):
            for ti in range(t_data.shape[1]-1):
                sys.stdout.flush()
                
                is_training = True
                
                N_data = x_data.shape[0] 
                idx_data = np.random.choice(N_data, NUM_POINT)
                
                N_eqns = x_eqn.shape[0]
                idx_eqns = np.random.choice(N_eqns, NUM_POINT)
                
                
                feed_dict = {ops['t0_pl']: t_data[idx_data,ti].reshape(-1,1),
                             ops['x0_pl']: x_data[idx_data,ti].reshape(-1,1),
                             ops['y0_pl']: y_data[idx_data,ti].reshape(-1,1),
                             ops['z0_pl']: z_data[idx_data,ti].reshape(-1,1),
                             ops['u0_pl']: u_data[idx_data,ti].reshape(-1,1),
                             ops['v0_pl']: v_data[idx_data,ti].reshape(-1,1),
                             ops['w0_pl']: w_data[idx_data,ti].reshape(-1,1),
                             ops['t1_pl']: t_data[idx_data,(ti+1)].reshape(-1,1),
                             ops['x1_pl']: x_data[idx_data,(ti+1)].reshape(-1,1),
                             ops['y1_pl']: y_data[idx_data,(ti+1)].reshape(-1,1),
                             ops['z1_pl']: z_data[idx_data,(ti+1)].reshape(-1,1),
                             ops['u1_pl']: u_data[idx_data,(ti+1)].reshape(-1,1),
                             ops['v1_pl']: v_data[idx_data,(ti+1)].reshape(-1,1),
                             ops['w1_pl']: w_data[idx_data,(ti+1)].reshape(-1,1),
                             ops['x_eqns_pl']: x_eqn[idx_eqns,:],
                             ops['y_eqns_pl']: y_eqn[idx_eqns,:],
                             ops['z_eqns_pl']: z_eqn[idx_eqns,:],
                             ops['t0_num_pl']: t_data[0,ti].reshape(-1,1), 
                             ops['t1_num_pl']: t_data[0,(ti+1)].reshape(-1,1), 
                             # ops['x_wall_pl']: x_wall[idx_wall,:],
                             # ops['y_wall_pl']: y_wall[idx_wall,:],
                             # ops['z_wall_pl']: z_wall[idx_wall,:],
                             ops['is_training_pl']: is_training}
                             
                [summary, step, _, loss_val,
                                  e_eqn, e_data, ti, ti_next, t_middle] = sess.run([ops['merged'], ops['step'],
                                                            ops['train_op'], ops['loss'], 
                                                            ops['e_eqn'], ops['e_data'], ops['ti'], ops['ti_next'], ops['t_middle']], feed_dict=feed_dict)

                train_writer.add_summary(summary, step)
            
            if epoch % 1 == 0:    
                with open('train_loss.csv','ab') as f:               
                          np.savetxt(f,loss_val.reshape(1,-1))
                
                with open('e_eqn.csv','ab') as f:               
                          np.savetxt(f,e_eqn.reshape(1,-1))
                
                with open('e_data.csv','ab') as f:               
                          np.savetxt(f,e_data.reshape(1,-1))
                          
                log_string('**** EPOCH %03d ****' % (epoch))
                log_string('loss: %f' % loss_val)
                
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
            
            
            #train_loss = train_one_epoch(sess, ops, train_writer)
            #eval_loss = eval_one_epoch(sess, ops, test_writer)
            
            # if ((train_loss<0.0001) & (eval_loss<0.0005)):
                # save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                # log_string("Model saved in file end: %s" % save_path)
                # break
            
            
            # Save the variables to disk.
                

    

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
