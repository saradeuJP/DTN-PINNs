import os
import sys
import numpy as np
import csv
import tensorflow as tf
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def read_data():

    
    t_star=np.arange(0,(6*0.00512),0.00512)
    # print(t_star)
    # print(t_star.shape)

    file_dir='../DATA_magPINNs/Bmode_flow/velocity_csv'

    x_star = []
    y_star = [] # N x 1
    z_star = [] # N x 1

    U_star = [] # N x T
    Vel_star = [] # N x T
    W_star = [] # N x T
    #P_star = []

    for j in range(1,7,1):
        csv_file = f'{file_dir}/velocity_frame_{j}.csv'
        pd_data = pd.read_csv(csv_file)
        x = pd_data['x']
        z = pd_data['z']
        u = pd_data['velocity_x']
        w = pd_data['velocity_y']
        vel = pd_data['velocity_magnitude']

        x_star=np.concatenate([x_star,x],0)
        z_star=np.concatenate([z_star,z],0)
        U_star=np.concatenate([U_star,u],0)
        W_star=np.concatenate([W_star,w],0)
        Vel_star=np.concatenate([Vel_star,vel],0)
 
    X_star=np.array(x_star)
    Z_star=np.array(z_star)
    
    U_star=np.array(U_star)
    W_star=np.array(W_star)
    Vel_star=np.array(Vel_star)
    
    # t_star = (t_star - t_star.min()) / (t_star.max() - t_star.min())
    # X_star = (X_star - X_star.min()) / (X_star.max() - X_star.min())
    # Y_star = (Y_star - Y_star.min()) / (Y_star.max() - Y_star.min())
    # Z_star = (Z_star - Z_star.min()) / (Z_star.max() - Z_star.min())
    
    # U_star = (U_star - U_star.min()) / (U_star.max() - U_star.min())
    # V_star = (V_star - V_star.min()) / (V_star.max() - V_star.min())
    # W_star = (W_star - W_star.min()) / (W_star.max() - W_star.min())
  
    Ntot = len(U_star)
    
    # print(t_star)
    # print(X_star)
    # print(U_star)
    # dummy
    
    X_star = X_star.reshape(len(t_star),int(Ntot/len(t_star))).T
    Z_star = Z_star.reshape(len(t_star),int(Ntot/len(t_star))).T
    
    U_star=U_star.reshape(len(t_star),int(Ntot/len(t_star))).T
    W_star=W_star.reshape(len(t_star),int(Ntot/len(t_star))).T
    Vel_star=Vel_star.reshape(len(t_star),int(Ntot/len(t_star))).T

    T = t_star.shape[0]
    N = U_star.shape[0]
    
    # Rearrange Data 
    t_star = t_star.reshape(T,1)
    T_star = np.tile(t_star, (1,N)).T # N x T
    
    print(X_star.shape) # N x 1
    
    return T_star, X_star, Z_star, U_star, W_star, Vel_star



def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x
   
def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred.reshape(-1,1) - exact))
    return tf.reduce_mean(tf.square(tf.reshape(pred,[-1,1]) - exact))