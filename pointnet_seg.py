import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

 
class get_model(object):
    def __init__(self, t_pl, x_pl, y_pl, z_pl, layers, is_training, bn_decay=None):
        
        
        self.layers = layers
        self.num_layers = len(self.layers)
        bn=False
        initializer = tf.contrib.layers.xavier_initializer()
        
        self.weights = []
        self.biases = []
        
        for i in range(0,self.num_layers-1):
            
            net_inputs = self.layers[i]
            num_outputs = self.layers[i+1]
            
            w_name = "w" + str(i)
            b_name = "b" + str(i)
            
            weights = tf.get_variable(w_name, [net_inputs, num_outputs], initializer=initializer, dtype=tf.float32)
            biases = tf.get_variable(b_name, [num_outputs], initializer=tf.constant_initializer(0.0))                
            
            self.weights.append(weights)
            self.biases.append(biases)

    def __call__(self, t_pl, x_pl, y_pl, z_pl):   
    
        activation_fn=tf.nn.sigmoid
        padding='SAME'
        
        net = tf.keras.layers.concatenate([t_pl, x_pl, y_pl, z_pl], axis = 1)
        
        
        for l in range(0,self.num_layers-1):
            W = self.weights[l]
            b = self.biases[l]
            net = tf.matmul(net, W)
            net = net + b
            if l < self.num_layers-2:
                net = net * activation_fn(net)
        
        
        u, v, w, p = tf.split(net, num_or_size_splits=net.shape[1], axis=1) # N x 1
        
        #mag = tf.keras.layers.concatenate([mag0, mag1], axis = 1) # N x 2
        #mag = tf.argmax(mag, 1) # N x 1
        
        return  u, v, w, p    

def nPINNs_time(x, y, z, t0, t1, Re, model):

    dt = t1 - t0

    # 模型在两个时间点的预测
    u0, v0, w0, p0 = model(t0, x, y, z)
    u1, v1, w1, p1 = model(t1, x, y, z)

    # 时间导数项 (Backward Euler)
    ut = (u1 - u0) / dt
    vt = (v1 - v0) / dt
    wt = (w1 - w0) / dt
    
    u,v,w,p = u1, v1, w1, p1
    
    dx, dy, dz = 0.25, 0.25, 0.25 # 0.25mm
    
    xE, xW = x + dx, x - dx
    yN, yS = y + dy, y - dy
    zU, zD = z + dz, z - dz
    
    uE, vE, wE, pE = model(t1, xE, y, z)
    uW, vW, wW, pW = model(t1, xW, y, z)
    uN, vN, wN, pN = model(t1, x, yN, z)
    uS, vS, wS, pS = model(t1, x, yS, z)
    uU, vU, wU, pU = model(t1, x, y, zU)
    uD, vD, wD, pD = model(t1, x, y, zD)

    
    uc_e, uc_w = 0.5*(uE + u), 0.5*(uW + u) 
    vc_n, vc_s = 0.5*(vN + v), 0.5*(vS + v)
    wc_u, wc_d = 0.5*(wU + w), 0.5*(wD + w)
    
    div = (uc_e - uc_w) /dx + (vc_n - vc_s) /dy + (wc_u - wc_d) /dz
    
    # second order
    xEE, xWW = x + 2.0*dx, x - 2.0*dx
    yNN, ySS = y + 2.0*dy, y - 2.0*dy   
    zUU, zDD = z + 2.0*dz, z - 2.0*dz
    
    uEE, vEE, wEE, pEE = model(t1, xEE, y, z)
    uWW, vWW, wWW, pWW = model(t1, xWW, y, z)
    uNN, vNN, wNN, pNN = model(t1, x, yNN, z)
    uSS, vSS, wSS, pSS = model(t1, x, ySS, z)
    uUU, vUU, wUU, pUU = model(t1, x, y, zUU)
    uDD, vDD, wDD, pDD = model(t1, x, y, zDD)
    
    # 2nd upwind
    # u
    Uem_uw2 = 1.5*u  - 0.5*uW
    Uep_uw2 = 1.5*uE - 0.5*uEE  
    Uwm_uw2 = 1.5*uW - 0.5*uWW
    Uwp_uw2 = 1.5*u  - 0.5*uE
    Ue_uw2 = tf.where(uc_e >= 0.0, Uem_uw2, Uep_uw2)
    Uw_uw2 = tf.where(uc_w >= 0.0, Uwm_uw2, Uwp_uw2)

        
    Unm_uw2 = 1.5*u  - 0.5*uS
    Unp_uw2 = 1.5*uN - 0.5*uNN    
    Usm_uw2 = 1.5*uS - 0.5*uSS
    Usp_uw2 = 1.5*u  - 0.5*uN
    Un_uw2 = tf.where(vc_n >= 0.0, Unm_uw2, Unp_uw2)
    Us_uw2 = tf.where(vc_s >= 0.0, Usm_uw2, Usp_uw2)
    
    Uum_uw2 = 1.5*u  - 0.5*uU
    Uup_uw2 = 1.5*uU - 0.5*uUU    
    Udm_uw2 = 1.5*uD - 0.5*uDD
    Udp_uw2 = 1.5*u  - 0.5*uD
    Uu_uw2 = tf.where(wc_u >= 0.0, Uum_uw2, Uup_uw2)
    Ud_uw2 = tf.where(wc_d >= 0.0, Udm_uw2, Udp_uw2)
    
    # v
    Vem_uw2 = 1.5*v  - 0.5*vW
    Vep_uw2 = 1.5*vE - 0.5*vEE
    Vwm_uw2 = 1.5*vW - 0.5*vWW
    Vwp_uw2 = 1.5*v  - 0.5*vE
    Ve_uw2 = tf.where(uc_e >= 0.0, Vem_uw2, Vep_uw2)
    Vw_uw2 = tf.where(uc_w >= 0.0, Vwm_uw2, Vwp_uw2)
        
    Vnm_uw2 = 1.5*v  - 0.5*vS
    Vnp_uw2 = 1.5*vN - 0.5*vNN    
    Vsm_uw2 = 1.5*vS - 0.5*vSS
    Vsp_uw2 = 1.5*v  - 0.5*vN
    Vn_uw2 = tf.where(vc_n >= 0.0, Vnm_uw2, Vnp_uw2)
    Vs_uw2 = tf.where(vc_s >= 0.0, Vsm_uw2, Vsp_uw2)
    
    Vum_uw2 = 1.5*v  - 0.5*vU
    Vup_uw2 = 1.5*vU - 0.5*vUU    
    Vdm_uw2 = 1.5*vD - 0.5*vDD
    Vdp_uw2 = 1.5*v  - 0.5*vD
    Vu_uw2 = tf.where(wc_u >= 0.0, Vum_uw2, Vup_uw2)
    Vd_uw2 = tf.where(wc_d >= 0.0, Vdm_uw2, Vdp_uw2)
    
    #w
    Wem_uw2 = 1.5*w  - 0.5*wW
    Wep_uw2 = 1.5*wE - 0.5*wEE
    Wwm_uw2 = 1.5*wW - 0.5*wWW
    Wwp_uw2 = 1.5*w  - 0.5*wE
    We_uw2 = tf.where(uc_e >= 0.0, Wem_uw2, Wep_uw2)
    Ww_uw2 = tf.where(uc_w >= 0.0, Wwm_uw2, Wwp_uw2)
        
    Wnm_uw2 = 1.5*w  - 0.5*wS
    Wnp_uw2 = 1.5*wN - 0.5*wNN    
    Wsm_uw2 = 1.5*wS - 0.5*wSS
    Wsp_uw2 = 1.5*w  - 0.5*wN
    Wn_uw2 = tf.where(vc_n >= 0.0, Wnm_uw2, Wnp_uw2)
    Ws_uw2 = tf.where(vc_s >= 0.0, Wsm_uw2, Wsp_uw2)
    
    Wum_uw2 = 1.5*w  - 0.5*wU
    Wup_uw2 = 1.5*wU - 0.5*wUU    
    Wdm_uw2 = 1.5*wD - 0.5*wDD
    Wdp_uw2 = 1.5*w  - 0.5*wD
    Wu_uw2 = tf.where(wc_u >= 0.0, Wum_uw2, Wup_uw2)
    Wd_uw2 = tf.where(wc_d >= 0.0, Wdm_uw2, Wdp_uw2)
    
    UUx_uw2 = (uc_e*Ue_uw2 - uc_w*Uw_uw2) /dx
    VUy_uw2 = (vc_n*Un_uw2 - vc_s*Us_uw2) /dy
    WUz_uw2 = (wc_u*Uu_uw2 - wc_d*Ud_uw2) /dz
    
    UVx_uw2 = (uc_e*Ve_uw2 - uc_w*Vw_uw2) /dx
    VVy_uw2 = (vc_n*Vn_uw2 - vc_s*Vs_uw2) /dy
    WVz_uw2 = (wc_u*Vu_uw2 - wc_d*Vd_uw2) /dz
    
    UWx_uw2 = (uc_e*We_uw2 - uc_w*Ww_uw2) /dx
    VWy_uw2 = (vc_n*Wn_uw2 - vc_s*Ws_uw2) /dy
    WWz_uw2 = (wc_u*Wu_uw2 - wc_d*Wd_uw2) /dz
    
    # 2nd central difference    
    Uxx_cd2 = (uE - 2.0*u + uW)/ (dx*dx) 
    Uyy_cd2 = (uN - 2.0*u + uS)/ (dy*dy) 
    Uzz_cd2 = (uU - 2.0*u + uD)/ (dz*dz)
    
    Vxx_cd2 = (vE - 2.0*v + vW)/ (dx*dx) 
    Vyy_cd2 = (vN - 2.0*v + vS)/ (dy*dy) 
    Vzz_cd2 = (vU - 2.0*v + vD)/ (dz*dz) 
    
    Wxx_cd2 = (wE - 2.0*w + wW)/ (dx*dx) 
    Wyy_cd2 = (wN - 2.0*w + wS)/ (dy*dy)
    Wzz_cd2 = (wU - 2.0*w + wD)/ (dz*dz)

    pe_cd2 = (p + pE) /2.0 
    pw_cd2 = (pW + p) /2.0 
    pn_cd2 = (p + pN) /2.0 
    ps_cd2 = (pS + p) /2.0 
    pu_cd2 = (p + pU) /2.0 
    pd_cd2 = (pD + p) /2.0 
    
    Px_cd2 = (pe_cd2 - pw_cd2) /dx
    Py_cd2 = (pn_cd2 - ps_cd2) /dy
    Pz_cd2 = (pu_cd2 - pd_cd2) /dz
        
    r1 = div
    r2 = UUx_uw2 + VUy_uw2 + WUz_uw2 - 1.0/Re *(Uxx_cd2 + Uyy_cd2 + Uzz_cd2) - u*div + Px_cd2
    r3 = UVx_uw2 + VVy_uw2 + WVz_uw2 - 1.0/Re *(Vxx_cd2 + Vyy_cd2 + Vzz_cd2) - v*div + Py_cd2 
    r4 = UWx_uw2 + VWy_uw2 + WWz_uw2 - 1.0/Re *(Wxx_cd2 + Wyy_cd2 + Wzz_cd2) - w*div + Pz_cd2  
    
    return r1, r2, r3, r4

def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x

def Navier_Stokes_3D(u, v, w, p, x, y, z, Pec, Rey):
    
    Y = tf.concat([u, v, w, p], 1)
    
    # Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    Y_zz = fwd_gradients(Y_z, z)
    
    u = Y[:,0:1]
    v = Y[:,1:2]
    w = Y[:,2:3]
    p = Y[:,3:4]
    
    u_t = 0.0 # Y_t[:,0:1]
    v_t = 0.0 # Y_t[:,1:2]
    w_t = 0.0 # Y_t[:,2:3]
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    w_x = Y_x[:,2:3]
    p_x = Y_x[:,3:4]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    w_y = Y_y[:,2:3]
    p_y = Y_y[:,3:4]
       
    u_z = Y_z[:,0:1]
    v_z = Y_z[:,1:2]
    w_z = Y_z[:,2:3]
    p_z = Y_z[:,3:4]
    
    u_xx = Y_xx[:,0:1]
    v_xx = Y_xx[:,1:2]
    w_xx = Y_xx[:,2:3]
    
    u_yy = Y_yy[:,0:1]
    v_yy = Y_yy[:,1:2]
    w_yy = Y_yy[:,2:3]
       
    u_zz = Y_zz[:,0:1]
    v_zz = Y_zz[:,1:2]
    w_zz = Y_zz[:,2:3]

    e1 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_yy + u_zz)
    e2 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1.0/Rey)*(v_xx + v_yy + v_zz)
    e3 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_yy + w_zz)
    e4 = u_x + v_y + w_z

    
    return e1, e2, e3, e4

def Navier_Stokes_2D(u, w, p, t, x, z, Pec, Rey):
    
    Y = tf.concat([u, w, p], 1)
    
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_z = fwd_gradients(Y, z)
    Y_xx = fwd_gradients(Y_x, x)
    Y_zz = fwd_gradients(Y_z, z)
    
    u = Y[:,0:1]
    w = Y[:,1:2]
    p = Y[:,2:3]
    
    u_t = Y_t[:,0:1]
    w_t = Y_t[:,1:2]
    
    u_x = Y_x[:,0:1]
    w_x = Y_x[:,1:2]
    p_x = Y_x[:,2:3]
       
    u_z = Y_z[:,0:1]
    w_z = Y_z[:,1:2]
    p_z = Y_z[:,2:3]
    
    u_xx = Y_xx[:,0:1]
    w_xx = Y_xx[:,1:2]
       
    u_zz = Y_zz[:,0:1]
    w_zz = Y_zz[:,1:2]

    e1 = u_t + (u*u_x + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_zz)
    e2 = w_t + (u*w_x + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_zz)
    e3 = u_x + w_z
    
    # e1m = tf.multiply(e1,tf.cast(m, tf.float32))
    # e2m = tf.multiply(e2,tf.cast(m, tf.float32))
    # e3m = tf.multiply(e3,tf.cast(m, tf.float32))
    
    return e1, e2, e3

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred.reshape(-1,1) - exact))
    return tf.reduce_mean(tf.square(tf.reshape(pred,[-1,1]) - exact))

def get_loss(x_pl, y_pl, z_pl, u_pred, v_pred, w_pred, p_pred, u_pl, v_pl, w_pl):
    """ pred: BxNxC,
        label: BxN, """
    
    # pc = tf.concat([x_pl, y_pl, z_pl],2) # 1 x N x 3
    # pred = tf.concat([u_pred, v_pred, w_pred, p_pred],2)
    
    # vel = tf.concat([u_pl, v_pl, w_pl],2)
    # pred_vel = tf.concat([u_pred, v_pred, w_pred],2)
    
    x_pl = tf.squeeze(x_pl,[0]) # N x 1
    y_pl = tf.squeeze(y_pl,[0])
    z_pl = tf.squeeze(z_pl,[0])
    u_pl = tf.squeeze(u_pl,[0])
    v_pl = tf.squeeze(v_pl,[0])
    w_pl = tf.squeeze(w_pl,[0])
    u_pred = tf.squeeze(u_pred,[0])
    v_pred = tf.squeeze(v_pred,[0])
    w_pred = tf.squeeze(w_pred,[0])
    p_pred = tf.squeeze(p_pred,[0])
    
    u_x = fwd_gradients(u_pred, x_pl)
    
    e_eqn = Navier_Stokes_3D(x_pl, y_pl, z_pl, u_pred, v_pred, w_pred, p_pred)
    e_data = tf.reduce_mean(tf.square(tf.reshape(pred_vel,[-1,1]) - tf.reshape(vel,[-1,1])))

    return e_eqn + e_data


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
