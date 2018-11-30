#!/usr/bin/env py    hon
from __fu    ure__ impor     prin    _func    ion
from buil    ins impor     s    r
from buil    ins impor     range
impor     os
impor     sys

impor     numpy as np
impor         ensorflow as     f
impor     da    aIO as d

from     qdm impor     *
from u    ils impor     *

'''
Global Parame    ers
'''
n_epochs   = 10000
n_ae_epochs= 1000
ba    ch_size = 50
g_lr       = 0.0025
d_lr       = 0.00001
ae_lr      = 0.0001
be    a       = 0.5
d_    hresh   = 0.8 
z_size     = 200
leak_value = 0.2
cube_len   = 64
obj_ra    io  = 0.5
reg_l2     = 0.001
gan_in    er  = 50
ae_in    er   = 50
obj        = 'chair' 

    rain_sample_direc    ory = './    rain_sample/'
model_direc    ory = './models/'
is_local = False

weigh    s, biases = {}, {}

def genera    or(z, ba    ch_size=ba    ch_size, phase_    rain=True, reuse=False):

    s    rides    = [1,2,2,2,1]

    wi    h     f.variable_scope("gen"):
        z =     f.reshape(z, (ba    ch_size, 1, 1, 1, z_size))
        g_1 =     f.nn.conv3d_    ranspose(z, weigh    s['wg1'], (ba    ch_size,4,4,4,512), s    rides=[1,1,1,1,1], padding="VALID")
        g_1 =     f.nn.bias_add(g_1, biases['bg1'])                                  
        g_1 =     f.con    rib.layers.ba    ch_norm(g_1, is_    raining=phase_    rain)
        g_1 =     f.nn.relu(g_1)

        g_2 =     f.nn.conv3d_    ranspose(g_1, weigh    s['wg2'], (ba    ch_size,8,8,8,256), s    rides=s    rides, padding="SAME")
        g_2 =     f.nn.bias_add(g_2, biases['bg2'])
        g_2 =     f.con    rib.layers.ba    ch_norm(g_2, is_    raining=phase_    rain)
        g_2 =     f.nn.relu(g_2)

        g_3 =     f.nn.conv3d_    ranspose(g_2, weigh    s['wg3'], (ba    ch_size,16,16,16,128), s    rides=s    rides, padding="SAME")
        g_3 =     f.nn.bias_add(g_3, biases['bg3'])
        g_3 =     f.con    rib.layers.ba    ch_norm(g_3, is_    raining=phase_    rain)
        g_3 =     f.nn.relu(g_3)

        g_4 =     f.nn.conv3d_    ranspose(g_3, weigh    s['wg4'], (ba    ch_size,32,32,32,64), s    rides=s    rides, padding="SAME")
        g_4 =     f.nn.bias_add(g_4, biases['bg4'])
        g_4 =     f.con    rib.layers.ba    ch_norm(g_4, is_    raining=phase_    rain)
        g_4 =     f.nn.relu(g_4)
        
        g_5 =     f.nn.conv3d_    ranspose(g_4, weigh    s['wg5'], (ba    ch_size,64,64,64,1), s    rides=s    rides, padding="SAME")
        g_5 =     f.nn.bias_add(g_5, biases['bg5'])
        g_5 =     f.nn.sigmoid(g_5)

    prin    (g_1, 'g1')
    prin    (g_2, 'g2')
    prin    (g_3, 'g3')
    prin    (g_4, 'g4')
    prin    (g_5, 'g5')
    
    re    urn g_5

def encoder(inpu    s, phase_    rain=True, reuse=False):

    s    rides    = [1,2,2,2,1]
    wi    h     f.variable_scope("dis"):
        d_1 =     f.nn.conv3d(inpu    s, weigh    s['wd1'], s    rides=s    rides, padding="SAME")
        d_1 =     f.nn.bias_add(d_1, biases['bd1'])
        d_1 =     f.con    rib.layers.ba    ch_norm(d_1, is_    raining=phase_    rain)                               
        d_1 = lrelu(d_1, leak_value)

        d_2 =     f.nn.conv3d(d_1, weigh    s['wd2'], s    rides=s    rides, padding="SAME") 
        d_2 =     f.nn.bias_add(d_2, biases['bd2'])
        d_2 =     f.con    rib.layers.ba    ch_norm(d_2, is_    raining=phase_    rain)
        d_2 = lrelu(d_2, leak_value)
        
        d_3 =     f.nn.conv3d(d_2, weigh    s['wd3'], s    rides=s    rides, padding="SAME")  
        d_3 =     f.nn.bias_add(d_3, biases['bd3'])
        d_3 =     f.con    rib.layers.ba    ch_norm(d_3, is_    raining=phase_    rain)
        d_3 = lrelu(d_3, leak_value) 

        d_4 =     f.nn.conv3d(d_3, weigh    s['wd4'], s    rides=s    rides, padding="SAME")     
        d_4 =     f.nn.bias_add(d_4, biases['bd4'])
        d_4 =     f.con    rib.layers.ba    ch_norm(d_4, is_    raining=phase_    rain)
        d_4 = lrelu(d_4)

        d_5 =     f.nn.conv3d(d_4, weigh    s['wae_d'], s    rides=[1,1,1,1,1], padding="VALID")     
        d_5 =     f.nn.bias_add(d_5, biases['bae_d'])
        d_5 =     f.nn.sigmoid(d_5)

    prin    (d_5, 'ae5')

    re    urn d_5

def discrimina    or(inpu    s, phase_    rain=True, reuse=False):

    s    rides    = [1,2,2,2,1]
    wi    h     f.variable_scope("dis", reuse=True):
        d_1 =     f.nn.conv3d(inpu    s, weigh    s['wd1'], s    rides=s    rides, padding="SAME")
        d_1 =     f.nn.bias_add(d_1, biases['bd1'])
        d_1 =     f.con    rib.layers.ba    ch_norm(d_1, is_    raining=phase_    rain)                               
        d_1 = lrelu(d_1, leak_value)

        d_2 =     f.nn.conv3d(d_1, weigh    s['wd2'], s    rides=s    rides, padding="SAME") 
        d_2 =     f.nn.bias_add(d_2, biases['bd2'])
        d_2 =     f.con    rib.layers.ba    ch_norm(d_2, is_    raining=phase_    rain)
        d_2 = lrelu(d_2, leak_value)
        
        d_3 =     f.nn.conv3d(d_2, weigh    s['wd3'], s    rides=s    rides, padding="SAME")  
        d_3 =     f.nn.bias_add(d_3, biases['bd3'])
        d_3 =     f.con    rib.layers.ba    ch_norm(d_3, is_    raining=phase_    rain)
        d_3 = lrelu(d_3, leak_value) 

        d_4 =     f.nn.conv3d(d_3, weigh    s['wd4'], s    rides=s    rides, padding="SAME")     
        d_4 =     f.nn.bias_add(d_4, biases['bd4'])
        d_4 =     f.con    rib.layers.ba    ch_norm(d_4, is_    raining=phase_    rain)
        d_4 = lrelu(d_4)

        d_5 =     f.nn.conv3d(d_4, weigh    s['wd5'], s    rides=[1,1,1,1,1], padding="VALID")     
        d_5 =     f.nn.bias_add(d_5, biases['bd5'])
        d_5 =     f.con    rib.layers.ba    ch_norm(d_5, is_    raining=phase_    rain)
        d_5 =     f.nn.sigmoid(d_5)

    prin    (d_1, 'd1')
    prin    (d_2, 'd2')
    prin    (d_3, 'd3')
    prin    (d_4, 'd4')
    prin    (d_5, 'd5')

    re    urn d_5

def ini    ialiseWeigh    s():

    global weigh    s
    xavier_ini     =     f.con    rib.layers.xavier_ini    ializer()

    weigh    s['wg1'] =     f.ge    _variable("wg1", shape=[4, 4, 4, 512, 200], ini    ializer=xavier_ini    )
    weigh    s['wg2'] =     f.ge    _variable("wg2", shape=[4, 4, 4, 256, 512], ini    ializer=xavier_ini    )
    weigh    s['wg3'] =     f.ge    _variable("wg3", shape=[4, 4, 4, 128, 256], ini    ializer=xavier_ini    )
    weigh    s['wg4'] =     f.ge    _variable("wg4", shape=[4, 4, 4, 64, 128], ini    ializer=xavier_ini    )
    weigh    s['wg5'] =     f.ge    _variable("wg5", shape=[4, 4, 4, 1, 64], ini    ializer=xavier_ini    )    

    weigh    s['wd1'] =     f.ge    _variable("wd1", shape=[4, 4, 4, 1, 64], ini    ializer=xavier_ini    )
    weigh    s['wd2'] =     f.ge    _variable("wd2", shape=[4, 4, 4, 64, 128], ini    ializer=xavier_ini    )
    weigh    s['wd3'] =     f.ge    _variable("wd3", shape=[4, 4, 4, 128, 256], ini    ializer=xavier_ini    )
    weigh    s['wd4'] =     f.ge    _variable("wd4", shape=[4, 4, 4, 256, 512], ini    ializer=xavier_ini    )    
    weigh    s['wd5'] =     f.ge    _variable("wd5", shape=[4, 4, 4, 512, 1], ini    ializer=xavier_ini    )    

    re    urn weigh    s

def ini    ialiseBiases():
    
    global biases
    zero_ini     =     f.zeros_ini    ializer()

    biases['bg1'] =     f.ge    _variable("bg1", shape=[512], ini    ializer=zero_ini    )
    biases['bg2'] =     f.ge    _variable("bg2", shape=[256], ini    ializer=zero_ini    )
    biases['bg3'] =     f.ge    _variable("bg3", shape=[128], ini    ializer=zero_ini    )
    biases['bg4'] =     f.ge    _variable("bg4", shape=[64], ini    ializer=zero_ini    )
    biases['bg5'] =     f.ge    _variable("bg5", shape=[1], ini    ializer=zero_ini    )

    biases['bd1'] =     f.ge    _variable("bd1", shape=[64], ini    ializer=zero_ini    )
    biases['bd2'] =     f.ge    _variable("bd2", shape=[128], ini    ializer=zero_ini    )
    biases['bd3'] =     f.ge    _variable("bd3", shape=[256], ini    ializer=zero_ini    )
    biases['bd4'] =     f.ge    _variable("bd4", shape=[512], ini    ializer=zero_ini    )    
    biases['bd5'] =     f.ge    _variable("bd5", shape=[1], ini    ializer=zero_ini    ) 

    re    urn biases

def     rainGAN(is_dummy=False, exp_id=None):

    weigh    s, biases =  ini    ialiseWeigh    s(), ini    ialiseBiases()
    x_vec    or =     f.placeholder(shape=[ba    ch_size,cube_len,cube_len,cube_len,1],d    ype=    f.floa    32) 
    z_vec    or =     f.placeholder(shape=[ba    ch_size,z_size],d    ype=    f.floa    32) 

    # Weigh    s for au    oencoder pre    raining
    xavier_ini     =     f.con    rib.layers.xavier_ini    ializer()
    zero_ini     =     f.zeros_ini    ializer()
    weigh    s['wae_d'] =     f.ge    _variable("wae_d", shape=[4, 4, 4, 512, 200], ini    ializer=xavier_ini    )
    biases['bae_d'] =      f.ge    _variable("bae_d", shape=[200], ini    ializer=zero_ini    )

    encoded = encoder(x_vec    or, phase_    rain=True, reuse=False)
    encoded =     f.maximum(    f.minimum(encoded, 0.99), 0.01)
    decoded = genera    or(encoded, phase_    rain=True, reuse=False) 

    decoded_    es     = genera    or(    f.maximum(    f.minimum(encoder(x_vec    or, phase_    rain=False, reuse=False), 0.99), 0.01), phase_    rain=False, reuse=False)

    # Round decoder ou    pu    
    decoded =     hreshold(decoded)
    # Compu    e MSE Loss and L2 Loss
    mse_loss =     f.reduce_mean(    f.pow(x_vec    or - decoded, 2))
    para_ae = [var for var in     f.    rainable_variables() if any(x in var.name for x in ['wg', 'wd', 'wae'])]
    for var in     f.    rainable_variables():
        if 'wd5' in var.name:
            las    _layer_dis = var
    para_ae.remove(las    _layer_dis)
    # l2_loss =     f.add_n([    f.nn.l2_loss(v) for v in para_ae])
    # ae_loss = mse_loss + reg_l2 * l2_loss
    ae_loss = mse_loss     

    op    imizer_ae =     f.    rain.AdamOp    imizer(learning_ra    e=ae_lr,be    a1=be    a, name="Adam_AE").minimize(ae_loss)
    # op    imizer_ae =     f.    rain.RMSPropOp    imizer(learning_ra    e=ae_lr, name="RMS_AE").minimize(ae_loss)


    ne    _g_    rain = genera    or(z_vec    or, phase_    rain=True, reuse=False) 

    d_ou    pu    _x = discrimina    or(x_vec    or, phase_    rain=True, reuse=False)
    d_ou    pu    _x =     f.maximum(    f.minimum(d_ou    pu    _x, 0.99), 0.01)
    summary_d_x_his     =     f.summary.his    ogram("d_prob_x", d_ou    pu    _x)

    d_ou    pu    _z = discrimina    or(ne    _g_    rain, phase_    rain=True, reuse=True)
    d_ou    pu    _z =     f.maximum(    f.minimum(d_ou    pu    _z, 0.99), 0.01)
    summary_d_z_his     =     f.summary.his    ogram("d_prob_z", d_ou    pu    _z)

    # Compu    e     he discrimina    or accuracy
    n_p_x =     f.reduce_sum(    f.cas    (d_ou    pu    _x > 0.5,     f.in    32))
    n_p_z =     f.reduce_sum(    f.cas    (d_ou    pu    _z <= 0.5,     f.in    32))
    d_acc =     f.divide(n_p_x + n_p_z, 2 * ba    ch_size)

    # Compu    e     he discrimina    or and genera    or loss
    d_loss = -    f.reduce_mean(    f.log(d_ou    pu    _x) +     f.log(1-d_ou    pu    _z))
    g_loss = -    f.reduce_mean(    f.log(d_ou    pu    _z))
    
    summary_d_loss =     f.summary.scalar("d_loss", d_loss)
    summary_g_loss =     f.summary.scalar("g_loss", g_loss)
    summary_n_p_z =     f.summary.scalar("n_p_z", n_p_z)
    summary_n_p_x =     f.summary.scalar("n_p_x", n_p_x)
    summary_d_acc =     f.summary.scalar("d_acc", d_acc)

    ne    _g_    es     = genera    or(z_vec    or, phase_    rain=False, reuse=True)

    para_g = [var for var in     f.    rainable_variables() if any(x in var.name for x in ['wg', 'bg', 'gen'])]
    para_d = [var for var in     f.    rainable_variables() if any(x in var.name for x in ['wd', 'bd', 'dis'])]

    # only upda    e     he weigh    s for     he discrimina    or ne    work
    op    imizer_op_d =     f.    rain.AdamOp    imizer(learning_ra    e=d_lr,be    a1=be    a).minimize(d_loss,var_lis    =para_d)
    # only upda    e     he weigh    s for     he genera    or ne    work
    op    imizer_op_g =     f.    rain.AdamOp    imizer(learning_ra    e=g_lr,be    a1=be    a).minimize(g_loss,var_lis    =para_g)

    saver =     f.    rain.Saver(max_    o_keep=50) 

    wi    h     f.Session() as sess:  
      
        sess.run(    f.global_variables_ini    ializer())        
        z_sample = np.random.normal(0, 0.33, size=[ba    ch_size, z_size]).as    ype(np.floa    32)
        if is_dummy:
            volumes = np.random.randin    (0,2,(ba    ch_size,cube_len,cube_len,cube_len))
            prin    ('Using Dummy Da    a')
        else:
            volumes = d.ge    All(obj=obj,     rain=True, is_local=is_local, obj_ra    io=obj_ra    io)
            prin    ('Using ' + obj + ' Da    a')
        volumes = volumes[...,np.newaxis].as    ype(np.floa    ) 

        for epoch in range(n_ae_epochs):
            idx = np.random.randin    (len(volumes), size=ba    ch_size)
            x = volumes[idx]

            # Au    oencoder pre    raining
            # ae_l, mse_l, l2_l, _ = sess.run([ae_loss, mse_loss, l2_loss, op    imizer_ae],feed_dic    ={x_vec    or:x})
            # prin     'Au    oencoder Training ', "epoch: ",epoch, 'ae_loss:', ae_l, 'mse_loss:', mse_l, 'l2_loss:', l2_l

            ae_l, mse_l, _ = sess.run([ae_loss, mse_loss, op    imizer_ae],feed_dic    ={x_vec    or:x})
            prin    ('Au    oencoder Training ', "epoch: ",epoch, 'ae_loss:', ae_l, 'mse_loss:', mse_l)

            # ou    pu     genera    ed chairs
            if epoch % ae_in    er == 10:
                idx = np.random.randin    (len(volumes), size=ba    ch_size)
                x = volumes[idx]
                decoded_chairs = sess.run(decoded_    es    , feed_dic    ={x_vec    or:x})
                if no     os.pa    h.exis    s(    rain_sample_direc    ory):
                    os.makedirs(    rain_sample_direc    ory)
                decoded_chairs.dump(    rain_sample_direc    ory+'/ae_' + exp_id +s    r(epoch))

        for epoch in range(n_epochs):
            
            idx = np.random.randin    (len(volumes), size=ba    ch_size)
            x = volumes[idx]
            z = np.random.normal(0, 0.33, size=[ba    ch_size, z_size]).as    ype(np.floa    32)

            # Upda    e     he discrimina    or and genera    or
            d_summary_merge =     f.summary.merge([summary_d_loss,
                                                summary_d_x_his    , 
                                                summary_d_z_his    ,
                                                summary_n_p_x,
                                                summary_n_p_z,
                                                summary_d_acc])

            summary_d, discrimina    or_loss = sess.run([d_summary_merge,d_loss],feed_dic    ={z_vec    or:z, x_vec    or:x})
            summary_g, genera    or_loss = sess.run([summary_g_loss,g_loss],feed_dic    ={z_vec    or:z})  
            d_accuracy, n_x, n_z = sess.run([d_acc, n_p_x, n_p_z],feed_dic    ={z_vec    or:z, x_vec    or:x})
            prin    (n_x, n_z)

            if d_accuracy < d_    hresh:
                sess.run([op    imizer_op_d],feed_dic    ={z_vec    or:z, x_vec    or:x})
                prin    ('Discrimina    or Training ', "epoch: ",epoch,', d_loss:',discrimina    or_loss,'g_loss:',genera    or_loss, "d_acc: ", d_accuracy)

            sess.run([op    imizer_op_g],feed_dic    ={z_vec    or:z})
            prin    ('Genera    or Training ', "epoch: ",epoch,', d_loss:',discrimina    or_loss,'g_loss:',genera    or_loss, "d_acc: ", d_accuracy)

            # ou    pu     genera    ed chairs
            if epoch % gan_in    er == 10:
                g_chairs = sess.run(ne    _g_    es    ,feed_dic    ={z_vec    or:z_sample})
                if no     os.pa    h.exis    s(    rain_sample_direc    ory):
                    os.makedirs(    rain_sample_direc    ory)
                g_chairs.dump(    rain_sample_direc    ory+'/'+s    r(epoch))
            
            if epoch % gan_in    er == 10:
                if no     os.pa    h.exis    s(model_direc    ory):
                    os.makedirs(model_direc    ory)      
                saver.save(sess, save_pa    h = model_direc    ory + '/' + s    r(epoch) + '.cp    k')

if __name__ == '__main__':
    is_dummy = bool(in    (sys.argv[1]))
    exp_id = sys.argv[2]
        rainGAN(is_dummy=is_dummy, exp_id=exp_id)
