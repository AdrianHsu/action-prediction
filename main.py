
# coding: utf-8

# In[13]:


import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import util
import cv2
import os

import vgg
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[68]:


SAMPLE_VIDEO_TRAIN = 10
SAMPLE_VIDEO_TEST  = 25           # Sample this many frames per video
BATCH = 3   # Process this many videos concurrently
LR = 1e-3
C = 16
path = "/tmp3/4dr14nh5u/charades/Charades_v1_rgb/"
PRINT_EVERY = 1
EPOCHES = 1
crop_size = 224

# In[15]:


#label_dict = { 'c'+str(i).zfill(3) : i for i in range( 157 )}
label_dict = { "c106" : 0, "c107" : 1, "c108" : 2, "c109" : 3, "c110" : 4, "c111" : 5, 
               "c059" : 6, "c060" : 7, "c025" : 8, "c026" : 9, "c027" : 10, "c028" : 11, "c029" : 12,
               "c030" : 13, "c031" : 14, "c032" : 15 }


# In[16]:


#fps_dict = { l.strip().split(' ')[0] : float( l.strip().split(' ')[1] ) for l in open( "video_fps.txt" ) }


# In[17]:


#X = tf.placeholder( "float", [None, crop_size*crop_size*3] )
Y = tf.placeholder( "float", [None, C] )

vgg16 = vgg.VGG()
vgg16.build()

fc1 = util.fc( vgg16.pool5, C, "fc1" )
pre = tf.nn.softmax_cross_entropy_with_logits( logits = fc1, labels = Y )
loss= tf.reduce_mean( pre )


# In[19]:


train_op  = tf.train.AdamOptimizer( learning_rate = LR, epsilon=1e-8 ).minimize(loss)


# In[20]:


conf = tf.ConfigProto(
      gpu_options = tf.GPUOptions( allow_growth = True ),
      device_count = { 'GPU': 1 }
    )


# In[21]:


train_files = [l.strip() for l in open( "train.txt")]
test_files = [l.strip() for l in open( "test.txt" )]


# In[22]:


num_train = len(train_files)
num_test  = len(test_files)
test_vids = [l.split('|')[0] for l in test_files]


# In[82]:


def compute_mAP( data, labels, num_classes=4 ):

  mAP = 0.
  for i in range( num_classes ):
    # Sort the activations for class i by magnitude
    dd = data[ :, i ]
    ll = labels[:,i]
    idx = np.argsort( -dd )

    # True positives, False positives
    tp = (ll == 1).astype( np.int32 )
    fp = (ll == 0).astype( np.int32 )

    # Number of instances with label i
    num= np.sum( ll )

    # In case a class has been completely filtered by preprocessing:
    # For example c136: No frames before action that is to be predicted
    if num == 0:
      continue

    # Reorder according to the sorting
    tp = tp[idx]
    fp = fp[idx]

    tp = np.cumsum( tp ).astype( np.float32 )
    fp = np.cumsum( fp ).astype( np.float32 )

    prec = tp / (tp + fp)

    ap = 0.
    tmp = ll[idx]
    for j in range( data.shape[0] ):

      ap += tmp[j] * prec[j]
    ap /= num

    mAP += ap

  return mAP / num_classes


# In[23]:


def sample_points( ts, te, num ):
    assert( ts <= te )
    diff = te-ts
    pos = []
    for i in range( num ):    
        pos.append( ts + int(round(i*(diff/float(num)))) )
    return pos


# In[72]:


def get_frame(t, path, f):
    
    if t == 0:
        t = 1

    path_t = path + f + '/' + f + '-' + str(t).zfill(6) + '.jpg'
    ALL_COLOR = 1 # 0 is grey scale

    frame = cv2.imread(path_t, ALL_COLOR)
    #print(path_t)
    dmin = min( frame.shape[0], frame.shape[1] )
    ratio = 256.0/dmin
    # https://www.scivision.co/np-image-bgr-to-rgb/
    # frame = frame[...,::-1] # bgr -> rgb
    frame = cv2.resize( frame, None, fx=ratio, fy=ratio )
    x = frame.shape[0]
    y = frame.shape[1]
    x = (x - crop_size)//2
    y = (y - crop_size)//2
    crop = frame[ x : x+crop_size, y : y+crop_size, : ].astype( np.float32 )

    VGG_MEAN = np.array( [104., 117., 123.], dtype='float32')
    crop -= VGG_MEAN
    crop = crop.flatten()
    return crop


# In[73]:


def load_data(files, sample_size = 1, is_test = False):
    data = np.zeros((len(files) * sample_size, crop_size*crop_size*3), dtype=np.float32 )
    labels = np.zeros((len(files) * sample_size, C), dtype=np.float32 )
    
    for i, file in enumerate( files ):
        f = file.split('|')[0]
        classes = file.split('|')[1].split(';')
        ts = []
        te = []
        name = []

        for c in classes:
            name.append(c.split(' ')[0])
            ts.append(int(c.split(' ')[1]))
            te.append(int(c.split(' ')[2]))
        
        max_num = len(os.listdir(path + f + '/'))

        if is_test == False:
            pos = sample_points( ts[0], te[0], sample_size )
        else:
            pos = sample_points( 0, max_num - 1, sample_size )
        
        pos = np.array( pos )
        #pos = pos * (24.0 / fps_dict[ ff[0] ])
        #pos = pos // 4                          # Every 4-th frame was sampled
        pos = pos.astype( np.int32 )
        
        for j,t in enumerate( pos ):
            t = min( t, max_num-1 )
            
            data[i*sample_size + j] = get_frame(t, path, f)

            for c in name:
              labels[i*sample_size + j, label_dict[ c ]] += 1./len(name)
            
    return data, labels


# In[87]:


with tf.Session( config = conf ) as sess:
    tf.global_variables_initializer().run()

    for e in range( EPOCHES ):
        print("\nEpoch", str(e).zfill(3))

        # Train Phase.
        random.shuffle( train_files )
        cnt_iteration = 0
        for start, end in list( zip( range( 0, num_train, BATCH ), range( BATCH, num_train+1, BATCH ) )):
            
            data, labels = load_data( train_files[ start : end ], sample_size = SAMPLE_VIDEO_TRAIN )
            # data /= 127.5
            data = data.reshape((-1, 224, 224, 3))
            tf_op, tf_loss = sess.run( [train_op, loss], feed_dict = { vgg16.X : data, Y : labels } )
            
            print("Train Phase\t", "Iteration", str(start).zfill(5), "\tLoss", tf_loss)
            cnt_iteration += 1
        
        # Test Phase. Make sure not to shuffle, otherwise `test_vids` won't match.
        results  = np.zeros( (num_test, C) )
        gt       = np.zeros( (num_test, C) )

        test_range = list(zip( range( 0, num_test, BATCH ), range( BATCH, num_test+1, BATCH ) ))

        if num_test % BATCH != 0:
            test_range.append( (num_test-(num_test%BATCH), num_test) )


        for start, end in test_range:
            data, labels = load_data( test_files[ start : end ], sample_size = SAMPLE_VIDEO_TEST, is_test = True )
            data = data.reshape((-1, 224, 224, 3))

            tf_res = sess.run( fc1, feed_dict = { vgg16.X : data } )

            # During test phase, we pool the SAMPLE_VIDEO results of each video (averaging)
            tf_res = np.reshape( tf_res, [-1, SAMPLE_VIDEO_TEST, C] )
            tf_res = np.average( tf_res, axis=1 )
            labels = np.reshape( labels, [-1, SAMPLE_VIDEO_TEST, C] )[:,0,:]
            
            results[ start : end ]  = tf_res
            gt[ start : end ]       = labels

        mAP = compute_mAP( results, gt, num_classes = C )
#         save_to_eval( test_vids, results )

        print("Test Phase -- Mean AP: mAP =", mAP)

