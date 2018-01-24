import util
import tensorflow as tf

class VGG( ):
  def __init__( self ):
    self.X = tf.placeholder( tf.float32, [None, 224, 224, 3] )

  def build( self ):
    with tf.variable_scope( "vgg" ) as scope:
      conv1_1 = tf.nn.relu( util.conv2d( self.X, [3,3,64], 'conv1_1' ) )
      conv1_2 = tf.nn.relu( util.conv2d( conv1_1 , [3,3,64], 'conv1_2' ) )
      pool1   = tf.nn.max_pool( conv1_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME' )

      conv2_1 = tf.nn.relu( util.conv2d( pool1, [3,3,128], 'conv2_1' ) )
      conv2_2 = tf.nn.relu( util.conv2d( conv2_1, [3,3,128], 'conv2_2' ) )
      pool2   = tf.nn.max_pool( conv2_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME' )

      conv3_1 = tf.nn.relu( util.conv2d( pool2, [3,3,256], 'conv3_1' ) )
      conv3_2 = tf.nn.relu( util.conv2d( conv3_1, [3,3,256], 'conv3_2' ) )
      conv3_3 = tf.nn.relu( util.conv2d( conv3_2, [3,3,256], 'conv3_3' ) )
      pool3   = tf.nn.max_pool( conv3_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME' )

      conv4_1 = tf.nn.relu( util.conv2d( pool3, [3,3,512], 'conv4_1' ) )
      conv4_2 = tf.nn.relu( util.conv2d( conv4_1, [3,3,512], 'conv4_2' ) )
      conv4_3 = tf.nn.relu( util.conv2d( conv4_2, [3,3,512], 'conv4_3' ) )
      pool4   = tf.nn.max_pool( conv4_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME' )

      conv5_1 = tf.nn.relu( util.conv2d( pool4, [3,3,512], 'conv5_1' ) )
      conv5_2 = tf.nn.relu( util.conv2d( conv5_1, [3,3,512], 'conv5_2' ) )
      conv5_3 = tf.nn.relu( util.conv2d( conv5_2, [3,3,512], 'conv5_3' ) )
      pool5   = tf.nn.max_pool( conv5_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME' )

    self.pool5 = pool5
