
import numpy as np
import tensorflow as tf
import vgg16
from scipy.misc import imread, imresize

def np_gram_matrix(v):

    dim = np.shape(v)
    v = np.reshape(v, [dim[1] * dim[2], dim[3]])
    return np.matmul(v.T, v)


def tf_gram_matrix(v):
    assert isinstance(v, tf.Tensor)
    v.get_shape().assert_has_rank(4)

    dim = v.get_shape().as_list()
    v = tf.reshape(v, [dim[1] * dim[2], dim[3]])        
    return tf.matmul(v, v, transpose_a=True)

sess = tf.Session()

opt_img = tf.Variable( tf.truncated_normal( [1,224,224,3],
                                        dtype=tf.float32,
                                        stddev=1e-1), name='opt_img' )

tmp_img = tf.clip_by_value( opt_img, 0.0, 255.0 )

vgg = vgg16.vgg16( tmp_img, 'vgg16_weights.npz', sess )

style_img = imread( 'wonderwoman.jpg', mode='RGB' )
style_img = imresize( style_img, (224, 224) )
style_img = np.reshape( style_img, [1,224,224,3] )

content_img = imread( 'audrey_hepburn_bw.jpg', mode='RGB' )
content_img = imresize( content_img, (224, 224) )
content_img = np.reshape( content_img, [1,224,224,3] )

layers = [ 'conv1_1', 'conv1_2',
           'conv2_1', 'conv2_2',
           'conv3_1', 'conv3_2', 'conv3_3',
           'conv4_1', 'conv4_2', 'conv4_3',
           'conv5_1', 'conv5_2', 'conv5_3' ]

ops = [ getattr( vgg, x ) for x in layers ]

content_acts = sess.run( ops, feed_dict={vgg.imgs: content_img } )
style_acts = sess.run( ops, feed_dict={vgg.imgs: style_img} )

#
# --- construct your cost function here
#
# Content loss
opt_img_content_acts = vgg.conv4_2
L_content = .5*tf.reduce_mean(tf.pow(content_acts[8]-opt_img_content_acts,2))

# Style loss
gram_opt_11 = tf_gram_matrix(vgg.conv1_1)
gram_opt_21 = tf_gram_matrix(vgg.conv2_1)
gram_opt_31 = tf_gram_matrix(vgg.conv3_1)
gram_opt_41 = tf_gram_matrix(vgg.conv4_1)
gram_opt_51 = tf_gram_matrix(vgg.conv5_1)

gram_style_11 = np_gram_matrix(style_acts[0])
gram_style_21 = np_gram_matrix(style_acts[2])
gram_style_31 = np_gram_matrix(style_acts[4])
gram_style_41 = np_gram_matrix(style_acts[7])
gram_style_51 = np_gram_matrix(style_acts[10])

print gram_opt_11.shape
print gram_opt_21.shape
print gram_opt_31.shape
print gram_opt_41.shape
print gram_opt_51.shape
E1 = tf.reduce_mean(tf.pow(gram_opt_11-gram_style_11,2))/4.0/np.power(64,4)
E2 = tf.reduce_mean(tf.pow(gram_opt_21-gram_style_21,2))/4.0/np.power(128,4)
E3 = tf.reduce_mean(tf.pow(gram_opt_31-gram_style_31,2))/4.0/np.power(256,4)
E4 = tf.reduce_mean(tf.pow(gram_opt_41-gram_style_41,2))/4.0/np.power(512,4)
E5 = tf.reduce_mean(tf.pow(gram_opt_51-gram_style_51,2))/4.0/np.power(196,4)
L_style = E1 + E2 + E3 + E4 + E5

# Total loss
alpha = 1e-3
beta = 1.0
total_loss = alpha*L_content + beta*L_style

# Relevant snippets from the paper:
#   For the images shown in Fig 2 we matched the content representation on layer 'conv4_2'
#   and the style representations on layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1' and 'conv5_1'
#   The ratio alpha/beta was  1x10-3
#   The factor w_l was always equal to one divided by the number of active layers (ie, 1/5)

# --- place your adam optimizer call here
#     (don't forget to optimize only the opt_img variable)
train = tf.train.AdamOptimizer(1e-1).minimize(total_loss,var_list=opt_img)

# this clobbers all VGG variables, but we need it to initialize the
# adam stuff, so we reload all of the weights...
sess.run( tf.initialize_all_variables() )
vgg.load_weights( 'vgg16_weights.npz', sess )

# initialize with the content image
sess.run( opt_img.assign( content_img ))

writer = tf.summary.FileWriter("./tf_logs", sess.graph)
tf.summary.image('opt_img',opt_img)
merged = tf.summary.merge_all()

# --- place your optimization loop here
for i in xrange(0,6000):
    print "\n\nIteration: ",i
    print "Content Loss: ",sess.run(L_content)
    print "Style Loss: ",sess.run(L_style)
    print "Total Loss: ",sess.run(total_loss)    
    sess.run(train)
    my_summaries = sess.run(merged)
    writer.add_summary(my_summaries,i)
    if(i%10==0):
        temporary_image = sess.run( tmp_img )
        sess.run( opt_img.assign( temporary_image ))
        
    
