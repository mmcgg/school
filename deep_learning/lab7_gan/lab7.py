import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def conv( x, filter_size=3, stride=2, num_filters=64, is_output=False, name="conv", mypadding="SAME", dropout_p = 1.0,reuse_vars=False ):

    # The format of shape is [batch_size, height, width, channels]
    input_shape = x.get_shape().as_list()

    with tf.variable_scope(name+'vars'):
        if(reuse_vars):
            tf.get_variable_scope().reuse_variables()
        W = tf.get_variable(name+"_W", shape=[filter_size,filter_size,input_shape[3],num_filters], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable(name+"_b", shape=num_filters)

        if(not is_output):
            h = tf.contrib.layers.layer_norm(tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,W, strides=[1,stride,stride,1],padding=mypadding),b)),dropout_p),scope=tf.get_variable_scope())
        else:
            h = tf.nn.dropout(tf.nn.bias_add(tf.nn.conv2d(x,W, strides=[1,stride,stride,1],padding=mypadding),b),dropout_p)

    return h


def upconv( x, num_filters, stride=3, filter_size=3, is_output=False, name="upconv", mypadding="SAME", dropout_p = 1.0,reuse_vars=False):

    # The format of shape is [batch_size, height, width, channels]
    input_shape = np.array(x.get_shape().as_list(),dtype=np.int64)
    out_shape = [input_shape[0],input_shape[1]*stride,input_shape[1]*stride,num_filters]

    with tf.variable_scope(name+'vars'):
        if(reuse_vars):
            tf.get_variable_scope().reuse_variables()
        W = tf.get_variable(name+"_W", shape=[filter_size,filter_size,num_filters,input_shape[3]], initializer=tf.contrib.layers.variance_scaling_initializer())

        
        b = tf.get_variable(name+"_b", shape=num_filters)

        if(not is_output):
            h = tf.contrib.layers.layer_norm(tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(x,W,out_shape, strides=[1,stride,stride,1],padding=mypadding),b)),dropout_p),scope=tf.get_variable_scope())
        else:
            h = tf.nn.dropout(tf.nn.bias_add(tf.nn.conv2d_transpose(x,W,out_shape, strides=[1,stride,stride,1],padding=mypadding),b),dropout_p)

    return h

def fc( x, out_size=50, name="fc", is_output=False,reuse_vars=False):


    input_shape = x.get_shape().as_list()

    with tf.variable_scope(name+'vars'):
        if(reuse_vars):
            tf.get_variable_scope().reuse_variables()

        W = tf.get_variable(name+"_W", shape=[input_shape[0],input_shape[2],out_size], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable(name+"_b", shape=[out_size])

        if(not is_output):
            h = tf.contrib.layers.layer_norm(tf.nn.relu(tf.nn.bias_add(tf.matmul(x,W),b)),scope=tf.get_variable_scope())
        else:
            h = tf.nn.bias_add(tf.matmul(x,W),b)


    return h

# HARDCORE Resnet Generator
# Resnet Generator

def Generator(z,reuse_vars=False):    
    with tf.name_scope("Generator"):
        z_reshaped = tf.reshape(z,[BATCH_SIZE,1,100])

        g0 = fc(z_reshaped, out_size=8192*2, name="gfc1",reuse_vars=reuse_vars)
        g0_reshaped = tf.reshape(g0,[BATCH_SIZE,4,4,1024])

        g1 = upconv( g0_reshaped, name="gupconv1",filter_size=1, stride=2,num_filters=512,reuse_vars=reuse_vars,is_output=True )
        g2 = upconv( g1, name="gupconv2",filter_size=3, stride=1,num_filters=256,reuse_vars=reuse_vars )
        g2h = upconv( g2, name="gupconv2h",filter_size=3, stride=1,num_filters=256,reuse_vars=reuse_vars )
        
        g2h_concat = tf.concat([g1,g2h],axis=3,name="gpuconv2h_concat")

        g2i = upconv( g2h_concat, name="gupconv2i",filter_size=3, stride=1,num_filters=256,reuse_vars=reuse_vars )
        g2j = upconv( g2i, name="gupconv2j",filter_size=3, stride=1,num_filters=256,reuse_vars=reuse_vars )
        g2j_concat = tf.concat([g1,g2j],axis=3,name="gpuconv2j_concat")
        
        g3 = upconv( g2j_concat, name="gupconv3",filter_size=1, stride=2,num_filters=256,reuse_vars=reuse_vars,is_output=True )
        g4 = upconv( g3, name="gupconv4",filter_size=3, stride=1,num_filters=128,reuse_vars=reuse_vars )
        g4h = upconv( g4, name="gupconv4h",filter_size=3, stride=1,num_filters=128,reuse_vars=reuse_vars )
        g4h_concat = tf.concat([g3,g4h],axis=3,name="gpuconv4h_concat")

        g4i = upconv( g4h_concat, name="gupconv4i",filter_size=3, stride=1,num_filters=128,reuse_vars=reuse_vars )
        g4j = upconv( g4i, name="gupconv4j",filter_size=3, stride=1,num_filters=128,reuse_vars=reuse_vars )
        g4j_concat = tf.concat([g3,g4j],axis=3,name="gpuconv4j_concat")

        g5 = upconv( g4j_concat, name="gupconv5",filter_size=1, stride=2,num_filters=128,reuse_vars=reuse_vars,is_output=True )
        g6 = upconv( g5, name="gupconv6",filter_size=3, stride=1,num_filters=64,reuse_vars=reuse_vars )
        g6h = upconv( g6, name="gupconv6h",filter_size=3, stride=1,num_filters=64,reuse_vars=reuse_vars )
        g6h_concat = tf.concat([g5,g6h],axis=3,name="gpuconv6h_concat")

        g6i = upconv( g6h_concat, name="gupconv6i",filter_size=3, stride=1,num_filters=64,reuse_vars=reuse_vars )
        g6j = upconv( g6i, name="gupconv6j",filter_size=3, stride=1,num_filters=64,reuse_vars=reuse_vars )
        g6j_concat = tf.concat([g5,g6j],axis=3,name="gpuconv6j_concat")

        g7 = upconv( g6j_concat, name="gupconv7",filter_size=1, stride=2,num_filters=64,reuse_vars=reuse_vars, is_output=True )

        g8 = upconv( g7, name="gupconv8",filter_size=3, stride=1,num_filters=32,reuse_vars=reuse_vars )
        g8h = upconv( g8, name="gupconv8h",filter_size=3, stride=1,num_filters=32,reuse_vars=reuse_vars )
        g8h_concat = tf.concat([g7,g8h],axis=3,name="gpuconv8h_concat")

        g8i = upconv( g8h_concat, name="gupconv8i",filter_size=3, stride=1,num_filters=32,reuse_vars=reuse_vars )
        g8j = upconv( g8i, name="gupconv8j",filter_size=3, stride=1,num_filters=32,reuse_vars=reuse_vars )
        g8j_concat = tf.concat([g7,g8j],axis=3,name="gpuconv8j_concat")

        g9 = tf.nn.tanh(upconv( g8j_concat, name="gupconv9",filter_size=3, stride=1,num_filters=3,reuse_vars=reuse_vars, is_output=True ))

    return g9


'''
#DCGAN Generator
def Generator(z,reuse_vars=False):
    with tf.name_scope("Generator"):
        z_reshaped = tf.reshape(z,[BATCH_SIZE,1,100])

        g0 = fc(z_reshaped, out_size=8192*2, name="gfc1",reuse_vars=reuse_vars)
        g0_reshaped = tf.reshape(g0,[BATCH_SIZE,4,4,1024])
        
        g1 = upconv( g0_reshaped, name="gupconv2",filter_size=3, stride=2,num_filters=512,reuse_vars=reuse_vars )
        g2 = upconv( g1, name="gupconv3",filter_size=3, stride=2,num_filters=256,reuse_vars=reuse_vars )
        g3 = upconv( g2, name="gupconv4",filter_size=3, stride=2,num_filters=128,reuse_vars=reuse_vars )
        g4 = tf.nn.tanh(upconv( g3, name="gupconv5",filter_size=3, stride=2,num_filters=3,reuse_vars=reuse_vars, is_output=True ))

    return g4

'''
# HARDCORE Resnet Discriminator
def Discriminator(input_image,reuse_vars=False):
    with tf.name_scope("Discriminator"):
        d0 = conv( input_image, name="dconv0", filter_size=3, stride=1, num_filters=32,reuse_vars=reuse_vars )
        d1 = conv( d0, name="dconv1", filter_size=3, stride=1, num_filters=32,reuse_vars=reuse_vars )
        concat1 = tf.concat([input_image,d1],axis=3)

        d2 = conv( concat1, name="dconv2",filter_size=1, stride=2,num_filters=64,reuse_vars=reuse_vars, is_output=True )
        d3 = conv(d2, name="dconv3",filter_size=3, stride=1,num_filters=64,reuse_vars=reuse_vars )
        d4 = conv(d3, name="dconv4",filter_size=3, stride=1,num_filters=64,reuse_vars=reuse_vars )
        concat2 = tf.concat([d2,d4],axis=3)

        d5 = conv( concat2, name="dconv5",filter_size=1, stride=2,num_filters=128,reuse_vars=reuse_vars, is_output=True )
        d6 = conv(d5, name="dconv6",filter_size=3, stride=1,num_filters=128,reuse_vars=reuse_vars )
        d7 = conv(d6, name="dconv7",filter_size=3, stride=1,num_filters=128,reuse_vars=reuse_vars )
        concat3 = tf.concat([d5,d7],axis=3)

        d8 = conv(concat3, name="dconv8",filter_size=1, stride=2,num_filters=256,reuse_vars=reuse_vars, is_output=True )        
        
        # Flatten to prepare for fully connected layer
        d8_flat = tf.reshape(d8,shape=[BATCH_SIZE,1,-1])

        d9 = fc(d8_flat, out_size=100, name="dfc1",reuse_vars=reuse_vars)
        d10 = fc(d9, out_size=1, name="dfc2",reuse_vars=reuse_vars, is_output=True)

    return d10

'''
def Discriminator(input_image,reuse_vars=False):
    with tf.name_scope("Discriminator"):
        d0 = conv( input_image, name="dconv1", filter_size=3, stride=2, num_filters=64,reuse_vars=reuse_vars )
        d1 = conv( d0, name="dconv2",filter_size=3, stride=2,num_filters=128,reuse_vars=reuse_vars )
        d2 = conv( d1, name="dconv3",filter_size=3, stride=2,num_filters=256,reuse_vars=reuse_vars )
        d3 = conv( d2, name="dconv4",filter_size=3, stride=2,num_filters=512,reuse_vars=reuse_vars )
        d4 = conv( d3, name="dconv5",filter_size=3, stride=2,num_filters=1024,reuse_vars=reuse_vars )

        # Flatten to prepare for fully connected layer
        d4_flat = tf.reshape(d4,shape=[BATCH_SIZE,1,-1])

        d5 = fc(d4_flat, out_size=100, name="dfc1",reuse_vars=reuse_vars)
        d6 = fc(d5, out_size=1, name="dfc2",reuse_vars=reuse_vars, is_output=True)

    return d6
'''










if __name__=='__main__':

    from skimage import io as skio
    from skimage.viewer import ImageViewer
    from skimage import transform
    import os
    import pickle


    BATCH_SIZE = 5
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 64
    lambduh = 10
    ncritic = 1
    alpha = .0001
    beta1 = 0.5
    beta2 = .999

    # Define the computation graph for training the DNN----------------------------#
    input_image = tf.placeholder( tf.float32, [BATCH_SIZE,IMAGE_WIDTH,IMAGE_HEIGHT,3] )
    learning_rate = tf.placeholder( tf.float32, name="learning_rate")
    z = tf.placeholder( tf.float32, [BATCH_SIZE,100] )

    
    dout = Discriminator(input_image)
        
    generator_image = Generator(z)

    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    d_vars = []
    g_vars = []
    for i in xrange(0,len(all_vars)):
        if(all_vars[i].name[0]=='d'):
            d_vars.append(all_vars[i])
        if(all_vars[i].name[0]=='g'):
            g_vars.append(all_vars[i])

    with tf.name_scope("Train_Discriminator"):
        epsilon = tf.random_uniform(shape=[BATCH_SIZE,1,1,1], minval=0, maxval=1)
        xtilde = Generator(z,reuse_vars=True)
        xhat = tf.multiply(epsilon,input_image) + tf.multiply((1-epsilon),xtilde)

        gradients = tf.gradients(Discriminator(xhat,reuse_vars=True),xhat)
        D_x_tilde = Discriminator(xtilde,reuse_vars=True)
        D_x = Discriminator(input_image,reuse_vars=True)
        d_loss = tf.reduce_mean(D_x_tilde - D_x + lambduh*(tf.norm(gradients)-1)**2)
        train_d = tf.train.AdamOptimizer( learning_rate = learning_rate,
                                          beta1 = beta1,
                                          beta2 = beta2 ).minimize( d_loss,  var_list = d_vars )

    with tf.name_scope("Train_Generator"):
        G_z = Generator(z,reuse_vars=True)
        D_G_z = Discriminator(G_z,reuse_vars=True)
        g_loss = tf.reduce_mean(-D_G_z)
        train_g = tf.train.AdamOptimizer( learning_rate = learning_rate,
                                          beta1 = beta1,
                                          beta2 = beta2 ).minimize( g_loss,  var_list = g_vars )
    
    #----------------------------------END COMPUTATION GRAPH----------------------------------#






















    

    # Get list of all training images
    input_filenames = os.listdir('./img_align_celeba')

    # Find how many test and how many train
    num_train = len(input_filenames)

    print num_train, " training cases"

    # Switch to CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    
    # Restore all variables
    saver = tf.train.Saver()
    sess = tf.Session()
    #saver.restore(sess, "./tmp/epoch_27000.ckpt")

    # Initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    if(True):
        # These are the things we want summarized
        tf.summary.image('input_image',input_image)
        tf.summary.image('generator_image',generator_image)
        tf.summary.scalar('d_loss',d_loss)
        tf.summary.scalar('g_loss',g_loss)

        
        
        
        
        writer = tf.summary.FileWriter("./tf_logs", sess.graph)
        
        merged = tf.summary.merge_all()
        
        # These variables will be passed into the net
        my_x = np.zeros([BATCH_SIZE,IMAGE_WIDTH,IMAGE_HEIGHT,3]).astype(np.float32)
        
        rate = 2*1e-4


        
        for epoch in xrange(0,100000):
            print epoch

            # Train the Discriminator
            for t in xrange(0,ncritic):
                # Sample real data x
                train_image_indices = np.random.randint(0,num_train,size=BATCH_SIZE)
                for i in xrange(0,BATCH_SIZE):
                    # Get the training images
                    my_x[i,:,:,:] = transform.resize(skio.imread('./img_align_celeba/'+input_filenames[train_image_indices[i]]),[IMAGE_WIDTH,IMAGE_HEIGHT,3]).astype(np.float32)
                    my_x[i,:,:,:] = my_x[i,:,:,:]*2.0 - 1.0

            

                # Sample latent variable z
                my_z = np.random.normal(size=[BATCH_SIZE,100])
                sess.run(train_d, {z: my_z, input_image:my_x,learning_rate: rate})
            
            # Train the Generator
            my_z = np.random.normal(size=[BATCH_SIZE,100])
            sess.run(train_g, {z: my_z, input_image:my_x,learning_rate: rate})

        
            my_summaries = sess.run(merged, {z: my_z, input_image:my_x,learning_rate: rate})
            writer.add_summary(my_summaries,epoch)

            if(epoch%100==0):
                save_path = saver.save(sess, "./tmp/epoch_"+str(epoch)+".ckpt")
                print("Model saved in file: %s" % save_path)

        writer.close()        

    
    # Do z-space interpolation
    z1 = np.random.normal(size=[BATCH_SIZE,100])
    z2 = np.random.normal(size=[BATCH_SIZE,100])

    z1_hat = z1/np.linalg.norm(z1,axis=1).reshape(BATCH_SIZE,1)
    z2_hat = z2/np.linalg.norm(z2,axis=1).reshape(BATCH_SIZE,1)

    omega = np.zeros(BATCH_SIZE)
    for i in xrange(0,BATCH_SIZE):
        omega[i] = np.arccos(np.dot(z1_hat[i,:],z2_hat[i,:].T))


    z_interp = np.zeros([11,BATCH_SIZE,100])
    z_interp[0,:,:] = z1
    z_interp[10,:,:] = z2
    for i in xrange(1,10):
        t = i/10.0
        for j in xrange(0,BATCH_SIZE):
            z_interp[i,j,:] = np.sin((1-t)*omega[j])/np.sin(omega[j])*z1[j,:] + np.sin(t*omega[j])/np.sin(omega[j])*z2[j,:]


    # Run all of the interpolated z's through the generator and save the images
    image_interp = np.zeros(shape=[11,BATCH_SIZE,64,64,3])
    for i in xrange(0,11):
        
        image_interp[i,:,:,:] = sess.run(generator_image, {z: z_interp[i,:,:]})/2.0 + 0.5

    # Display all of the interpolated faces
    fig = plt.figure()
    for i in xrange(0,BATCH_SIZE):
        for j in xrange(0,11):
            fig.add_subplot(BATCH_SIZE,11,11*i+j+1)
            plt.imshow(image_interp[j,i,:,:])
            plt.axis('off')

    plt.show()

            
