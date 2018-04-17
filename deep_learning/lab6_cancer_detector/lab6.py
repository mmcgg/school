import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def conv( x, filter_size=3, stride=2, num_filters=64, is_output=False, name="conv", mypadding="SAME", dropout_p = .8 ):

    # The format of shape is [batch_size, height, width, channels]
    input_shape = x.get_shape().as_list()

    with tf.name_scope(name):
        W = tf.get_variable(name+"_W", shape=[filter_size,filter_size,input_shape[3],num_filters], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable(name+"_b", shape=num_filters)

        if(not is_output):
            h = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,W, strides=[1,stride,stride,1],padding=mypadding),b)),dropout_p)
        else:
            h = tf.nn.dropout(tf.nn.bias_add(tf.nn.conv2d(x,W, strides=[1,stride,stride,1],padding=mypadding),b),dropout_p)

    return h


def upconv( x, num_filters, stride=3, filter_size=3, is_output=False, name="upconv", mypadding="SAME", dropout_p = .8):

    # The format of shape is [batch_size, height, width, channels]
    input_shape = np.array(x.get_shape().as_list(),dtype=np.int64)
    out_shape = [input_shape[0],input_shape[1]*stride,input_shape[1]*stride,num_filters]

    with tf.name_scope(name):
        W = tf.get_variable(name+"_W", shape=[filter_size,filter_size,num_filters,input_shape[3]], initializer=tf.contrib.layers.variance_scaling_initializer())

        
        b = tf.get_variable(name+"_b", shape=num_filters)

        if(not is_output):
            h = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(x,W,out_shape, strides=[1,stride,stride,1],padding=mypadding),b)),dropout_p)
        else:
            h = tf.nn.dropout(tf.nn.bias_add(tf.nn.conv2d_transpose(x,W,out_shape, strides=[1,stride,stride,1],padding=mypadding),b),dropout_p)

    return h

def maxpool( x, pool_size=3, stride=2, name="maxpool"):

    
    input_shape = x.get_shape().as_list()

    with tf.name_scope(name):
        h = tf.nn.max_pool(x,
                           ksize=[1,pool_size,pool_size,1],
                           strides=[1,stride,stride,1],
                           padding="SAME")

    return h






















if __name__=='__main__':

    from skimage import io as skio
    from skimage.viewer import ImageViewer
    from skimage import transform
    import os
    import pickle


    BATCH_SIZE = 3
    IMAGE_SIZE = 128

    # Define the computation graph for training the DNN----------------------------#
    input_data = tf.placeholder( tf.float32, [BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3] )
    dropout_percentage = tf.placeholder( tf.float32)
    
    h0 = conv( input_data, name="conv1", filter_size=3, stride=1, num_filters=2, dropout_p=dropout_percentage)    
    h1 = conv( h0, name="conv2",filter_size=3, stride=1,num_filters=4, dropout_p=dropout_percentage )
    h1_and_half = conv(h1,name="conv2_and_half", filter_size=3, stride=1, num_filters=8, dropout_p=dropout_percentage)
    h2 = maxpool( h1_and_half, name="maxpool1", pool_size=3, stride=2)
    h3 = conv( h2, name="conv3",filter_size=3, stride=1, num_filters=16, dropout_p=dropout_percentage)
    h4 = conv( h3, name="conv4",filter_size=3, stride=1, num_filters=32, dropout_p=dropout_percentage)
    h4_and_half = conv(h4,name="conv1_and_half", filter_size=3, stride=1, num_filters=64, dropout_p=dropout_percentage)
    h5 = maxpool( h4_and_half, name="maxpool2", pool_size=3, stride=2)
    
    h6 = conv( h5, name="conv5", stride=1, num_filters=64, dropout_p=dropout_percentage)
    h7 = conv( h6, name="conv6", stride=1, num_filters=64, dropout_p=dropout_percentage)

    h8 = upconv( h7,stride=2,num_filters=64, name="upconv1", dropout_p=dropout_percentage)

    with tf.name_scope("Concat1"):
        h4_and_h8 = tf.concat([h4_and_half,h8],axis=3)
    
    h9 = conv( h4_and_h8, name="conv7", stride=1, num_filters=64, dropout_p=dropout_percentage)
    h10 = conv( h9, name="conv8", stride=1, num_filters=32, dropout_p=dropout_percentage)
    h10_and_half = conv( h10, name="conv8_and_half", stride=1, num_filters=16, dropout_p=dropout_percentage)
    h11 = upconv( h10_and_half,stride=2,num_filters=8, name="upconv2", dropout_p=dropout_percentage)

    with tf.name_scope("Concat2"):
        h1_and_h11 = tf.concat([h1_and_half,h11],axis=3)

    
    h12 = conv( h1_and_h11, name="conv9", stride=1, num_filters=8, dropout_p=dropout_percentage)
    h12_and_half = conv( h12, name="conv9_and_half", stride=1, num_filters=4, dropout_p=dropout_percentage)
    h13 = conv( h12_and_half, name="conv10", stride=1, num_filters=2, is_output=True, dropout_p=dropout_percentage)

    out_image_shape = h13.get_shape().as_list()

    mask1 = tf.reshape(h13[:,:,:,0],[out_image_shape[0],out_image_shape[1],out_image_shape[2],1])
    mask2 = tf.reshape(h13[:,:,:,1],[out_image_shape[0],out_image_shape[1],out_image_shape[2],1])

    mask3 = tf.cast(tf.reshape(tf.argmax(h13,axis=3),[out_image_shape[0],out_image_shape[1],out_image_shape[2],1]),tf.float32)

    labels = tf.placeholder( tf.int32, [BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE] )
    true_mask = tf.reshape(tf.cast(labels,tf.float32),[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,1])


    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=h13)
        loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(h13,3),tf.cast(labels,tf.int64)),tf.float32))
    
    
    learning_rate = tf.placeholder( tf.float32, name="learning_rate")
    training_step = tf.train.AdamOptimizer( learning_rate ).minimize( loss )
    
    #----------------------------------END COMPUTATION GRAPH----------------------------------#


    # Get list of all training images
    train_input_filenames = os.listdir('./cancer_data/inputs/train/pos_and_neg')
    test_input_filenames = os.listdir('./cancer_data/inputs/test/pos_and_neg')

    print test_input_filenames


    # Find how many test and how many train
    num_test = len(test_input_filenames)
    num_train = len(train_input_filenames)

    print num_test, " test cases"
    print num_train, " training cases"

    # Load the mean and std dev for whitening
    mean = pickle.load(open("mean_"+str(IMAGE_SIZE),"rb"))
    std_dev = pickle.load(open("std_dev_"+str(IMAGE_SIZE),"rb"))
    
    # Switch to CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    
    sess = tf.Session()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("./tf_logs", sess.graph)
    
    # Restore all variables
    #saver.restore(sess, "./tmp/epoch_120.ckpt")

    # Initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    

    

    # These are the things we want summarized
    tf.summary.image('input_data',input_data)
    tf.summary.image('true_mask',true_mask)
    tf.summary.image('mask1',mask1)
    tf.summary.image('mask2',mask2)
    tf.summary.image('mask3',mask3)
    
    tf.summary.scalar('loss',loss)
    merged = tf.summary.merge_all()

    test_accuracy = tf.summary.scalar('test_accuracy',accuracy)
    training_accuracy = tf.summary.scalar('training_accuracy',accuracy)

    # These variables will be passed into the net
    train_image_batch = np.zeros([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3]).astype(np.float32)
    train_label_batch = np.zeros([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE]).astype(np.int32)
    test_image_batch = np.zeros([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3]).astype(np.float32)
    test_label_batch = np.zeros([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE]).astype(np.int32)

    rate = 1e-3
    
    for epoch in xrange(0,100000):

        print epoch

        if(epoch%100==0):
            rate = rate*.9
            print "learning_rate: ",rate


        train_image_indices = np.random.randint(0,num_train,size=BATCH_SIZE)
        test_image_indices = np.random.randint(0,num_test,size=BATCH_SIZE)

    
        for i in xrange(0,BATCH_SIZE):

            # Get the training images
            train_image_batch[i,:,:,:] = transform.resize(skio.imread('./cancer_data/inputs/train/pos_and_neg/'+train_input_filenames[train_image_indices[i]]),[IMAGE_SIZE,IMAGE_SIZE,3])

            # Get the training labels
            train_label_batch[i,:,:] = np.array(transform.resize(skio.imread('./cancer_data/outputs/train/pos_and_neg/'+train_input_filenames[train_image_indices[i]]),[IMAGE_SIZE,IMAGE_SIZE])).astype(np.int32)


        train_image_batch = (train_image_batch-mean)/std_dev
            
        # Training step
        sess.run(training_step, {dropout_percentage:.9,input_data:train_image_batch,labels:train_label_batch,learning_rate: rate})


        for i in xrange(0,BATCH_SIZE):
            
            # Get the test images
            test_image_batch[i,:,:,:] = transform.resize(skio.imread('./cancer_data/inputs/test/pos_and_neg/'+test_input_filenames[test_image_indices[i]]),[IMAGE_SIZE,IMAGE_SIZE,3])
                
            # Get the test labels
            test_label_batch[i,:,:] = np.array(transform.resize(skio.imread('./cancer_data/outputs/test/pos_and_neg/'+test_input_filenames[test_image_indices[i]]),[IMAGE_SIZE,IMAGE_SIZE])).astype(np.int32)


        test_image_batch = (test_image_batch-mean)/std_dev

            
        # Run on the test set to get accuracies
        #my_summaries = sess.run(merged, {input_data:test_image_batch,labels:test_label_batch,learning_rate: rate})
            
        my_summaries = sess.run(merged, {dropout_percentage:1,input_data:test_image_batch,labels:test_label_batch,learning_rate: rate})

        test_acc = sess.run(test_accuracy, {dropout_percentage:1,input_data:test_image_batch,labels:test_label_batch,learning_rate: rate})
        train_acc = sess.run(training_accuracy, {dropout_percentage:1,input_data:train_image_batch,labels:train_label_batch,learning_rate: rate})
            
        writer.add_summary(my_summaries,epoch)
        writer.add_summary(test_acc,epoch)
        writer.add_summary(train_acc,epoch)

        if(epoch%10==0):


            save_path = saver.save(sess, "/tmp/epoch_"+str(epoch)+".ckpt")
            print("Model saved in file: %s" % save_path)

    writer.close()        

    
