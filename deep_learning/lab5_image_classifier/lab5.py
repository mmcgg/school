import tensorflow as tf
import numpy as np

def conv( x, filter_size=3, stride=2, num_filters=64, is_output=False, name="conv" ):
    '''
    x is an input tensor
    Declare a name scope using the "name" parameter
    Within that scope:
      Create a W filter variable with the proper size
      Create a B bias variable with the proper size
      Convolve x with W by calling the tf.nn.conv2d function
      Add the bias
      If is_output is False,
        Call the tf.nn.relu function
      Return the final op
    '''
    # The format of shape is [batch_size, height, width, channels]
    input_shape = x.get_shape().as_list()

    with tf.name_scope(name):
        W = tf.get_variable(name+"_W", shape=[filter_size,filter_size,input_shape[3],num_filters], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable(name+"_b", shape=num_filters)

        if(not is_output):
            h = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,W, strides=[1,stride,stride,1],padding="SAME"),b))
        else:
            h = tf.nn.bias_add(tf.nn.conv2d(x,W, strides=[1,stride,stride,1],padding="SAME"),b)

    return h

def fc( x, out_size=50, name="fc", is_output=False):
    '''
    x is an input tensor
    Declare a name scope using the "name" parameter
    Within that scope:
      Create a W filter variable with the proper size
      Create a B bias variable with the proper size
      Multiply x by W and add b
      If is_output is False,
        Call the tf.nn.relu function
      Return the final op
    '''
    input_shape = x.get_shape().as_list()

    with tf.name_scope(name):
        W = tf.get_variable(name+"_W", shape=[input_shape[1],out_size], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable(name+"_b", shape=[out_size])

        if(not is_output):
            h = tf.nn.relu(tf.nn.bias_add(tf.matmul(x,W),b))
        else:
            h = tf.nn.bias_add(tf.matmul(x,W),b)


    return h


def unpickle( file ):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

if __name__=='__main__':

    # Define the computation graph for training the DNN
    input_data = tf.placeholder( tf.float32, [1,32,32,3] )
    h0 = conv( input_data, name="conv1", stride=1)
    h1 = conv( h0, name="conv2", stride=2 )
    h2 = conv( h1, name="conv3", stride=2)
    h3 = conv( h2, name="conv4", stride=1)

    # Flatten to prepare for fully connected layer
    h3_flat = tf.reshape(h3,shape=[1,-1])

    h4 = fc(h3_flat, name="fc1", out_size=256)
    h5 = fc(h4, name="fc2", out_size=100)
    h6 = fc(h5, name="fc3", out_size=10, is_output=True)

    with tf.name_scope("loss_function"):
        true_label = tf.placeholder(tf.int64, 1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_label[0],logits=h6[0])


    with tf.name_scope("accuracy"):
        y = tf.nn.softmax(h6)
        prediction = tf.argmax(y,1)
        accuracy = tf.equal(prediction,true_label)


    training_step = tf.train.AdamOptimizer( 1e-5 ).minimize( loss )

    #----------------------------------END COMPUTATION GRAPH----------------------------------#

    # Load the data
    data = unpickle( 'cifar-10-batches-py/data_batch_1' )

    features = data['data']
    labels = data['labels']
    labels = np.atleast_2d( labels ).T

    train_features = features[0:8000]
    train_labels = labels[0:8000]
    val_features = features[8000:10000]
    val_labels = labels[8000:10000]

    print np.shape(train_features[0])

    # Run the DNN
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Declare things you want to visualize
    tf.summary.scalar('loss',loss)
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter("./tf_logs", sess.graph)

    for epoch in xrange(0,1000):

        for i in xrange(0,len(train_labels)):

            # Make a step
            '''
            print "\nPrediction: ", sess.run(prediction,{input_data:train_features[i].reshape(1,32,32,3),true_label:train_labels[i]})
            print "True label: ", sess.run(true_label,{input_data:train_features[i].reshape(1,32,32,3),true_label:train_labels[i]})
            print "h5: ", sess.run(h5,{input_data:train_features[i].reshape(1,32,32,3),true_label:train_labels[i]})
            print "loss: ", sess.run(loss,{input_data:train_features[i].reshape(1,32,32,3),true_label:train_labels[i]})
            '''
            sess.run(training_step, {input_data:train_features[i].reshape(-1,32,32,3),true_label:train_labels[i]})



            '''
            if i%1000==0:
                # Calculate accuracy
                acc = 0
                for j in xrange(0,len(val_features)):
                    my_y = sess.run(y,{input_data:val_features[j].reshape(1,32,32,3),true_label:val_labels[j]})
                    #print "\n\nMy y: ", np.argmax(my_y[0])
                    #print "Actual label: ", val_labels[j]
                    acc = acc + sess.run(accuracy,{input_data:val_features[j].reshape(1,32,32,3),true_label:val_labels[j]})
                    #print acc
                acc_percent = acc/(float(len(val_labels)))*100.0
                print acc_percent
            '''

            # For visualization
            my_scalars = sess.run(merged, {input_data:features[i].reshape(-1,32,32,3),true_label:labels[i]})

            writer.add_summary(my_scalars,i)
            #writer.add_summary(acc_percent[0],i)

        # Calculate accuracy
        acc = 0
        for j in xrange(0,len(val_features)):
            my_y = sess.run(y,{input_data:val_features[j].reshape(-1,32,32,3),true_label:val_labels[j]})
            #print "\n\nMy y: ", np.argmax(my_y[0])
            #print "Actual label: ", val_labels[j]
            acc = acc + sess.run(accuracy,{input_data:val_features[j].reshape(-1,32,32,3),true_label:val_labels[j]})
            #print acc
        acc_percent = acc/(float(len(val_labels)))*100.0
        print "Epoch: ", epoch
        print "Percent Accuracy: ", acc_percent

    writer.close()
