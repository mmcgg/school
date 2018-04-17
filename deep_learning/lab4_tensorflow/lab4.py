import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#--------------------Task 1--------------------------------------------------#
with tf.name_scope("Task_1"):
    x = tf.placeholder(tf.float32,None, name="x")
    noise = tf.placeholder(tf.float32,None, name="noise")


    with tf.name_scope("noisy_line"):
        noisy_data = tf.add(tf.multiply(-6.7,x),2+noise)



xhat = np.random.uniform(-10,10,[100,1])
noise_hat = np.random.uniform(-1,1,[100,1])

# Run the computation graph
sess = tf.Session()
print(xhat,sess.run(noisy_data,{x: xhat, noise: noise_hat}))
writer = tf.summary.FileWriter("./tf_logs", sess.graph)
writer.close()



'''
def noisy_line(x):
    return -6.7 * x + 2 + np.random.uniform(-1, 1)



# Define a computation graph
x = tf.placeholder(tf.float32, None)
ytrue = tf.placeholder(tf.float32, None)
learning_rate = tf.placeholder(tf.float32, 1)

with tf.name_scope("layer1") as scope:
    m = tf.Variable(1e-3*np.random.randn(1,1).astype(np.float32), name="m")
    b = tf.Variable(1e-3*np.random.randn(1,1).astype(np.float32), name="b")
    y = tf.add(tf.multiply(x,m), b)

with tf.name_scope("loss_function") as scope:
    myerror = ytrue-y

with tf.name_scope("Modify weights") as scope:
    tf.assign(m,m+learning_rate*my_error)
    tf.assign(b,b+learning_rate*my_error)

#train_step = tf.train.AdamOptimizer(0.001).minimize(myerror)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

train_writer = tf.summary.FileWriter("./tf_logs", sess.graph)

#tf.summary.scalar('error',myerror)
#merged = tf.summary.merge_all()

if __name__=='__main__':

    # Generate and visualize some data
    xdata = []
    ydata = []

    for _ in range(100):
        x_hat = np.random.uniform(-10, 10)
        y_hat = noisy_line(x_hat)
        xdata.append(x_hat)
        ydata.append(y_hat)

    #plt.scatter(xdata,ydata)
    #plt.show()
    xdata = np.array(xdata).astype(np.float32).reshape(len(xdata),1)
    ydata = np.array(ydata).astype(np.float32).reshape(len(xdata),1)

    for i in xrange(0,100):
        #sess.run( train_step, feed_dict={x:xdata[i], y:ydata[i]})
        err = sess.run(myerror, feed_dict={x:xdata[i], y:ydata[i]})
        #train_writer.add_summary(ss,i)
        print( "%d %.2f" % ( i, err ) )

    train_writer.close()
'''
