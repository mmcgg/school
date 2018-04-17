import tensorflow as tf
import numpy as np
from textloader import TextLoader

# Switch to CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




from tensorflow.python.ops.rnn_cell import RNNCell
 
class myGRU( RNNCell ):
    def __init__( self, state_dim ):
        self.state_dim = state_dim
        self.scope = None
    @property
    def state_size(self):
    	return self.state_dim
    @property
    def output_size(self):
    	return self.state_dim        
    def __call__( self, inputs, state):

        inputs_shape = inputs.shape.as_list()

        with tf.variable_scope("myGRU") as scope:        
            if self.scope==None:                

                Wz = tf.get_variable("Wz",[inputs_shape[1],self.state_dim],initializer=tf.variance_scaling_initializer())
                Uz = tf.get_variable("Uz",[self.state_dim,self.state_dim],initializer=tf.variance_scaling_initializer())
                bz = tf.get_variable("bz",[self.state_dim],initializer=tf.variance_scaling_initializer())
                
                Wr = tf.get_variable("Wr",[inputs_shape[1],self.state_dim],initializer=tf.variance_scaling_initializer())
                Ur = tf.get_variable("Ur",[self.state_dim,self.state_dim],initializer=tf.variance_scaling_initializer())
                br = tf.get_variable("br",[self.state_dim],initializer=tf.variance_scaling_initializer())

                Wh = tf.get_variable("Wh",[inputs_shape[1],self.state_dim],initializer=tf.variance_scaling_initializer())
                Uh = tf.get_variable("Uh",[self.state_dim,self.state_dim],initializer=tf.variance_scaling_initializer())
                bh = tf.get_variable("bh",[self.state_dim],initializer=tf.variance_scaling_initializer())
                self.scope = "myGRU"

            else:
                scope.reuse_variables()
                Wz = tf.get_variable("Wz")
                Uz = tf.get_variable("Uz")
                bz = tf.get_variable("bz")
                
                Wr = tf.get_variable("Wr")
                Ur = tf.get_variable("Ur")
                br = tf.get_variable("br")
                
                Wh = tf.get_variable("Wh")
                Uh = tf.get_variable("Uh")
                bh = tf.get_variable("bh")
            
            
    	z = tf.sigmoid(tf.matmul(inputs,Wz) + tf.matmul(state,Uz) + bz)
    	r = tf.sigmoid(tf.matmul(inputs,Wr) + tf.matmul(state,Ur) + br)

        h = z*state + (1-z)*tf.tanh(tf.matmul(inputs,Wh) + tf.matmul(r*state,Uh) + bh)

        return h,h


#
# -------------------------------------------
#
# Global variables

batch_size = 50
sequence_length = 50

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

num_layers = 2

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.  
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

inputs = tf.split(  in_onehot,sequence_length, 1 )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split(  targ_ph,sequence_length, 1 )

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

# ------------------
# YOUR COMPUTATION GRAPH HERE

cells = []
for i in xrange(0,num_layers):
    # For using an LSTM
    #cells.append(tf.nn.rnn_cell.BasicLSTMCell(state_dim))

    # For using a GRU
    #cells.append(tf.contrib.rnn.GRUCell(state_dim))

    # For using my GRU
    cells.append(myGRU(state_dim))
    
rnn = tf.nn.rnn_cell.MultiRNNCell(cells)

initial_state = rnn.zero_state(batch_size, tf.float32)


with tf.variable_scope("decoder"):
    outputs,final_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs,initial_state,rnn)


    
W = tf.get_variable("W",initializer=tf.random_normal([state_dim,vocab_size]))
b = tf.get_variable("b",initializer=tf.random_normal([vocab_size]))

logits = [tf.matmul(output,W) + [b]*batch_size for output in outputs]

loss_w = [1.0]*len(logits)

loss = tf.contrib.legacy_seq2seq.sequence_loss(logits,targets,loss_w)

optim = tf.train.AdamOptimizer().minimize(loss)


# ------------------
# YOUR SAMPLER GRAPH HERE
s_inputs = tf.placeholder( tf.int32, [1], name='s_inputs' )
s_in_onehot = tf.one_hot( s_inputs, vocab_size, name="s_input_onehot" )

s_inputs2 = tf.split( s_in_onehot, 1 )

s_initial_state = rnn.zero_state(1, tf.float32)


with tf.variable_scope("decoder"):
    s_outputs,s_final_state = tf.contrib.legacy_seq2seq.rnn_decoder(s_inputs2,s_initial_state,rnn)

s_outputs = tf.squeeze(s_outputs,0)
    
s_logits = tf.matmul(s_outputs,W) + b

s_probs = tf.nn.softmax(s_logits)

#
# ==================================================================
# ==================================================================
# ==================================================================
#

def sample( num=200, prime='ab' ):

    # prime the pump 

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run( s_initial_state )

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_inputs:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')

        # plug the most recent character in...
        feed = { s_inputs:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        #sample = np.argmax( s_probsv[0] )
        sample = np.random.choice( vocab_size, p=s_probsv[0] )

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches



for j in range(1000):

    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()

    for i in range( data_loader.num_batches ):
        
        x,y = data_loader.next_batch()

        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]

        ops = [optim,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%1000==0:
            print "%d %d\t%.4f" % ( j, i, lt )
            lts.append( lt )

    print sample( num=500, prime="\n And " )
#    print sample( num=500, prime="CORIOLANUS" )    
#    print sample( num=60, prime="ababab" )
#    print sample( num=60, prime="foo ba" )
#    print sample( num=60, prime="abcdab" )

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()
