import tensorflow as tf
import numpy as np

if __name__=='__main__':

    from skimage import io as skio
    from skimage.viewer import ImageViewer
    from skimage import transform
    import os
    import pickle

    # Get list of all images
    input_filenames = os.listdir('./cancer_data/inputs/')

    # Find how many test and how many train
    num_test = 0
    num_train = 0
    for i in xrange(0,len(input_filenames)):
        if( input_filenames[i][4:8]=='test' ):
            num_test = num_test + 1
        elif( input_filenames[i][4:9]=='train' ):
            num_train = num_train + 1
        else:
            print input_filenames[i]

    print num_test, " test cases"
    print num_train, " training cases"

    # Create empty tensors to hold all data
    test_images = np.zeros([num_test,512,512,3]).astype(np.float32)
    train_images = np.zeros([num_train,512,512,3]).astype(np.float32)
    test_labels = np.zeros([num_test,512,512]).astype(np.int8)
    train_labels = np.zeros([num_train,512,512]).astype(np.int8)

    # Fill the tensors with the data
    train_iterator = 0
    test_iterator = 0
    for i in xrange(0,len(input_filenames)):
        print i
        if( input_filenames[i][4:8]=='test' ):
            
            img = skio.imread('./cancer_data/inputs/'+input_filenames[i])
            label = skio.imread('./cancer_data/outputs/'+input_filenames[i])
            test_images[test_iterator,:,:,:] = transform.resize(img,[512,512,3])
            test_labels[test_iterator,:,:] = transform.resize(label,[512,512])
            test_iterator = test_iterator + 1
            
        elif( input_filenames[i][4:9]=='train' ):
    
            img = skio.imread('./cancer_data/inputs/'+input_filenames[i])
            label = skio.imread('./cancer_data/outputs/'+input_filenames[i])
            train_images[train_iterator,:,:,:] = transform.resize(img,[512,512,3])
            train_labels[train_iterator,:,:] = transform.resize(label,[512,512])
            train_iterator = train_iterator + 1            
    

    print "Dumping training images 1"
    pickle.dump(train_images[0:num_train/4],open('training_images1','wb'))
    print "Dumping training images 2"
    pickle.dump(train_images[num_train/4:2*num_train/4],open('training_images2','wb'))
    print "Dumping training images 3"
    pickle.dump(train_images[2*num_train/4:3*num_train/4],open('training_images3','wb'))
    print "Dumping training images 4"
    pickle.dump(train_images[3*num_train/4:4*num_train/4],open('training_images4','wb'))

    print "Dumping training labels 1"
    pickle.dump(train_labels[0:num_train/4],open('training_labels1','wb'))
    print "Dumping training labels 2"
    pickle.dump(train_labels[num_train/4:2*num_train/4],open('training_label2','wb'))
    print "Dumping training labels 3"
    pickle.dump(train_labels[2*num_train/4:3*num_train/4],open('training_labels3','wb'))
    print "Dumping training labels 4"
    pickle.dump(train_labels[3*num_train/4:4*num_train/4],open('training_labels4','wb'))

    print "Dumping test images"
    pickle.dump(test_images,open('testing_images','wb'))
    print "Dumping test labels"
    pickle.dump(test_labels,open('testing_labels','wb'))

            
    # Whiten the data
    print "Whitening the data"
    train_images = (train_images-np.mean(train_images,axis=0))/np.std(train_images,axis=0)
    test_images = (test_images-np.mean(train_images,axis=0))/np.std(train_images,axis=0)


    print "Dumping whitened training images1"
    pickle.dump(train_images[0:num_train/4],open('whitened_training_images1','wb'))

    print "Dumping whitened training images2"
    pickle.dump(train_images[num_train/4:2*num_train/4],open('whitened_training_images2','wb'))
    print "Dumping whitened training images3"            
    pickle.dump(train_images[2*num_train/4:3*num_train/4],open('whitened_training_images3','wb'))
    print "Dumping whitened training images4"            
    pickle.dump(train_images[3*num_train/4:4*num_train/4],open('whitened_training_images4','wb'))                


    print "Dumping whitened test images"
    pickle.dump(test_images,open('whitened_testing_images','wb'))





    
