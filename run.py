'''
Demo for ASG

Author:
    Wei-Yang Qu  quwy@lamda.nju.edu.cn
Date:
    2018.04.15
'''

import numpy as np
from sklearn.svm import SVC
from asg import ASG
from class_filter import ClassFilter
from components import *


'''
Procedure:
        Input the origin data, and filter data by category.
        Generate positive and negative data of each class. 
        Use ASG method to get the margin of each class.
        Input test data, use the model given by Asg to decide which class the test data should be.
'''

if __name__ == '__main__':
    # an example of MNIST 
    (train_x, train_y), (valid_x, valid_y), (test_X, test_y) = np.load('data/mnist.pkl')
    '''
    Split data by classification message
    '''
    seen_class = [0,1,3]

    # seen class can be set here! default seen class is all class in train data
    cf = ClassFilter(train_x,train_y, SeenClass = seen_class)
    data_by_label = cf.Filter()
    print "Total label:", cf.getDistinctLabel()
    print "Seen class is:", seen_class
    '''
    # classifier model in ASG can be set here.
    # Note that classifier needs to support the parameter 'sample_weight' in the 'fit' function.
    # Otherwise, you need to modify the function 'train_Dminus' and 'train_Dplus' in the file gen_data.py 
    '''
    classifier_model = SVC(kernel='rbf',probability = True)

    # ASG method: initial
    asg = ASG(classifier=classifier_model, classfilter = cf)

    # run_ASG:
    # generate_size: the size of the sample you want to generate
    # sample_size: sample size in origin data when generating data
    asg.run_ASG(generate_size = 300, sample_size = 500)
    
    # predict for the test data with unseen class. If the test data belongs to unseen class, then output -1
    print("[ASG] performance on test data")
    result = asg.predict(test_X)

    # set unseen label to -1
    test_label = dealTesty(test_y,seen_class)
    get_macroF1(result,test_label)
