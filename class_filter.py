'''
This file filters data by class

Author:
    Wei-Yang Qu  quwy@lamda.nju.edu.cn
Date:
    2018.04.15
'''

import numpy as np

'''
Data deal process: Filter data by class, X is the data features and y is the label array of data, 
which means that the label in y is seen class.
'''
class ClassFilter:

    def __init__(self, X, y, SeenClass = None):
        assert len(X) == len(y)                     
        self.__data = X     
        self.__label = y
        self.__length = len(y)                                  # data size
        self.__label_num = 0                                    # the label size of the training set
        self.__data_by_label = {}                               # data dictionary filtered by class
        self.__distinguish_label = []                           # the label of training set
        self.__seen_class = SeenClass                           # seen class in training set, the default is all classes in the training set

    '''
    Filter data by category
    '''
    def Filter(self):
        print "Start filter data by category!"
        data_by_label = self.__data_by_label

        # get the data of each category
        for idx in range(self.__length):
            idx_label = self.__label[idx]
            idx_data = self.__data[idx]
            if data_by_label.has_key(idx_label):
                data_by_label[idx_label].append(idx_data)
            else:
                data_by_label[idx_label] = []
                data_by_label[idx_label].append(idx_data)

        for key in data_by_label.keys():
            data_by_label[key] = np.array(data_by_label[key])

        self.__label_num = len(data_by_label.keys())
        self.__distinguish_label = data_by_label.keys()
        self.__data_by_label = data_by_label

        if self.__seen_class == None :
            self.__seen_class = self.__distinguish_label
        #print "Seen class num is:", self.__label_num
        #print "Seen class is:", self.__distinguish_label
        return data_by_label

    '''
    Get the filtered data
    '''
    def getDatabyLabel(self):
        return self.__data_by_label

    '''
    Get the data size
    '''
    def getLength(self):
        return self.__length

    '''
    Get the label in training set
    '''
    def getDistinctLabel(self):
        return self.__distinguish_label

    '''
    Get the label size in training set
    '''
    def getLabelNum(self):
        return self.__label_num

    '''
    Get the seen class in training set
    ''' 
    def getSeenClass(self):
        return self.__seen_class