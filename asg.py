'''
ASG is a framework for open-category problems

Author:
    Wei-Yang Qu quwy@lamda.nju.edu.cn
Date:
    2018.04.15
'''

import numpy as np
import copy
from gen_data import GenData
#from components import dealTesty,getPrecisionRecall,getNovelPrecisionRecall


'''
ASG process: 
        Genearte the positive and negative examples of each seen category by adversarial methods. 
        Determine the classification boundaries of each category.
        For the test instance:
            If it is within the classification boundary of the seen category, \
                then it should be divided into the corresponding category
            If it does not belong to any seen categories, then it should belong to novel class.
'''
class ASG:
    '''
    Init ASG parameters:
        classifier: the classification model in ASG
        classfilter: the data which has been filtered by category
    '''
    def __init__(self, classifier, classfilter):

        '''
            # Note that classifier needs to support sample classification with weight.
            # Otherwise, you need to modify the function 'train_Dminus' and 'train_Dplus' in the file gen_data.py 
        '''
        self.__classifier = classifier
        self.__classifier_list = []
        self.__classfilter = classfilter
        self.__gen_data_list = None
        self.__plus_label = None

    '''
    Generate positive and negative data for ASG:
        generate_size: the size of data to be generated in ASG
        budget: Zoopt parameter, budget size in racos
        sample_size: sample size in origin data when generating data
    '''
    def generate_data(self, generate_size, sample_size, budget = 100):
        gen_data_list = []
        classfilter = self.__classfilter
        # This step can be parallelized for each category
        for NUM in classfilter.getSeenClass():
            original_data = classfilter.getDatabyLabel()[NUM]
            gen_data = GenData(original_data[:sample_size,],class_num = NUM, generate_size = generate_size,classifier = self.__classifier, budget = budget)
            #print "Generate positive data of class ", NUM
            gen_data.generate_negative_data(dim_range = [0,1])
            #print "Generate negative data of class ", NUM
            gen_data.generate_positive_data(dim_range = [0,1])
            gen_data_list.append(gen_data)
        self.__gen_data_list = gen_data_list
        self.__plus_label = []
        for idx in range(len(self.__gen_data_list)):
            self.__plus_label.append(gen_data_list[idx].getClassNum())

    '''
    Predict label for open category data set
    '''
    def predict(self, test_X):
        #print 'test_X.shape', test_X.shape
        pred_prob = []
        for idx in range(len(self.__classifier_list)):
            classifier = self.__classifier_list[idx]
            proba_idx = classifier.predict_proba(test_X)[:,1]
            proba_idx = np.array(proba_idx).reshape(-1,)
            pred_prob.append(proba_idx)

        pred_prob = np.array(pred_prob)
        result = []
        #print pred_prob.shape
        for idx in range(pred_prob.shape[1]):
            prob_list = np.array(pred_prob[:,idx])

            # test data does not belongs to any seen class 
            if prob_list.max() < 0.5:
                result.append(-1)

            # test data belongs to the seen class with max prob
            else:
                label_idx = prob_list.argmax()
                #print"label_idx", label_idx
                result.append(self.__plus_label[label_idx])
        return np.array(result)


    '''
    Run the ASG algorithm, and get Macro-F1 for test data
    '''
    def train_classifier(self):
        plus_label = self.__plus_label
        data = []
        plus_data,minus_data = [],[]

        # get original data, plus data and minus data
        for i in range(len(self.__gen_data_list)):
            gen_data = self.__gen_data_list[i]
            data.append(gen_data.getOriginData())
            plus_data.append(gen_data.getGenPositiveData())
            minus_data.append(gen_data.getGenNegativeData())

        prob,res = [],[]
        P_total, R_total = [],[]
        novel_P, novel_R = [],[]
        err_p,err_n = [],[]

        # get classification model for each seen class
        for i in range(len(self.__plus_label)):
            cplus_X = np.array(data[i])
            cplus_X = np.concatenate((cplus_X,plus_data[i][:]))
            cminus_X = []
            for j in range(len(data)):
                if j == i :
                    continue;
                if len(cminus_X) == 0:
                    cminus_X = np.array(data[j])
                else:
                    cminus_X = np.concatenate((cminus_X,np.array(data[j])))

            # get the training data and label for plus_label[i]
            cminus_X = np.concatenate((cminus_X,np.array(minus_data[i])))
            cplus_y,cminus_y = np.zeros(cplus_X.shape[0]) + 1, np.zeros(cminus_X.shape[0]) - 1
            weight_plus = np.zeros(cplus_X.shape[0]) + 100.0/cplus_X.shape[0]
            weight_minus = np.zeros(cminus_X.shape[0]) + 100.0/cminus_X.shape[0]
            weight = np.concatenate((weight_plus,weight_minus))
            train_X = np.concatenate((cplus_X,cminus_X))
            train_y = np.concatenate((cplus_y,cminus_y))

            # get the classification model of plus_label[i]
            clf = copy.deepcopy(self.__classifier)
            clf.fit(train_X,train_y,sample_weight = weight)

            self.__classifier_list.append(clf)

        return 

    def run_ASG(self, generate_size, sample_size):
        self.generate_data(generate_size, sample_size)
        self.train_classifier()