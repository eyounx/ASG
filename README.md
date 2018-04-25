# ASG: Adversarial Sample Generation

This project implements the ASG algorithm in the paper: 

> Yang Yu, Wei-Yang Qu, Nan Li, and Zimin Guo. Open category classification by adversarial sample generation. In: Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI'17), Melbourne, Australia, 2017 ([PDF](http://lamda.nju.edu.cn/yuy/GetFile.aspx?File=papers/ijcai17-ASG-longer.pdf))

**Open category classification** aims at training a classifier that is aware of possible unseen classes at the test time. The classifier should tell *unseen* for a test instance, if this instance belongs to a class that never appeared in the training data. Open category classifier is much more robust for real-world classification tasks, where the environment is open and changing as always.

**ASG** achieves the open category classification goal by a two step training:

1. For each class in the training data, generate a set of boundary samples
2. For each class in the training data, train a classifier to tell between the boundary samples and the training data. This classifier approximates the boundary of the class.

At the test time, ASG uses intra-class classifiers of each class to classify a test instance. If no classifiers say this test instance belongs to it, ASG outputs *unseen*, otherwise ASG outputs the class with the highest confidence.

# Implementation

The implementation is in Python. 

It uses [ZOOpt](https://github.com/eyounx/ZOOpt/) as the optimizer. So ZOOpt is required to be installed.

[sklearn](http://scikit-learn.org) is used to provide the base classifier codes. As in the paper, SVM is used as the base classifier. Other classifiers can be used. But we need the classifier to be able to handle instance weights by supporting the `sample_weight` parameter in its `fit` function.

'run.py' is an example running ASG on MNIST datasets (decompress the zip file under 'data' fold before running). The seen classes can be set by the user.

'components.py' implements F1 score and other useful functions for data processing.

'class_filter.py' filters the training set by category.

'gen_data.py' implements the class GenData for data generating

'asg.py' is the main body of the ASG framework.


If you have any questions related to the codes, please feel free to contact: quwy@lamda.nju.edu.cn. 
