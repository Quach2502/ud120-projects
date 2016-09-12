#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# create classifier
clf = GaussianNB()
# fit/train
t0 = time()
clf.fit(features_train[:len(features_train)/10], labels_train[:len(labels_train)/10])
print "training time:", round(time()-t0, 3), "s"
# predict
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"
# accuracy
accuracy = accuracy_score(labels_test, pred)
print accuracy





#########################################################
# your code goes here ###


#########################################################
