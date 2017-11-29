'''
@Author: Larry Preuett
@Version: 11.28.2017
'''

from Single_Neuron_Classifier import Single_Neuron_Classifier
from Test_Dataset import Test_Dataset

classifier = Single_Neuron_Classifier()
classifier.online_train(2, 5000)
classifier.batch_train(10, 1000)

test_set = Test_Dataset()

classifier.classify_dataset(test_set.getData()[0:5000])