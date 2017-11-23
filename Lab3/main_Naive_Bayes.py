'''
@Author: Larry Preuett
@Version: 11.22.2017
'''

import Naive_Bayes_Classifier
import Test_Dataset

print('Test Set:')
test_set = Test_Dataset.Test_Dataset()
print('Naive Bayes Classifier: ')
classifier = Naive_Bayes_Classifier.Naive_Bayes_Classifier()
outputs = classifier.classify_data(test_set.getData())

print('Naive Bayes Classifier Accuracy: {:.2f}'.format(test_set.getAccuracy(outputs)))