from Naive_Bayes_Classifier_V2 import Naive_Bayes_Classifier_V2
from Test_Dataset_V2 import Test_Dataset_V2

classifier = Naive_Bayes_Classifier_V2()
test_set = Test_Dataset_V2()

outputs = classifier.classify_data(test_set.getData())

print('Naive Bayes Classifier V2 Accuracy: {:.2f}'.format(test_set.getAccuracy(outputs)))