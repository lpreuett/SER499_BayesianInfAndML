from KNN_Classifier_V2 import KNN_Classifier
from Test_Dataset_V2 import Test_Dataset_V2

classifier = KNN_Classifier(10)
test_set = Test_Dataset_V2()

outputs = classifier.classify_data(test_set.getData(), 150)

print('KNN Classifier V2 Accuracy: {:.2f}'.format(test_set.getAccuracy(outputs)))