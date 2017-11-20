import KNN_Classifier_Identical_Train_Test_Dataset
import Test_Dataset

#***************DOES NOT RESULT IN ACCURACY = 1.0 BECAUSE ONE OR MORE VALUES EXIST WITH EQUAL VARIABLES
#***************BUT DIFFERENT VALUES OF Y

print('KNN Classifier:')
classifier = KNN_Classifier_Identical_Train_Test_Dataset.KNN_Classifier(1) # results 1.00 accuracy
print('Test_Dataset:')
test_data = Test_Dataset.Test_Dataset()

output = classifier.classify_data(test_data.getData())

print('Output of classifier: ' + str(output))
print("Accuracy of KNN Classifier: %.2f" % test_data.getAccuracy(output))
