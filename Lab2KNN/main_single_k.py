import KNN_Classifier
import Test_Dataset


print('KNN Classifier:')
classifier = KNN_Classifier.KNN_Classifier(5)
print('Test_Dataset:')
test_data = Test_Dataset.Test_Dataset()

# standardize data
classifier.standardize_data()

output = classifier.classify_data(test_data.getData())

print('Output of classifier: ' + str(output))
print("Accuracy of KNN Classifier: %.2f" % test_data.getAccuracy(output))
