from Naive_Bayes_Classifier_V2 import Naive_Bayes_Classifier_V2
from KNN_Classifier_V2 import KNN_Classifier
from Single_Neuron_Classifier_V2 import Single_Neuron_Classifier
from Test_Dataset_V2 import Test_Dataset_V2

# initialize classifiers
bayes_classifier = Naive_Bayes_Classifier_V2()
knn_classifier = KNN_Classifier(8) # 8 - 10 give an accuracy of 98%
neuron_batch_classifier = Single_Neuron_Classifier()
neuron_online_classifier = Single_Neuron_Classifier()

# initialize test set
test_set = Test_Dataset_V2()

# get outputs
bayes_outputs = bayes_classifier.classify_data(test_set.getData())
knn_outputs = knn_classifier.classify_data(test_set.getData(), 150)

# train neuron classifiers
neuron_batch_classifier.batch_train(10, 15) # 10 batches with 15 points each - data set consists of 150 points
neuron_online_classifier.online_train(5, 150) # 5 epochs containing all the data points

# neuron classifier prints accuracy
print("Single Neuron Classifier with Batch Train Results: ")
neuron_batch_classifier.classify_dataset(test_set.getData())

print("Single Neuron Classifier with Online Train Results: ")
neuron_online_classifier.classify_dataset(test_set.getData())

# display bayes results
print("\nNaive Bayes Classifier Results:")
print("Accuracy: {:.2f}".format(test_set.getAccuracy(bayes_outputs)))

# display knn results
print("\nKNN Classifier Results:")
print("Accuracy: {:.2f}".format(test_set.getAccuracy(knn_outputs)))