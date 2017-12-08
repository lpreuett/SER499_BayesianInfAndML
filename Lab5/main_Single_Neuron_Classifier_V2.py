from Single_Neuron_Classifier_V2 import Single_Neuron_Classifier
from Test_Dataset_V2 import Test_Dataset_V2

classifier = Single_Neuron_Classifier()
test_set = Test_Dataset_V2()

# train
classifier.batch_train(100, 50)

# classify
classifier.classify_dataset(test_set.getData())
