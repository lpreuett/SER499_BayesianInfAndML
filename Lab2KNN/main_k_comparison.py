import KNN_Classifier
import Test_Dataset

import matplotlib.pyplot as pyplot


print('KNN Classifier:')
# plot accuracy for k = 1, 2, 3, 4, 5, 10, 15, 20
k_vals = [1, 2, 3, 4, 5, 10, 15, 20]
classifiers = []
for k in range(0, len(k_vals)):
    classifiers.append(KNN_Classifier.KNN_Classifier(k_vals[k]))
print('Test_Dataset:')
test_data = []
for k in range(0, len(k_vals)):
    test_data.append(Test_Dataset.Test_Dataset())
    #standardize test_dataset
    test_data[k].standardize_data(classifiers[k].get_means(), classifiers[k].get_stdevs())
outputs = []
for c in range(0, len(classifiers)):
    #standardize data
    classifiers[c].standardize_data()
    outputs.append(classifiers[c].classify_data(test_data[c].getData()))

# get accuracies
accuracies = []
for o in range(0, len(outputs)):
    accuracies.append(test_data[o].getAccuracy(outputs[o]))

print('K Values: {}'.format(k_vals))
print('Accuracies: {}'.format(accuracies))

pyplot.figure('Model Accuracy against K Value')
pyplot.plot(k_vals, accuracies, 'rx')
pyplot.ylabel('Model Accuracy')
pyplot.xlabel('K Value')
pyplot.axis([0, 21, 0.75, 1.0])
pyplot.show()