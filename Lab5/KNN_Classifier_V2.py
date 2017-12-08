'''
@author: Larry Preuett
@version: 12.8.2017
'''

### CONVERT DATA TO NUMPY.ARRAY
### CREATE ENUM VALUES OF EACH CATEGORICAL ATTRIBUTE
### UPDATE CATEGORICAL ATTRIBUTES WITH NUMERICAL EQUIVALENT

import numpy
import csv
from scipy.spatial import distance

class KNN_Classifier:

    # USE ODD VALUES FOR K
    def __init__(self, knn_k=1):
        self.__DATA_FILE_PATH = 'Iris_Dataset/bezdekIris.data'
        self.__data = []
        self.__debug = False
        self.__NUM_INPUT_DATA = 150  # number between 1 and 45211

        if knn_k > 0:
            self.k = knn_k
        else:
            self.knn_k = 1

        with open(self.__DATA_FILE_PATH, newline='') as dataFile:
            reader = csv.reader(dataFile, delimiter=',')
            for row in reader:
                self.__data.append(row)

        # convert data lists to numpy arrays
        self.__data = numpy.array(self.__data)
        # randomize data
        numpy.random.shuffle(self.__data)

        self.__replace_categorical_data()

        #convert array of strings into array of ints
        self.__data = self.__data.astype(float)

        print(self.__data)

    def __replace_categorical_data(self):
        for row in self.__data:
            row[4] = self.__replace_y(row[4])

    def __replace_y(self, y_str):
        if y_str == 'Iris-setosa':
            return 0
        elif y_str == 'Iris-versicolor':
            return 1
        elif y_str == 'Iris-virginica':
            return 2

    def __calc_distance(self, v1, v2):
        delta = distance.euclidean(v1[0:4], v2[0:4])
        if self.__debug:
            print('KNN_Classifier.__calc_distance(v1, v2):')
            print('v1: ' + str(v1[0:4]))
            print('v2: ' + str(v2[0:4]))
            print('distance: ' + str(delta))
        return abs(delta)

    def __calc_k_neighbors(self, input_entry):
        k_neighbors = numpy.empty([self.k]).astype(int)
        k_neighbors_delta = numpy.empty([self.k])
        furthest_neighbor_delta = 0.0
        furthest_neighbor_index = 0
        num_entries = 0

        for classifier_entry in self.__data:
            delta = self.__calc_distance(classifier_entry, input_entry)

            if num_entries < self.k:
                k_neighbors[num_entries] = classifier_entry[4]
                k_neighbors_delta[num_entries] = delta
                num_entries += 1
            elif delta < furthest_neighbor_delta:
                k_neighbors[furthest_neighbor_index] = classifier_entry[4]
                k_neighbors_delta[furthest_neighbor_index] = delta
                furthest_neighbor_delta = delta
            # update furthest_neighbor
            for i in range(0, len(k_neighbors)):
                if k_neighbors_delta[i] > furthest_neighbor_delta:
                    furthest_neighbor_index = i
                    furthest_neighbor_delta = k_neighbors_delta[i]

        print('k_neighbors after __calc_k_neighbors: ' + str(k_neighbors.astype(int)))
        return k_neighbors.astype(int)

    def __get_neighbor_classification(self, k_neighbors):
        setosa = 0
        versicolor = 0
        virginica = 0

        if self.__debug:
            print('KNN_Classifier.__get_neighbor_classification k_neighbor input: ' + str(k_neighbors))

        for i in range(0, len(k_neighbors)):
            if k_neighbors[i] == 0:
                setosa += 1
            elif k_neighbors[i] == 1:
                versicolor += 1
            elif k_neighbors[i] == 2:
                virginica += 1
            else:
                raise Exception("Invalid k_neighbors value found at index " + str(i) + ' k_neighbors value: ' + str(k_neighbors))

        if setosa >= virginica and setosa >= versicolor:
            return 0
        elif versicolor >= setosa and versicolor >= virginica:
            return 1
        else:
            return 2

    def classify_data(self, input_data, num_vals):
        if len(input_data[0]) != 5:
            raise Exception('Invalid data dimensions: n x 5 required. Data: {}'.format(input_data))

        classified_data = numpy.empty([num_vals])

        for i in range(0, num_vals):
            k_neighbors = self.__calc_k_neighbors(input_data[i])
            classification = self.__get_neighbor_classification(k_neighbors.astype(int))
            classified_data[i] = classification
            print("Step %d of %d" % (i+1, num_vals))


        if self.__debug:
            print('Output of KNN_Classifier.classified_data: ' + str(classified_data.astype(int)))

        return classified_data.astype(int)
