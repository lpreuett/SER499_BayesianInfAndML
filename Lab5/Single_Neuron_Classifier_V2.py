'''
@Author: Larry Preuett
@Version: 12.8.2017
'''
import numpy
import csv

class Single_Neuron_Classifier:
    # USE ODD VALUES FOR K
    def __init__(self):
        self.__DATA_FILE_PATH = 'Iris_Dataset/bezdekIris.data'
        self.__data = []
        self.__debug = False
        self.__NUM_INPUT_DATA = 150  # number between 1 and 45211
        self.__input_weights = numpy.random.uniform(0, 100, 4).astype(float)
        self.__learning_rate = 0.4


        with open(self.__DATA_FILE_PATH, newline='') as dataFile:
            reader = csv.reader(dataFile, delimiter=',')
            for row in reader:
                self.__data.append(row)

        # convert data lists to numpy arrays
        self.__data = numpy.array(self.__data)
        # randomize data
        numpy.random.shuffle(self.__data)

        self.__replace_categorical_data()

        # convert array of strings into array of ints
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

    def __sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def __d_sigmoid(self, x):
        return x * (1 - x)

    def online_train(self, epochs=1, data_size=150):
        for epoch in range(epochs):
            print("Starting epoch {}".format(epoch+1))
            for i in range(data_size):
                datapoint = self.__data[i]
                # get model probability
                if self.__debug:
                    print("datapoint: {}".format(datapoint))
                    print("input_weights: {}".format(self.__input_weights))
                    print("datapoints . input_weights: {}".format(numpy.dot(datapoint[0:4], self.__input_weights)))
                output = self.__sigmoid(numpy.dot(datapoint[0:4], self.__input_weights))
                # get actual
                target = datapoint[4] # add y value to actual
                # calculate loss
                loss = target - output
                # calculate gradient
                gradient = -loss * datapoint[0:4].astype(float) # omit y value
                # calculate input weight deltas
                weight_deltas = -self.__learning_rate * gradient
                #update weights
                temp_weights = []
                for i in range(len(self.__input_weights)):
                    temp_weights.append(self.__input_weights[i] + weight_deltas[i])

                self.__input_weights = numpy.array(temp_weights)

                if self.__debug:
                    print("output: {}".format(output))
                    print("target: {}".format(target))
                    print("loss: {}".format(loss))
                    print("gradient: {}".format(gradient))
                    print("weight_deltas: {}".format(weight_deltas))
                    print("update weights: {}".format(self.__input_weights))

    def batch_train(self, num_batches=3, batch_size=50):
        for batch in range(num_batches):
            gradient = 0.0
            print("Starting batch {}".format(batch+1))
            for i in range(batch_size):
                datapoint = self.__data[i + batch_size * batch]
                # get model probability
                if self.__debug:
                    print("datapoint: {}".format(datapoint))
                    print("input_weights: {}".format(self.__input_weights))
                    print("datapoints . input_weights: {}".format(numpy.dot(datapoint[0:4], self.__input_weights)))
                batch_predict = self.__sigmoid(numpy.dot(datapoint[0:4], self.__input_weights))
                # get actual
                batch_target = datapoint[4]  # add y value to actual
                # calculate loss
                loss = batch_target - batch_predict
                # calculate gradient
                gradient += -loss * datapoint[0:4].astype(float)  # omit y value
                if self.__debug:
                    print("batch_output: {}".format(batch_predict))
                    print("batch_target: {}".format(batch_target))
                    print("loss: {}".format(loss))
                    print("gradient: {}".format(gradient))

            gradient_avg = gradient / batch_size
            # calculate input weight deltas
            weight_deltas = -self.__learning_rate * gradient_avg
            # update weights
            temp_weights = []
            for i in range(len(self.__input_weights)):
                temp_weights.append(self.__input_weights[i] + weight_deltas[i])

            self.__input_weights = numpy.array(temp_weights)

            print("Batch Results: ")
            print("Probabilities: {}".format(batch_predict))
            print("Loss: {}".format(loss))

            if self.__debug:
                print("weight_deltas: {}".format(weight_deltas))
                print("update weights: {}".format(self.__input_weights))

    def classify_dataset(self, dataset):
        correctly_classified = 0
        for i in range(len(dataset)):
            datapoint = dataset[i]
            output = round(self.__sigmoid(numpy.dot(datapoint[0:4], self.__input_weights)))
            if output == datapoint[4]:
                #correctly classified
                correctly_classified += 1

        # print results
        print("Correctly classified: {}".format(correctly_classified))
        print("Classifier Accuracy: {:.2f}".format(correctly_classified / len(dataset)))