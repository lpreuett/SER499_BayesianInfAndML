'''
@author Larry Preuett
@version 12.7.2017
'''
import numpy
import csv

class Test_Dataset_V2:

    __DATA_FILE_PATH = 'Iris_Dataset/iris.data'
    __debug = False

    def __init__(self):
        self.__data = []

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

        # calculate counts for each class
        setosa = 0
        versicolor = 0
        virginica = 0
        for entry in self.__data:
            if entry[4].astype(int) == 0:
                setosa += 1
            elif entry[4].astype(int) == 1:
                versicolor += 1
            elif entry[4].astype(int) == 2:
                virginica += 1
            else:
                raise Exception('Test_Dataset: incompatible value: ' + str(entry[4]) + ' Entry: ' + str(entry))

        print("Number of Setosa: %d" % setosa)
        print("Number of Versicolor: %d" % versicolor)
        print("Number of Virginica: %d" % virginica)

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
        else:
            raise Exception('Test_Dataset: incompatible value: ' + str(y_str))

    def getData(self):
        return self.__data

    def getAccuracy(self, input_data):
        correctly_classified_count = 0
        data_y_col = self.__data[:, 4].astype(int)
        for i in range(0, len(input_data)):
            if input_data[i] == data_y_col[i]:
                correctly_classified_count += 1
            else:
                if self.__debug:
                    print('Incorrectly classified value at index %d input_data %d __data %d' % (i, input_data[i], data_y_col[i]))
        if self.__debug:
            print("Correctly classified values: %d" % correctly_classified_count)

        return correctly_classified_count / len(input_data)
