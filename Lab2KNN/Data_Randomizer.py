'''
@Author: Larry Preuett
@Version: 11.18.2017

Description: Randomizes CSV datasets
'''

import csv
import numpy

first_row = True
data = []
data_labels = []


# replace file name with 'bank/bank-full.csv' to randomize bank-full dataset

with open('bank-additional/bank-additional-full.csv', newline='') as dataFile:
    reader = csv.reader(dataFile, delimiter=';')
    first_row = True
    for row in reader:
        # first line of file contains the labels
        if first_row:
            data_labels.append(row)
            first_row = False
        else:
            data.append(row)

# convert data lists to numpy arrays
data = numpy.array(data)
data_labels = numpy.array(data_labels)


numpy.random.shuffle(data)
data = numpy.insert(data, 0, data_labels[0], axis=0)
print(data)
# save file name and location
numpy.savetxt('bank-additional/bank-additional-full-randomized.csv', data, delimiter=';', fmt='%s')