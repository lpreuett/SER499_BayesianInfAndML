'''
@author Larry Preuett
@version 11.3.2017

Citation Request:
  This dataset is public available for research. The details are described in [Moro et al., 2011].
  Please include this citation if you plan to use this database:

  [Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology.
  In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.

  Available at: [pdf] http://hdl.handle.net/1822/14838
                [bib] http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt

1. Title: Bank Marketing

2. Sources
   Created by: Paulo Cortez (Univ. Minho) and Sérgio Moro (ISCTE-IUL) @ 2012

3. Past Usage:

  The full dataset was described and analyzed in:

  S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology.
  In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães,
  Portugal, October, 2011. EUROSIS.

4. Relevant Information:

   The data is related with direct marketing campaigns of a Portuguese banking institution.
   The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required,
   in order to access if the product (bank term deposit) would be (or not) subscribed.

   There are two datasets:
      1) bank-full.csv with all examples, ordered by date (from May 2008 to November 2010).
      2) bank.csv with 10% of the examples (4521), randomly selected from bank-full.csv.
   The smallest dataset is provided to test more computationally demanding machine learning algorithms (e.g. SVM).

   The classification goal is to predict if the client will subscribe a term deposit (variable y).

5. Number of Instances: 45211 for bank-full.csv (4521 for bank.csv)

6. Number of Attributes: 16 + output attribute.

7. Attribute information:

   For more information, read [Moro et al., 2011].

   Input variables:
   # bank client data:
   1 - age (numeric)
   2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services")
   3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
   4 - education (categorical: "unknown","secondary","primary","tertiary")
   5 - default: has credit in default? (binary: "yes","no")
   6 - balance: average yearly balance, in euros (numeric)
   7 - housing: has housing loan? (binary: "yes","no")
   8 - loan: has personal loan? (binary: "yes","no")
   # related with the last contact of the current campaign:
   9 - contact: contact communication type (categorical: "unknown","telephone","cellular")
  10 - day: last contact day of the month (numeric)
  11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  12 - duration: last contact duration, in seconds (numeric)
   # other attributes:
  13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
  15 - previous: number of contacts performed before this campaign and for this client (numeric)
  16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

  Output variable (desired target):
  17 - y - has the client subscribed a term deposit? (binary: "yes","no")

8. Missing Attribute Values: None

'''
import numpy
import csv
import Bank_Data_Enum

class Test_Dataset:

    __DATA_FILE_PATH = 'bank/bank-full-randomized.csv'
    __debug = False

    def __init__(self):
        self.__data = []
        self.__data_labels = []
        with open(self.__DATA_FILE_PATH, newline='') as dataFile:
            reader = csv.reader(dataFile, delimiter=';')
            first_row = True
            for row in reader:
                # first line of file contains the labels
                if first_row:
                    self.__data_labels.append(row)
                    first_row = False
                else:
                    self.__data.append(row)

        # convert data lists to numpy arrays
        self.__data = numpy.array(self.__data)
        self.__data_labels = numpy.array(self.__data_labels)

        # keep only the first 7 columns of data
        temp_col_7 = self.__data[:, 7]
        temp_y = self.__data[:, 16]
        self.__data = numpy.insert(self.__data[:, 0:5], 5, self.__data[:, 6], axis=1)
        self.__data = numpy.insert(self.__data, 6, temp_col_7, axis=1)
        self.__data = numpy.insert(self.__data, 7, temp_y, axis=1)

        temp_label_col_7 = self.__data_labels[:, 7]
        temp_label_y = self.__data_labels[:, 16]
        self.__data_labels = numpy.insert(self.__data_labels[:, 0:5], 5, self.__data_labels[:, 6], axis=1)
        self.__data_labels = numpy.insert(self.__data_labels, 6, temp_label_col_7, axis=1)
        self.__data_labels = numpy.insert(self.__data_labels, 7, temp_label_y, axis=1)

        self.__replace_categorical_data()

        # convert string array into int array
        self.__data = self.__data.astype(int)

        # calculate number of y=yes and y=no
        yes = 0
        no = 0
        for entry in self.__data:
            if entry[7] == Bank_Data_Enum.Y.YES.value:
                yes+=1
            elif entry[7] == Bank_Data_Enum.Y.NO.value:
                no+=1
            else:
                raise Exception('Test_Dataset: incompatible value: ' + str(entry[7]) + ' Entry: ' + str(entry))

        print("Number of Y = Yes: %d" % yes)
        print("Number of Y = No: %d" % no)

        print(self.__data)
        print(self.__data_labels)


    def __replace_categorical_data(self):
        for row in self.__data:
            row[1] = self.__replace_job(row[1])
            row[2] = self.__replace_marital(row[2])
            row[3] = self.__replace_education(row[3])
            row[4] = self.__replace_default(row[4])
            row[5] = self.__replace_housing(row[5])
            row[6] = self.__replace_loan(row[6])
            row[7] = self.__replace_y(row[7])

    def __replace_job(self, job_str):
        #job: type of job(categorical: "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        # "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown")
        if job_str == 'admin':
            return Bank_Data_Enum.Job.ADMIN.value
        elif job_str == 'blue-collar':
            return Bank_Data_Enum.Job.BLUE_COLLAR.value
        elif job_str == 'entrepreneur':
            return Bank_Data_Enum.Job.ENTREPRENEUR.value
        elif job_str == 'housemaid':
            return Bank_Data_Enum.Job.HOUSEMAID.value
        elif job_str == 'management':
            return Bank_Data_Enum.Job.MANAGEMENT.value
        elif job_str == 'retired':
            return Bank_Data_Enum.Job.RETIRED.value
        elif job_str == 'self-employed':
            return Bank_Data_Enum.Job.SELF_EMPLOYED.value
        elif job_str == 'services':
            return Bank_Data_Enum.Job.SERVICES.value
        elif job_str == 'student':
            return Bank_Data_Enum.Job.STUDENT.value
        elif job_str == 'technician':
            return Bank_Data_Enum.Job.TECHNICIAN.value
        elif job_str == 'unemployed':
            return Bank_Data_Enum.Job.UNEMPLOYED.value
        else:
            return Bank_Data_Enum.Job.UNKNOWN.value

    def __replace_marital(self, marital_str):
        # marital : marital status (categorical: "married","divorced","single";
        if marital_str == 'divorced':
            return Bank_Data_Enum.Marital.DIVORCED.value
        elif marital_str == 'married':
            return Bank_Data_Enum.Marital.MARRIED.value
        elif marital_str == 'single':
            return Bank_Data_Enum.Marital.SINGLE.value
        else:
            return Bank_Data_Enum.Marital.UNKNOWN.value

    def __replace_education(self, education_str):
        # education (categorical: "unknown","secondary","primary","tertiary")
        if education_str == 'primary':
            return Bank_Data_Enum.Education.BASIC_4Y.value
        elif education_str == 'secondary':
            return Bank_Data_Enum.Education.BASIC_6Y.value
        elif education_str == 'tertiary':
            return Bank_Data_Enum.Education.BASIC_9Y.value
        else:
            return Bank_Data_Enum.Education.UNKNOWN.value

    def __replace_default(self, default_str):
        # default: has credit in default? (categorical: "no","yes","unknown")
        if default_str == 'no':
            return Bank_Data_Enum.Default.NO.value
        elif default_str == 'yes':
            return Bank_Data_Enum.Default.YES.value
        else:
            return Bank_Data_Enum.Default.UNKNOWN.value

    def __replace_housing(self, housing_str):
        # housing: has housing loan? (categorical: "no","yes","unknown")
        if housing_str == 'no':
            return Bank_Data_Enum.Housing.NO.value
        elif housing_str == 'yes':
            return Bank_Data_Enum.Housing.YES.value
        else:
            return Bank_Data_Enum.Housing.UNKNOWN.value

    def __replace_loan(self, loan_str):
        # loan: has personal loan? (categorical: "no","yes","unknown")
        if loan_str == 'no':
            return Bank_Data_Enum.Loan.NO.value
        elif loan_str == 'yes':
            return Bank_Data_Enum.Loan.YES.value
        else:
            return Bank_Data_Enum.Loan.UNKNOWN.value

    def __replace_y(self, y_str):
        if y_str == 'yes':
            return Bank_Data_Enum.Y.YES.value
        else:
            return Bank_Data_Enum.Y.NO.value

    def getData(self):
        return self.__data

    def getAccuracy(self, input_data):
        correctly_classified_count = 0
        data_y_col = self.__data[:, 7].astype(int)
        for i in range(0, len(input_data)):
            if input_data[i] == data_y_col[i]:
                correctly_classified_count += 1
            else:
                if self.__debug:
                    print('Incorrectly classified value at index %d input_data %d __data %d' % (i, input_data[i], data_y_col[i]))
        if self.__debug:
            print("Correctly classified values: %d" % correctly_classified_count)

        return correctly_classified_count / len(input_data)

    def standardize_data(self, means, stdevs):
        standardized_data = self.__data.astype(float)

        if self.__debug:
            print('means {}'.format(means))
            print('standard deviations {}'.format(stdevs))

        for c in range(0, len(self.__data[0]) - 1):
            standardized_data[:, c] -= means[c]
            standardized_data[:, c] /= stdevs[c]

        self.__data = standardized_data
