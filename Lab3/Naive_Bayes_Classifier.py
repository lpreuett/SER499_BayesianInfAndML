'''
@Author: Larry Preuett
@Version: 11.22.2017
'''
import numpy
import csv
import Bank_Data_Enum
from scipy.stats import multivariate_normal

class Naive_Bayes_Classifier:
    # USE ODD VALUES FOR K
    def __init__(self):
        self.__DATA_FILE_PATH = 'bank-additional/bank-additional-full-randomized.csv'
        self.__data = []
        self.__data_labels = []
        self.__debug = False
        self.__NUM_INPUT_DATA = 41188  # number between 1 and 45211
        self.__data_no = []
        self.__data_yes = []
        self.means_no = []
        self.means_yes = []
        self.covar_no = None
        self.covar_yes = None

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
        # randomize data
        numpy.random.shuffle(self.__data)

        # keep only the first 7 columns of data
        self.__data = numpy.insert(self.__data[:, 0:7], 7, self.__data[:, 20], axis=1)
        self.__data_labels = numpy.insert(self.__data_labels[:, 0:7], 7, self.__data_labels[:, 20], axis=1)

        self.__replace_categorical_data()

        # convert array of strings into array of ints
        self.__data = self.__data.astype(int)

        # store yes/no datasets
        for i in range(0, self.__NUM_INPUT_DATA):
            if self.__data[i][7] == Bank_Data_Enum.Y.YES.value:
                self.__data_yes.append(self.__data[i])
            elif self.__data[i][7] == Bank_Data_Enum.Y.NO.value:
                self.__data_no.append(self.__data[i])

        #convert to numpy array
        self.__data_yes = numpy.array(self.__data_yes).astype(int)
        self.__data_no = numpy.array(self.__data_no).astype(int)


        # calculate means
        for k in range(0, 7):
            self.means_no.append(numpy.mean(self.__data_no[k]))
            self.means_yes.append(numpy.mean(self.__data_yes[k]))

        self.covar_no = numpy.cov(self.__data_no[:, 0:7], rowvar=False)
        self.covar_yes = numpy.cov(self.__data_yes[:, 0:7], rowvar=False)

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
        # job: type of job(categorical: "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
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
        # marital : marital status (categorical: "divorced","married","single","unknown";
        if marital_str == 'divorced':
            return Bank_Data_Enum.Marital.DIVORCED.value
        elif marital_str == 'married':
            return Bank_Data_Enum.Marital.MARRIED.value
        elif marital_str == 'single':
            return Bank_Data_Enum.Marital.SINGLE.value
        else:
            return Bank_Data_Enum.Marital.UNKNOWN.value

    def __replace_education(self, education_str):
        # education (categorical: "basic.4y","basic.6y","basic.9y","high.school",
        # "illiterate","professional.course","university.degree","unknown")
        if education_str == 'basic.4y':
            return Bank_Data_Enum.Education.BASIC_4Y.value
        elif education_str == 'basic.6y':
            return Bank_Data_Enum.Education.BASIC_6Y.value
        elif education_str == 'basic.9y':
            return Bank_Data_Enum.Education.BASIC_9Y.value
        elif education_str == 'high.school':
            return Bank_Data_Enum.Education.HIGH_SCHOOL.value
        elif education_str == 'illiterate':
            return Bank_Data_Enum.Education.ILLITERATE.value
        elif education_str == 'professional.course':
            return Bank_Data_Enum.Education.PROFESSIONAL_COURSE.value
        elif education_str == 'university.degree':
            return Bank_Data_Enum.Education.UNIVERSITY_DEGREE.value
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

    def classify_data(self, input):
        #print(multivariate_normal.pdf(input[:, 0:7], self.means_no, self.covar_no))
        #print(multivariate_normal.pdf(input[:, 0:7], self.means_no, self.covar_yes))
        print(input)
        #classify_input = numpy.vectorize(self.classify_point)
        #outputs = classify_input(input[:,0:7])
        outputs = []
        for point in input[:, 0:7]:
            outputs.append(self.classify_point(point))

        print(len(outputs))

        return outputs

    def classify_point(self, point):
        # calc probability of yes
        prior_p_yes = len(self.__data_yes) / self.__NUM_INPUT_DATA
        # product of probability of yes and probability of point given yes
        p_yes = prior_p_yes * multivariate_normal.pdf(point, self.means_yes, self.covar_yes)

        # calc probability of no
        prior_p_no = len(self.__data_no) / self.__NUM_INPUT_DATA
        # product of probability of no and probability of point given no
        p_no = prior_p_no * multivariate_normal.pdf(point, self.means_no, self.covar_no)

        if (p_yes > p_no):
            return Bank_Data_Enum.Y.YES.value
        else:
            return Bank_Data_Enum.Y.NO.value
