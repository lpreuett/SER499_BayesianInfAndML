from enum import Enum

# bank client data:
class Job(Enum):
    # job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management",
    # "retired","self-employed","services","student","technician","unemployed","unknown")
    ADMIN = 0
    BLUE_COLLAR = 1
    ENTREPRENEUR = 2
    HOUSEMAID = 3
    MANAGEMENT = 4
    RETIRED = 5
    SELF_EMPLOYED = 6
    SERVICES = 7
    STUDENT = 8
    TECHNICIAN = 9
    UNEMPLOYED = 10
    UNKNOWN = 11


# bank client data:
class Marital(Enum):
    # marital : marital status (categorical: "divorced","married","single","unknown";
    # note: "divorced" means divorced or widowed)
    DIVORCED = 0
    MARRIED = 1
    SINGLE = 2
    UNKNOWN = 3


# bank client data:
class Education(Enum):
    # education (categorical: "basic.4y","basic.6y","basic.9y","high.school",
    # "illiterate","professional.course","university.degree","unknown")
    ILLITERATE = 0
    UNKNOWN = 1
    BASIC_4Y = 2
    BASIC_6Y = 3
    BASIC_9Y = 4
    HIGH_SCHOOL = 5
    PROFESSIONAL_COURSE = 6
    UNIVERSITY_DEGREE = 7

    # education(categorical: "unknown", "secondary", "primary", "tertiary")
    PRIMARY = 3
    SECONDARY = 4
    TERTIARY = 7 # assume tertiary refers to college coursework


# bank client data:
class Default(Enum):
    # default: has credit in default? (categorical: "no","yes","unknown")
    YES = 0
    UNKNOWN = 1
    NO = 2


# bank client data:
class Housing(Enum):
    # housing: has housing loan? (categorical: "no","yes","unknown")
    YES = 0
    UNKNOWN = 1
    NO = 2


# bank client data:
class Loan(Enum):
    # loan: has personal loan? (categorical: "no","yes","unknown")
    YES = 0
    UNKNOWN = 1
    NO = 2


# output data:
class Y(Enum):
    # Y
    YES = 1
    NO = 0
