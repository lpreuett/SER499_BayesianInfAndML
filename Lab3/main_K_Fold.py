'''
@Author: Larry Preuett
@Version: 11.22.2017
'''
import KNN_K_Fold_Classifier


classifier = KNN_K_Fold_Classifier.KNN_Classifier(5, 17)
classifier.k_fold_analysis()
