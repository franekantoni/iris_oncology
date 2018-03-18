"""
using data from Breast Cancer Wisconsin (Diagnostic) Data Set
About this Dataset
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server: ftp ftp.cs.wisc.edu cd math-prog/cpo-dataset/machine-learn/WDBC/

Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

Attribute Information:

1) ID number 2) Diagnosis (M = malignant, B = benign) 3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d) area e) smoothness (local variation in radius lengths) f) compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the contour) h) concave points (number of concave portions of the contour) i) symmetry j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

"""
from __future__ import division
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score, precision_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
import numpy as np



feature_labels = ['lable','radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 
'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 
'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

data_dict = {}
labels = []
features = []

with open("data.csv", 'r') as data:
	for line in data:
		person_dict = {}
		line = line.split(",")
		an_id = line[0]
		features.append(line[2:])
		line = line[1:]
		for lable, item in zip(feature_labels, line):
			if item == "B":
				item = 0
				labels.append(item)
			if item == "M":
				item = 1
				labels.append(item)
			person_dict[lable] = item
		data_dict[an_id] = person_dict
#print(data_dict)


features = features[1:]

features = np.array(features,  dtype=float)


"""
Reduces the number of features used by the model.
function takes train features, train labels and test features and returns transformed train and test features.
parameter k specifies the desired number of features
parameter print_msg specifies if system messages are to be printed
"""

def get_best_features(features_train, labels_train, features_test, k=10, print_msg= True):

	kbest = SelectKBest(k=k)
	kbest.fit(features_train, labels_train)
	features_train = kbest.transform(features_train)
	features_test = kbest.transform(features_test)
	if print_msg:
		ifso = kbest.get_support()
		ifso = list(ifso)
		print("FEATURES USED:")
		for i, l in zip(ifso, feature_labels[1:]):
			if i:
				print(l)
		print("\n")
	return features_train, features_test

"""
Trains two models: random forest and SVC

function takes train features, train labels, test features, test labels, 
nuber of iterations of evaluating models,
accuracy_test parameter specifies if testing should be performed
*models- names of sklearn models to train or test

returns list of trained clasifiers

"""

def train_models(features_train, labels_train, features_test, labels_test, accuracy_test = True, *models):

	trained_models = []

	if accuracy_test:

		scores = {}
		for model in models:
			scores[model] = 0

		for i in range(10):
			for model in models:
				clf = model()
				clf.fit(features_train, labels_train)
				pred = clf.predict(features_test)
				scores[model] += precision_score(labels_test, pred)

				if i == 0:
					trained_models.append(clf)


		for score in scores:

			print("\n")
			name = re.findall(r'\.\w+', str(score))
			print name[-1][1:], "score: ", scores[score]/10
			print("\n")
	else:
		for model in models:
			clf = model()
			clf.fit(features_train, labels_train)
			trained_models.append(clf)


	return trained_models


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.5, random_state=1)

features_train, features_test = get_best_features(features_train, labels_train, features_test)

lst_of_models = train_models(features_train, labels_train, features_test, labels_test, True, RandomForestClassifier, SVC)








