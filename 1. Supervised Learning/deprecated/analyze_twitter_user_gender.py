import csv
import math

from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Input fav_number, tweet_count [10, 15]
# Output gender [5]
# 20050

def integrate_gender(gender):
    if gender == "female": return 0
    elif gender == "male": return 1
    else: return 2

def printf_cv(name, scores):
    print("Cross-validating " + name + "... ", end = "")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def printf_score(name, classifier, input, output, testin, testout):
    print("Fitting and scoring " + name + "... ", end = "")
    classifier.fit(input, output)
    print(classifier.score(testin, testout))

with open("datasets/Twitter_User_Gender_Classification.csv", "r", encoding="utf8") as file:
    reader = csv.reader(file)
    csv_list = list(reader)
    csv_list = csv_list[1:] #Ignore first row of data

count = 0
setCount = 5
input_list = []
output_list = []
test_input = []
test_output = []

for record in csv_list:
    input_vector = []
    input_vector.append(int(record[10]))
    input_vector.append(int(record[15]))
    output_vector = integrate_gender(record[5])
    if (count <= 16040):
        input_list.append(input_vector)
        output_list.append(output_vector)
    else:
        test_input.append(input_vector)
        test_output.append(output_vector)
    count += 1

# Decision tree classifiers
AdaBoost = AdaBoostClassifier(n_estimators = 100)
NonBoost1 = tree.DecisionTreeClassifier(min_samples_leaf = 1);
NonBoost5 = tree.DecisionTreeClassifier(min_samples_leaf = 5);
NonBoost10 = tree.DecisionTreeClassifier(min_samples_leaf = 10);
NonBoost20 = tree.DecisionTreeClassifier(min_samples_leaf = 20);
NonBoost50 = tree.DecisionTreeClassifier(min_samples_leaf = 50);
NonBoost100 = tree.DecisionTreeClassifier(min_samples_leaf = 100);

# Support vector machine classifiers
SVC = svm.SVC();
SigmoidSVC = svm.SVC(kernel="sigmoid")
LinearSVC = svm.LinearSVC();

# k-nearest neighbors classifiers
KNN1 = KNeighborsClassifier(n_neighbors = 1)
KNN5 = KNeighborsClassifier(n_neighbors = 5)
KNN10 = KNeighborsClassifier(n_neighbors = 10)
KNN20 = KNeighborsClassifier(n_neighbors = 20)
KNN50 = KNeighborsClassifier(n_neighbors = 50)
KNN100 = KNeighborsClassifier(n_neighbors = 100)

# Nearest neighbor classifiers
NN = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(10, 4), random_state=1)

# Cross-validating phase
printf_cv("AdaBoost", cross_val_score(AdaBoost, input_list, output_list, cv = setCount))
printf_cv("NonBoost, min_samples=1", cross_val_score(NonBoost1, input_list, output_list, cv = setCount))
printf_cv("NonBoost, min_samples=5", cross_val_score(NonBoost5, input_list, output_list, cv = setCount))
printf_cv("NonBoost, min_samples=10", cross_val_score(NonBoost10, input_list, output_list, cv = setCount))
printf_cv("NonBoost, min_samples=20", cross_val_score(NonBoost20, input_list, output_list, cv = setCount))
printf_cv("NonBoost, min_samples=50", cross_val_score(NonBoost50, input_list, output_list, cv = setCount))
printf_cv("NonBoost, min_samples=100", cross_val_score(NonBoost100, input_list, output_list, cv = setCount))
printf_cv("KNN, k=1", cross_val_score(KNN1, input_list, output_list, cv = setCount))
printf_cv("KNN, k=5", cross_val_score(KNN5, input_list, output_list, cv = setCount))
printf_cv("KNN, k=10", cross_val_score(KNN10, input_list, output_list, cv = setCount))
printf_cv("KNN, k=20", cross_val_score(KNN20, input_list, output_list, cv = setCount))
printf_cv("KNN, k=50", cross_val_score(KNN50, input_list, output_list, cv = setCount))
printf_cv("KNN, k=100", cross_val_score(KNN100, input_list, output_list, cv = setCount))
printf_cv("NN", cross_val_score(NN, input_list, output_list, cv = setCount))
printf_cv("SVC", cross_val_score(SVC, input_list, output_list, cv = setCount))
printf_cv("SigmoidSVC", cross_val_score(SigmoidSVC, input_list, output_list, cv = setCount))
printf_cv("LinearSVC", cross_val_score(LinearSVC, input_list, output_list, cv = setCount))

# Normal fitting phase
printf_score("AdaBoost", AdaBoost, input_list, output_list, test_input, test_output)
printf_score("NonBoost, min_samples=1", NonBoost1, input_list, output_list, test_input, test_output)
printf_score("NonBoost, min_samples=5", NonBoost5, input_list, output_list, test_input, test_output)
printf_score("NonBoost, min_samples=10", NonBoost10, input_list, output_list, test_input, test_output)
printf_score("NonBoost, min_samples=20", NonBoost20, input_list, output_list, test_input, test_output)
printf_score("NonBoost, min_samples=50", NonBoost50, input_list, output_list, test_input, test_output)
printf_score("NonBoost, min_samples=100", NonBoost100, input_list, output_list, test_input, test_output)
printf_score("KNN, k=1", KNN1, input_list, output_list, test_input, test_output)
printf_score("KNN, k=5", KNN5, input_list, output_list, test_input, test_output)
printf_score("KNN, k=10", KNN10, input_list, output_list, test_input, test_output)
printf_score("KNN, k=20", KNN20, input_list, output_list, test_input, test_output)
printf_score("KNN, k=50", KNN50, input_list, output_list, test_input, test_output)
printf_score("KNN, k=100", KNN100, input_list, output_list, test_input, test_output)
printf_score("NN", NN, input_list, output_list, test_input, test_output)
printf_score("SVC", SVC, input_list, output_list, test_input, test_output)
printf_score("SigmoidSVC", SigmoidSVC, input_list, output_list, test_input, test_output)
printf_score("LinearSVC", LinearSVC, input_list, output_list, test_input, test_output)
