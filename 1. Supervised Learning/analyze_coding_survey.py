import csv
import math
import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from scipy import interp
from sklearn import datasets
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Input Age, Income, isEthnicMinority, isSoftwareDev [0, 48, 49, 51]
# Output Gender [36]
# 15620

def validate_age(age):
    if age == "NA": return False
    else: return True

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

with open("datasets/NEW2016_New_Coder_Survey.csv", "r", encoding="utf8") as file:
    reader = csv.reader(file)
    csv_list = list(reader)
    csv_list = csv_list[1:] #Ignore first row of data

setCount = 10
partition_input = []
partition_output = []
input_list = []
output_list = []
test_input = []
test_output = []

for record in csv_list:
    input_vector = []
    skip_record = False
    integer_gender = integrate_gender(record[36])
    if integer_gender == 2: skip_record = True
    if not validate_age(record[0]): skip_record = True
    if record[36] == "NA": skip_record = True
    if record[48] == "NA": skip_record = True
    if record[49] == "NA": skip_record = True
    if record[51] == "NA": skip_record = True

    if skip_record == False:
        input_vector.append(int(record[0]))
        input_vector.append(int(record[48]))
        input_vector.append(int(record[49]))
        output_vector = integer_gender
        input_vector.append(int(record[51]))
        partition_input.append(input_vector)
        partition_output.append(output_vector)

input_list = partition_input[:int(0.7 * len(partition_input))]
output_list = partition_output[:int(0.7 * len(partition_input))]
test_input = partition_input[int(0.7 * len(partition_input)):]
test_output = partition_output[int(0.7 * len(partition_input)):]

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
'''printf_cv("AdaBoost", cross_val_score(AdaBoost, input_list, output_list, cv = setCount))
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
printf_score("LinearSVC", LinearSVC, input_list, output_list, test_input, test_output)'''

# Analyze model complexity curve for NonBoost tree classifier, max_depth
'''file = open("coding_survey_nonboost_max_depth_results.csv", "w")
print("Beginning model complexity analysis for NonBoost...")
file.write("max_depth" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for max_depth in range(250):
    classifier = tree.DecisionTreeClassifier(max_depth = max_depth + 1)
    file.write(str(max_depth + 1) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    file.write(str(classifier.score(input_list, output_list)) + ", ")
    file.write(str(classifier.score(test_input, test_output)) + "\n")'''

# Analyze model complexity curve for AdaBoost tree classifier, n_estimators
'''file = open("coding_survey_adaboost_n_estimators_results.csv", "w")
print("Beginning model complexity analysis for AdaBoost...")
file.write("n_estimators" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for n_estimators in range(50):
    classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=(n_estimators + 1) * 10)
    file.write(str((n_estimators + 1) * 10) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    file.write(str(classifier.score(input_list, output_list)) + ", ")
    file.write(str(classifier.score(test_input, test_output)) + "\n")'''

# Analyze model complexity curve for KNN classifier
'''file = open("newknn_coding_survey_k_results.csv", "w")
print("Beginning model complexity analysis for KNN...")
file.write("k" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for k in range(250):
    classifier = KNeighborsClassifier(n_neighbors = k + 1)
    result = ""
    result += (str(k + 1) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    result += str(classifier.score(input_list, output_list)) + ", "
    result += str(classifier.score(test_input, test_output)) + "\n"
    print(result)
    file.write(result)'''

# Analyze model complexity curve for AdaBoost tree classifier, with max depth
'''file = open("NEWcoding_survey_adaboost_max_depth_results.csv", "w")
print("Beginning model complexity analysis for AdaBoost...")
file.write("max_depth" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for max_depth in range(100):
    classifier = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth + 1), n_estimators=50)
    result = ""
    result += (str(max_depth + 1) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    result += str(classifier.score(input_list, output_list)) + ", "
    result += str(classifier.score(test_input, test_output)) + "\n"
    print(result)
    file.write(result)'''

# Neural network ideal number of neurons
'''scaler = StandardScaler()
scaler.fit(input_list)
input_list = scaler.transform(input_list)
test_input = scaler.transform(test_input)

file = open("coding_survey_neural_network_layer_results.csv", "w")
print("Beginning model complexity analysis for NeuralNetwork...")
file.write("layers" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for neurons in range(100):
    layers = [neurons + 1]
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layers), random_state=1)
    result = ""
    result += (str(neurons + 1) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    result += str(classifier.score(input_list, output_list)) + ", "
    result += str(classifier.score(test_input, test_output)) + "\n"
    print(result)
    file.write(result)'''

# Neural network tuple length
'''scaler = StandardScaler()
scaler.fit(input_list)
input_list = scaler.transform(input_list)
test_input = scaler.transform(test_input)

file = open("coding_survey_neural_network_layer_length_results.csv", "w")
print("Beginning model complexity analysis for NeuralNetwork...")
file.write("layers" + ", " + "cross_val_score" + ", " + "training_score" + ", " + "testing_score\n")
for neurons in range(100):
    layers = []
    for neuron in range(neurons):
        layers.append(18)
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layers), random_state=1)
    result = ""
    result += (str(neurons + 1) + "," + str(cross_val_score(
    classifier, input_list, output_list, cv = setCount).mean()) + ", ")
    classifier.fit(input_list, output_list)
    result += str(classifier.score(input_list, output_list)) + ", "
    result += str(classifier.score(test_input, test_output)) + "\n"
    print(result)
    file.write(result)'''

# SVC Analysis
'''SVC = svm.SVC(); # rbf
SigmoidSVC = svm.SVC(kernel="sigmoid")
LinearSVC = svm.LinearSVC();

result = ""
result += ("RBF_SVC" + "," + str(cross_val_score(
SVC, input_list, output_list, cv = setCount).mean()) + ", ")
SVC.fit(input_list, output_list)
result += str(SVC.score(input_list, output_list)) + ", "
result += str(SVC.score(test_input, test_output)) + "\n"
print(result)
result = ""
result += ("Sigmoid_SVC" + "," + str(cross_val_score(
SigmoidSVC, input_list, output_list, cv = setCount).mean()) + ", ")
SigmoidSVC.fit(input_list, output_list)
result += str(SigmoidSVC.score(input_list, output_list)) + ", "
result += str(SigmoidSVC.score(test_input, test_output)) + "\n"
print(result)
result = ""
result += ("Linear_SVC" + "," + str(cross_val_score(
LinearSVC, input_list, output_list, cv = setCount).mean()) + ", ")
LinearSVC.fit(input_list, output_list)
result += str(LinearSVC.score(input_list, output_list)) + ", "
result += str(LinearSVC.score(test_input, test_output)) + "\n"
print(result)'''

#scaler = StandardScaler()
#scaler.fit(input_list)
#input_list = scaler.transform(input_list)
#test_input = scaler.transform(test_input)

'''layers = []
for i in range(14):
    layers.append(18)

# Learning curves
file = open("coding_survey_learning_curve_data.csv", "w")
print("Beginning learning curve analysis...")
file.write("input_size" + ", " + "cv_dt" + ", " + "cv_ab" + ", " + "cv_kn" + ", " + "cv_n" + ", " + "cv_svc" + ", " + "dt" + ", " + "ab" + ", " + "kn" + ", " + "n" + ", " + "svc\n")
for input_size in range(1, int(len(input_list) / 100)):
    input_partition = input_list[:input_size * 100]
    input_nn_partition = input_list[:input_size * 100]
    scaler.fit(input_nn_partition)
    input_nn_partition = scaler.transform(input_nn_partition)
    output_partition = output_list[:input_size * 100]
    output = str(input_size * 100) + ", "
    DT = tree.DecisionTreeClassifier(max_depth = 6)
    AB = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=8), n_estimators=50)
    KN = KNeighborsClassifier(n_neighbors = 20)
    N = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(layers), random_state=1)
    SV = svm.SVC(kernel="sigmoid")
    output += str(cross_val_score(DT, input_partition, output_partition, cv = setCount).mean()) + ", "
    output += str(cross_val_score(AB, input_partition, output_partition, cv = setCount).mean()) + ", "
    output += str(cross_val_score(KN, input_partition, output_partition, cv = setCount).mean()) + ", "
    output += str(cross_val_score(N, input_nn_partition, output_partition, cv = setCount).mean()) + ", "
    output += str(cross_val_score(SV, input_partition, output_partition, cv = setCount).mean()) + ", "
    DT.fit(input_partition, output_partition)
    AB.fit(input_partition, output_partition)
    KN.fit(input_partition, output_partition)
    N.fit(input_nn_partition, output_partition)
    SV.fit(input_partition, output_partition)
    output += str(DT.score(input_partition, output_partition)) + ", "
    output += str(AB.score(input_partition, output_partition)) + ", "
    output += str(KN.score(input_partition, output_partition)) + ", "
    output += str(N.score(input_nn_partition, output_partition)) + ", "
    output += str(SV.score(input_partition, output_partition)) + "\n"
    print(output)
    file.write(output)'''

layers = []
for i in range(14):
    layers.append(18)

file = open("roc_curve_data_svc.csv", "w")
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
lw = 2
color = "red"
NB1 = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=8), n_estimators=50)
preds = NB1.fit(input_list, output_list).predict(test_input)
probs = NB1.fit(input_list, output_list).predict_proba(test_input)
cm = confusion_matrix(test_output, preds)
print(cm)
'''fpr, tpr, thresholds = roc_curve(test_output, probs[:, 1])
file.write("tpr" + ", " + "fpr\n")
for i in range(len(fpr)):
    file.write(str(tpr[i]) + ", " + str(fpr[i]) + "\n")
mean_tpr += interp(mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold (area = %0.2f)' % (roc_auc))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('TPR vs. FPR (Coding Survey)')
plt.legend(loc="lower right")
plt.show()'''
