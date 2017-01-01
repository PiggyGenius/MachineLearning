#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from scipy import interp

# We retrieve the dataset
data_file = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

# We get the values, 30 features, and the two classes. We use LabelEncoder to encode B = 0 and M = 1.
values = data_file.loc[:, 2:].values
classes = data_file.loc[:, 1].values
le = LabelEncoder()
classes = le.fit_transform(classes)

# We devide the dataset into a train and test set
train_values, test_values, train_classes, test_classes = train_test_split(values, classes, test_size = 0.20, random_state = 1)

###########################################################################################
#
# We will plot an ROC curve of a classifier that only uses two features from the Breast 
# Cancer Wisconsin dataset to predict whether a tumor is benign or malignant. Although we 
# are going to use the same logistic regression pipeline that we defined previously, we are
# making the classification task more challenging for the classifier so that the resulting 
# ROC curve becomes visually more interesting. For similar reasons, we are also reducing 
# the number of folds in the StratifiedKFold validator to three.
#
###########################################################################################

# We construct the LR pipeline and use PCA to only save two features
pipeline = Pipeline([('ss', StandardScaler()), ('pca',PCA(n_components = 2)), ('lr', LogisticRegression(penalty = 'l2', random_state = 0))])

# We create a subset on the training values and create the k-folds
train_values_2 = train_values[:, [4, 14]]
cross_validation = list(StratifiedKFold(n_splits = 3, random_state = 1).split(train_values,train_classes))

# We create the mean True Positive Rate and False Positive Rate variables
fig = plt.figure(figsize = (15, 10))
mean_TPR = 0.0
mean_FPR = np.linspace(0, 1, 100)
all_TPR = []

# Main loop to compute the ROC curve
for i, (train,test) in enumerate(cross_validation):
    probas = pipeline.fit(train_values_2[train],train_classes[train]).predict_proba(train_values_2[test])
    FPR, TPR, thresholds = roc_curve(train_classes[test],probas[:, 1], pos_label = 1)
    # We interpolate the average ROC curve from the three folds
    mean_TPR += interp(mean_FPR, FPR, TPR)
    mean_TPR[0] = 0.0
    # We compute the Are Under the Curve of the ROC curve
    roc_auc = auc(FPR, TPR)
    plt.plot(FPR, TPR, lw = 1, label = 'ROC fold {} (area = {:.2f})'.format(i+1, roc_auc))

plt.plot([0, 1],[0, 1], linestyle = '--', color = (0.6, 0.6, 0.6), label = 'Random guessing')
mean_TPR /= len(cross_validation)
mean_TPR[-1] = 1.0
mean_AUC = auc(mean_FPR, mean_TPR)
plt.plot(mean_FPR, mean_TPR, 'k--', label = 'Mean ROC (area = {:.2f})'.format(mean_AUC), lw = 2)
plt.plot([0, 0, 1], [0, 1, 1], lw = 2, linestyle = ':', color = 'black', label = 'Perfect Performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc = "lower right")
plt.tight_layout()
plt.show()

###########################################################################################
#
# We can also directly compute and print the ROC AUC score.
#
###########################################################################################

pipeline = pipeline.fit(train_values_2, train_classes)
predicted_classes = pipeline.predict(test_values[:, [4, 14]])
print('ROC AUC: {:.3f}'.format(roc_auc_score(y_true = test_classes,y_score = predicted_classes)))
print('Accuracy: {:.3f}'.format(accuracy_score(y_true = test_classes, y_pred = predicted_classes)))

###########################################################################################
#
# While the weighted macro-average is the default for multiclass problems in
# scikit-learn, we can specify the averaging method via the average parameter
# inside the different scoring functions that we import from the sklean.metrics
# module, for example, the precision_score or make_scorer functions.
# In python:
#   pre_scorer = make_scorer(score_func=precision_score, pos_label = 1, greater_is_better =                 True, average='micro')
#
###########################################################################################

