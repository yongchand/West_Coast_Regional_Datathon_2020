# Code originated from ML CLass by Yi-Chieh Wu
# Code for InfoBoss UIBR Project

# python modules
import os
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import glob
import warnings
from sklearn.externals import joblib
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

#preprocessing
from sklearn import impute
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification

# numpy, pandas, and sklearn modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, \
    LeaveOneOut, LeavePOut, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn import metrics
from sklearn.externals import joblib
from skmultilearn.model_selection import iterative_train_test_split
# local ML modules
#from datasets import get_dataset TODO: Implement dataset
import classifiers
#from constant import DI_LABELS_CSV, PROTPARAM_FEATURES, SLIDING_WIN_FEATURES, EMBEDDING_5_7_FEATURES
#from learning_curve import plot_learning_curve
#import preprocessors as preprocessors


######################################################################
# globals
######################################################################

# no magic numbers in code

N_ITER = 100    # number of parameter settings sampled (trade-off runtime vs quality)
CV_train = StratifiedKFold(n_splits=10, random_state=0)       # number of folds in cross-validation
CV_lc = StratifiedKFold(n_splits=10, random_state=0)

######################################################################
# functions
######################################################################

def make_pipeline(classifier, n, d):
    """Make ML pipeline.
    Parameters
    --------------------
        classifier        -- classifier, string
        n                 -- number of samples, int
        d                 -- number of features, int
    """
    steps = []
    param_grid = {}

    # apply preprocessing (imputation) and append to steps
    transformer = impute.SimpleImputer(strategy='constant', fill_value=0)
    steps.append(("Imputation", transformer))

    # get classifier and hyperparameters to tune using cross-validation
    clf = getattr(classifiers, classifier)(n,d)
    name = type(clf).__name__
    transform =OneVsRestClassifier(clf.estimator_)
    steps.append((name, transform))
    for key, val in clf.param_grid_.items():
        param_grid[name + "__" + key] = val

    # stitch together preprocessors and classifier
    pipe = Pipeline(steps)
    return pipe, param_grid

def report_metrics(y_true, y_pred, labels=None, target_names=None):
    ## have to check dimension of y_true and y_pred
    """Report main classification metrics.
    Parameters
    --------------------
        y_true       -- ground truth (correct) target values, array of shape (n_samples, n_lables)
        y_pred       -- estimated target values returned by classifier, array of shape (n_samples, n_labels)
        labels       -- list of label indices to include in report, list of strings
        target_names -- display names matching labels (same order), list of strings
    Return
    --------------------
        C      -- confusion matrix, see sklearn.metrics.confusion_matrix
        a      -- accuracy score, see sklearn.metrics.accuracy_score
        p      -- precision score, see sklearn.metrics.precision_score
        r      -- recall score, see sklearn.metrics.recall_score
        f1     -- f1 score, see sklearn.metrics.f1_score
    """

    # confusion matrix, then wrap in pandas to pretty print
    C = metrics.multilabel_confusion_matrix(y_true, y_pred, labels)
    # df = pd.DataFrame(C, columns=target_names, index=target_names)
    accuracy = []
    for i in range(len(y_pred)):
        accuracy.append(metrics.accuracy_score(y_true[i,:],y_pred[i,:]))
    a = np.average(accuracy)
    #TODO: Check if I have to do this way for other metrics
    
    print("Confusion Matrix\n", C)


    #a = metrics.accuracy_score(y_true, y_pred)
    p = metrics.precision_score(y_true, y_pred, average='macro')
    r = metrics.recall_score(y_true, y_pred, average='macro')
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    m = metrics.hamming_loss(y_true, y_pred)
    print("Macro accuracy: ", a)
    print("precision: ",p)
    print("recall: ",r)
    print("f1: ", f1)
    print("hamming loss:", m)


    # Exact Match Ratio

    # print report (redundant with above but easier)
    # report = metrics.classification_report(y_true, y_pred, labels, target_names)
    # print(report)

    return C, a, p, r, f1, m


def run_one_featureset(
    classifier,
    X_train,
    y_train,
    X_test,
    y_test,
    n,
    d,
    scoring='f1_macro',
    iterations=N_ITER,
    n_jobs=4,
    n_splits=10,
    ):
    #include feature path later as input
    #Also label_path=DI_LABELS_CSV, labels = [0, 1], target_names = ['Low', 'High'],
    """Run ML pipeline.
    Parameters
    --------------------
        dataset           -- dataset, string
        classifier        -- classifier, string
        X_train           -- Feature training set, (train_samples, n_features)
        y_train           -- Target training set, (train_samples, n_labels)
        X_test            -- Feature test set, (test_samples, n_features)
        y_test            -- Target set set, (test_samples, n_labels)
        n,d               -- Shape of X_train
        scoring           -- Scoring parameter
        iterations        -- number of iterations
        n_jobs            -- number of jobs
        n_splits         -- number of splits
    """
        
    log = open(classifier+'.txt', 'w' , encoding='utf-8')
    # make pipeline
    pipe, param_grid = make_pipeline(classifier, n, d)

    # get param grid size
    sz = 1
    try:
        for vals in param_grid.values():
            sz *= len(vals)
            # tune model using randomized search
        n_iter = min(iterations, sz)    # cap max number of iterations
    except TypeError:
        n_iter = iterations

    #Randomized Search CV
    search = RandomizedSearchCV(
        estimator = pipe,
        param_distributions = param_grid,
        verbose=2,
        cv = 10,
        n_iter = n_iter,
        refit=scoring,
        scoring='f1_macro',
        return_train_score=True,
        n_jobs=n_jobs)


    #Best parameter set
    search.fit(X_train, y_train)
    print("Best parameters set found on development set:\n")
    print(search.best_params_)
    print("\n")
    log.write(classifier+" Report \n")
    log.write("Best Parameter" +str(search.best_params_)+"\n")

    # Report of training result
    print("Detailed classification report (training set):\n")
    y_true, y_pred = y_train, search.predict(X_train)
    C, a, p, r, f1, m = report_metrics(y_true, y_pred)
    log.write("Training Set Result) \n")
    log.write("Confusion Matrix: "+str(C)+"\n")
    log.write("Accuracy: "+str(a)+"\n")
    log.write("Macro Precision: "+str(p)+"\n")
    log.write("Macro Recall: "+str(r)+"\n")
    log.write("Macro F1: "+str(f1)+"\n")
    log.write("Hamming Loss: "+str(m)+"\n")
    
    print("\n")

    print("Draw ROC Curve")
    #Draw ROC curve, Only possible when there are few classes, or we can do top 5 classes
    y_score = search.predict_proba(X_train)
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    #check
    n_feature, n_labels = y_train.shape
    print(n_labels)
    for i in range(n_labels):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    
    highkey = roc_auc = sorted(roc_auc, key=roc_auc.get, reverse=True)[:3]

    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(highkey, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for'+classifier)
    plt.legend(loc="lower right")
    name = "sample_"+classifier+"_png"
    plt.savefig(name)
    plt.close()

    #TODO: Implement Learning Curve Later

    #Report final testing result
    print("\n")
    print("Test set Classification Result:\n")
    y_testtrue, y_testpred = y_test, search.predict(X_test)
    Ct, at, pt, rt, f1t, mt = report_metrics(y_testtrue, y_testpred)

    log.write("Test Set Result) \n")
    log.write("Confusion Matrix: "+str(Ct)+"\n")
    log.write("Accuracy: "+str(at)+"\n")
    log.write("Macro Precision: "+str(pt)+"\n")
    log.write("Macro Recall: "+str(rt)+"\n")
    log.write("Macro F1: "+str(f1t)+"\n")
    log.write("Hamming Loss: "+str(mt)+"\n")
    
    log.close()

    name = classifier+'_model.joblib'
    joblib.dump(search, name)
    #TODO: Create bar graph for each metrics


# ######################################################################
# # main
# ######################################################################

#TODO: Implement when ready
def main():
    # set random seed (for repeatability)
    np.random.seed(42)
    X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=4, n_labels=2, random_state=1)
    #TODO: Make X as binary
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size = 0.2)
    n,d = X_train.shape

    for clf in classifiers.CLASSIFIERS:
        print('training ', clf)
        if clf == 'MLP' or clf == 'SVM' or clf == 'RF':
            iterations = 50
        elif clf == 'XGBoost':
            iterations = 20
        else:
            iterations = N_ITER
        n_splits = 10
        run_one_featureset(
            classifier=clf,
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_test, 
            y_test=y_test, 
            n=n, 
            d=d,
            scoring='f1_macro',
            iterations = iterations,
            n_jobs=-1,
            n_splits=n_splits
            )

if __name__ == "__main__":
    main()