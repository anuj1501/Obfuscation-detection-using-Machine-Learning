import logging
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve

import utility
import static_analysis


def get_optimal_threshold(fpr, tpr, thresholds):
    
    youden_j = -1  
    pos_optimal_threshold = -1

    for i in range(0, len(fpr)):
        if (- fpr[i] + tpr[i]) > youden_j:
            youden_j = - fpr[i] + tpr[i]
            pos_optimal_threshold = i

    optimal_threshold = thresholds[pos_optimal_threshold]
    logging.info('Optimal threshold: ' + str(optimal_threshold))

    return optimal_threshold


def random_grid_search(js_dirs, labels_d, n=4, tolerance='false', dict_not_hash=True):
    
    names, train_features, train_labels = static_analysis.main_analysis \
        (js_dirs=js_dirs, labels_dirs=labels_d, js_files=None, labels_files=None,
         tolerance=tolerance, n=n, dict_not_hash=dict_not_hash)

    
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]

    
    max_features = ['auto', 'log2']

    max_depth = [int(x) for x in np.linspace(start=10, stop=120, num=12)]
    max_depth.append(None)

    min_samples_split = [2, 5, 10, 20, 30, 40, 50]

    min_samples_leaf = [1, 5, 10, 20, 30, 40, 50]

    oob_score = [True, False]

    criterion = ['gini', 'entropy']

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   }

    clf_rf = utility.classifier_choice(estimators=500)
    clf_rf_random = RandomizedSearchCV(estimator=clf_rf, param_distributions=random_grid,
                                       n_iter=360, cv=5, verbose=2, random_state=0, n_jobs=-1)

    clf_rf_random.fit(train_features[0], train_labels)

    logging.debug('##############################')
    logging.debug('Best parameters:\n')
    logging.debug(clf_rf_random.best_estimator_)

    return clf_rf_random


def grid_search(js_dirs, labels_d, n=4, tolerance='false', dict_not_hash=True):
    
    names, train_features, train_labels = static_analysis.main_analysis \
        (js_dirs=js_dirs, labels_dirs=labels_d, js_files=None, labels_files=None,
         tolerance=tolerance, n=n, dict_not_hash=dict_not_hash)

    n_estimators = [int(x) for x in np.linspace(start=500, stop=700, num=5)]

    max_features = [int(x) for x in np.linspace(start=200, stop=300, num=3)]

    max_depth = [90, 100, 110, None]

    criterion = ['gini', 'entropy']

    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'criterion': criterion
            }

    clf_rf = utility.classifier_choice(estimators=500)
    clf_rf_random = GridSearchCV(estimator=clf_rf, param_grid=grid, cv=5, verbose=2, n_jobs=-1)

    clf_rf_random.fit(train_features[0], train_labels)

    logging.debug('##############################')
    logging.debug('Best parameters:\n')
    logging.debug(clf_rf_random.best_estimator_)

    return clf_rf_random


def evaluate(model, test_features, test_labels):
    
    predictions = model.predict(test_features)
    predictions_proba = model.predict_proba(test_features)
    accuracy = model.score(test_features, test_labels)  # Detection rate

    tn_test, fp_test, fn_test, tp_test = confusion_matrix(test_labels, predictions_proba).ravel()

    fpr, tpr, thresholds = roc_curve(test_labels, predictions[:, 1], pos_label='malicious')
    get_optimal_threshold(fpr, tpr, thresholds)

    print("Detection: " + str(accuracy))
    print("TN: " + str(tn_test) + ", FP: " + str(fp_test) + ", TP: " + str(tp_test)
          + ", FN: " + str(fn_test))

    return accuracy


def test_param(best_random, js_dirs_train, labels_d_train, js_dirs_test, labels_d_test,
               n=4, tolerance='false', dict_not_hast=True):
    
    _, features_train, labels_train = static_analysis.main_analysis \
        (js_dirs=js_dirs_train, labels_dirs=labels_d_train, js_files=None, labels_files=None,
         tolerance=tolerance, n=n, dict_not_hash=dict_not_hast)

    _, features_test, labels_test = static_analysis.main_analysis \
        (js_dirs=js_dirs_test, labels_dirs=labels_d_test, js_files=None, labels_files=None,
         tolerance=tolerance, n=n, dict_not_hash=dict_not_hast)

    random_accuracy = evaluate(best_random, features_test[0], labels_test)

    base_model = utility.classifier_choice(estimators=500)
    base_model.fit(features_train[0], labels_train)
    base_accuracy = evaluate(base_model, features_test[0], labels_test)

    logging.info('Improve of {:0.2f}%.'.format(100 *
                                               (random_accuracy - base_accuracy) / base_accuracy))
