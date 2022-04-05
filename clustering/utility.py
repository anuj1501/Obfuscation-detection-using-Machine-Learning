import os
import logging
import pickle
import seaborn as sns
# import graphviz
import pandas as pd
# from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import __init__


def classifier_choice(estimators=500):


    return RandomForestClassifier(n_estimators=estimators, max_depth=50, random_state=0, n_jobs=-1)


def predict_labels_using_threshold(names_length, labels_predicted_proba, threshold):
    
    labels_predicted_test = ['non-obfuscated' for _ in range(names_length)]
    for i, _ in enumerate(labels_predicted_test):
        if labels_predicted_proba[i, 1] >= threshold:  # If the proba of the sample being obfuscated
            # is over the threshold...
            labels_predicted_test[i] = 'obfuscated'  # ... we classify the sample as obfuscated.

    return labels_predicted_test


def get_classification_results_verbose(names, labels, labels_predicted, labels_predicted_proba,
                                       model, attributes, threshold):
    counts_of_same_predictions = get_nb_trees_specific_label(model, attributes,
                                                             labels, labels_predicted, threshold)
    nb_trees = len(model.estimators_)
    for i, _ in enumerate(names):
        print(str(names[i]) + ': ' + str(labels_predicted[i]) + ' ('
              + str(labels[i]) + ') ' + 'Proba: ' + str(labels_predicted_proba[i])
              + ' Majority: ' + str(counts_of_same_predictions[i]) + '/' + str(nb_trees))
    print('> Name: labelPredicted (trueLabel) Probability[non-obfuscated, obfuscated] majorityVoteTrees')


def get_classification_results(names, labels_predicted):
    
    for i, _ in enumerate(names):
        print(str(names[i]) + ': ' + str(labels_predicted[i]))
    print('> Name: labelPredicted')


def get_score(labels, labels_predicted):
    
    if '?' in labels:
        logging.info("No ground truth given: unable to evaluate the accuracy of the "
                     + "classifier's predictions")
    else:
        try:
            tn, fp, fn, tp = confusion_matrix(labels, labels_predicted,
                                              labels=['non-obfuscated', 'obfuscated']).ravel()
            array = [[tp,fp], [fn,tn]]

            cm = pd.DataFrame(array)
            print("Detection: " + str((tp + tn) / (tp + tn + fp + fn)))
            print("TP: " + str(tp) + ", FP: " + str(fp) + ", FN: " + str(fn) + ", TN: "
                  + str(tn))

            print("accuracy: ",((tp+tn)/(tn + fp + fn + tp)))
            print("precision: ",(tp/(tn + fp)))
            print("recall: ",(tp/(tn + fn)))

            categories = ["Non-obfuscated", "Obfuscated"]
            labels = ['True Neg','False Pos','False Neg','True Pos']            
            labels = np.asarray(labels).reshape(2,2)
            confusionmatrix = sns.heatmap(cm, annot=labels,cmap='coolwarm',linecolor='white', linewidths=1, fmt ='')

            figure = confusionmatrix.get_figure()    
            figure.savefig('confusionmatrix.png', dpi=400)

        except ValueError as error_message:  # In the case of a binary classification
            # (i.e. non-obfuscated or obfuscated), if the confusion_matrix only contains one element, it
            # means that only one of the class was tested and all samples correctly classified.
            logging.exception(error_message)


def get_nb_trees_specific_label(model, attributes, labels, labels_predicted, threshold):
    
    
    counts_of_same_predictions = [0 for _, _ in enumerate(labels)]
    
    for each_tree in model.estimators_:
        single_tree_predictions_proba = each_tree.predict_proba(attributes)
        single_tree_predictions = predict_labels_using_threshold(len(labels),
                                                                 single_tree_predictions_proba,
                                                                 threshold)
        
        for j, _ in enumerate(single_tree_predictions):
            if single_tree_predictions[j] == labels_predicted[j]:
                counts_of_same_predictions[j] += 1

        

    return counts_of_same_predictions


def parsing_commands(parser):
    

    parser.add_argument('--t', metavar='TOLERANT', type=str, nargs=1, choices=['true', 'false'],
                        default=['false'], help='tolerates a few cases of syntax errors')
    parser.add_argument('--n', metavar='INTEGER', type=int, nargs=1, default=[4],
                        help='stands for the size of the sliding-window which goes through the '
                             + 'units contained in the files to be analyzed')
    parser.add_argument('--dnh', metavar='BOOL', type=str, nargs=1, default=['True'],
                        choices=['True', 'False'],
                        help='the n-grams are mapped to integers using a dictionary and not hashes')
    parser.add_argument('--v', metavar='VERBOSITY', type=int, nargs=1, choices=[0, 1, 2, 3, 4, 5],
                        default=[2], help='controls the verbosity of the output, from 0 (verbose) '
                                          + 'to 5 (less verbose)')

    return parser


def control_logger(logging_level):

    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.getLevelName(logging_level * 10))


def save_analysis_results(save_dir, names, attributes, labels):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pickle.dump(names, open(os.path.join(save_dir, 'Names'), 'wb'))
    pickle.dump(attributes, open(os.path.join(save_dir, 'Attributes'), 'wb'))
    pickle.dump(labels, open(os.path.join(save_dir, 'Labels'), 'wb'))

    logging.info('The results of the analysis have been successfully stored in ' + save_dir)
