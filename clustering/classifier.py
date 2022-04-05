import os
import pickle
import argparse
import logging

import utility
import static_analysis


def test_model(names, labels, attributes, model, print_res=True, print_res_verbose=False,
               print_score=True, threshold=0.29):
    
    if isinstance(model, str):
        model = pickle.load(open(model, 'rb'))

    labels_predicted_proba_test = model.predict_proba(attributes)
    
    labels_predicted_test = utility.\
        predict_labels_using_threshold(len(names), labels_predicted_proba_test, threshold)
    
    if print_res:
        utility.get_classification_results(names, labels_predicted_test)

    if print_res_verbose:
        utility.get_classification_results_verbose(names, labels, labels_predicted_test,
                                                   labels_predicted_proba_test, model,
                                                   attributes, threshold)

    if print_score:
        utility.get_score(labels, labels_predicted_test)

    return labels_predicted_test


def parsing_commands():
    
    parser = argparse.ArgumentParser(description='Given a list of directory or file paths,\
    detects the obfuscated JS inputs.')

    parser.add_argument('--d', metavar='DIR', type=str, nargs='+',
                        help='directories containing the JS files to be analyzed')
    parser.add_argument('--l', metavar='LABEL', type=str, nargs='+',
                        choices=['non-obfuscated', 'obfuscated', '?'],
                        help='labels of the JS directories to evaluate the model from')
    parser.add_argument('--f', metavar='FILE', type=str, nargs='+', help='files to be analyzed')
    parser.add_argument('--lf', metavar='LABEL', type=str, nargs='+',
                        choices=['non-obfuscated', 'obfuscated', '?'],
                        help='labels of the JS files to evaluate the model from')
    parser.add_argument('--m', metavar='MODEL', type=str, nargs=1,
                        help='path of the model used to classify the new JS inputs '
                             + '(see >$ python3 <path-of-clustering/learner.py> -help) '
                             + 'to build a model)')
    parser.add_argument('--th', metavar='THRESHOLD', type=float, nargs=1, default=[0.29],
                        help='threshold over which all samples are considered obfuscated')
    utility.parsing_commands(parser)

    return vars(parser.parse_args())


arg_obj = parsing_commands()
utility.control_logger(arg_obj['v'][0])


def main_classification(js_dirs=arg_obj['d'], js_files=arg_obj['f'], labels_f=arg_obj['lf'],
                        labels_d=arg_obj['l'], model=arg_obj['m'], threshold=arg_obj['th'],
                        n=arg_obj['n'][0], tolerance=arg_obj['t'][0],
                        dict_not_hash=arg_obj['dnh'][0]):
    
    if js_dirs is None and js_files is None:
        logging.error('Please, indicate a directory or a JS file to be studied')

    elif js_dirs is not None and labels_d is not None and len(js_dirs) != len(labels_d):
        logging.error('Please, indicate either as many directory labels as the number '
                      + str(len(js_dirs))
                      + ' of directories to analyze or no directory label at all')

    elif js_files is not None and labels_f is not None and len(js_files) != len(labels_f):
        logging.error('Please, indicate either as many file labels as the number '
                      + str(len(js_files))
                      + ' of files to analyze or no file label at all')

    elif model is None:
        logging.error('Please, indicate a model to be used to classify new files.\n'
                      + '(see >$ python3 <path-of-clustering/learner.py> -help)'
                      + ' to build a model)')

    else:
        print(labels_d)
        names, attributes, labels = static_analysis.main_analysis \
            (js_dirs=js_dirs, labels_dirs=labels_d, js_files=js_files, labels_files=labels_f,
             tolerance=tolerance, n=n, dict_not_hash=dict_not_hash)

        if names:

            test_model(names, labels, attributes, model=model[0], threshold=threshold[0])

        else:
            logging.warning('No file found for the analysis.\n'
                            + '(see >$ python3 <path-of-js/is_js.py> -help)'
                            + ' to check your files correctness.\n'
                            + 'Otherwise they may not contain enough n-grams)')


if __name__ == "__main__":  # Executed only if run as a script
    main_classification()


def classify_analysis_results(save_dir, model, threshold):
    
    names = pickle.load(open(os.path.join(save_dir, 'Names'), 'rb'))
    attributes = pickle.load(open(os.path.join(save_dir, 'Attributes'), 'rb'))
    labels = pickle.load(open(os.path.join(save_dir, 'Labels'), 'rb'))

    test_model(names, labels, attributes, model=model, threshold=threshold)
