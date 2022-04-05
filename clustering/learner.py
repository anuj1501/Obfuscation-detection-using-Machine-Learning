
"""
    Main module to build a model to classify future JavaScript files.
"""

import os
import pickle
import argparse
import logging

import utility
import static_analysis


src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def classify(names, labels, attributes, model_dir, model_name, estimators,
             print_score=False, print_res=False):
    
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    clf = utility.classifier_choice(estimators=estimators)
    trained = clf.fit(attributes, labels) 
    labels_predicted = clf.predict(attributes)  

    if print_score:
        utility.get_score(labels, labels_predicted)

    if print_res:
        utility.get_classification_results(names, labels_predicted)

    model_path = os.path.join(model_dir, model_name)
    pickle.dump(trained, open(model_path, 'wb'))
    logging.info('The model has been successfully stored in ' + model_path)

    return trained


def parsing_commands():
    
    parser = argparse.ArgumentParser(description='Given a list of directory or file paths, '
                                                 + 'builds a model to classify future '
                                                 + 'JS inputs.')

    parser.add_argument('--d', metavar='DIR', type=str, nargs='+',
                        help='directories to be used to build a model from')
    parser.add_argument('--l', metavar='LABEL', type=str, nargs='+',
                        choices=['non-obfuscated', 'obfuscated'],
                        help='labels of the JS directories used to build a model from')
    parser.add_argument('--f', metavar='FILE', type=str, nargs='+',
                        help='files to be used to build a model from')
    parser.add_argument('--lf', metavar='LABEL', type=str, nargs='+',
                        choices=['non-obfuscated', 'obfuscated'],
                        help='labels of the JS files used to build a model from')
    parser.add_argument('--md', metavar='MODEL-DIR', type=str, nargs=1,
                        default=[os.path.join(src_path, 'Classification')],
                        help='path to store the model that will be produced')
    parser.add_argument('--mn', metavar='MODEL-NAME', type=str, nargs=1,
                        default=['model'],
                        help='name of the model that will be produced')
    parser.add_argument('--ps', metavar='BOOL', type=bool, nargs=1, default=[False],
                        help='indicates whether to print or not the classifier\'s detection rate')
    parser.add_argument('--pr', metavar='BOOL', type=bool, nargs=1, default=[False],
                        help='indicates whether to print or not the classifier\'s predictions')
    parser.add_argument('--nt', metavar='NB_TREES', type=int, nargs=1,
                        default=[500], help='number of trees in the forest')
    utility.parsing_commands(parser)

    return vars(parser.parse_args())


arg_obj = parsing_commands()
utility.control_logger(arg_obj['v'][0])


def main_learn(js_dirs=arg_obj['d'], js_files=arg_obj['f'], labels_f=arg_obj['lf'],
               labels_d=arg_obj['l'], model_dir=arg_obj['md'], model_name=arg_obj['mn'],
               print_score=arg_obj['ps'], print_res=arg_obj['pr'], dict_not_hash=arg_obj['dnh'][0],
               n=arg_obj['n'][0], tolerance=arg_obj['t'][0], estimators=arg_obj['nt']):
    

    if js_dirs is None and js_files is None:
        logging.error('Please, indicate a directory or a JS file to be used to build a model from')

    elif labels_d is None and labels_f is None:
        logging.error('Please, indicate the labels (either benign or malicious) of the files'
                      + ' used to build the model')

    elif js_dirs is not None and (labels_d is None or len(js_dirs) != len(labels_d)):
        logging.error('Please, indicate as many directory labels as the number '
                      + str(len(js_dirs)) + ' of directories to analyze')

    elif js_files is not None and (labels_f is None or len(js_files) != len(labels_f)):
        logging.error('Please, indicate as many file labels as the number '
                      + str(len(js_files)) + ' of files to analyze')

    else:
        names, attributes, labels = static_analysis.main_analysis\
            (js_dirs=js_dirs, labels_dirs=labels_d, js_files=js_files, labels_files=labels_f,
             tolerance=tolerance, n=n, dict_not_hash=dict_not_hash)

        if names:

            classify(names, labels, attributes, model_dir=model_dir[0], model_name=model_name[0],
                     print_score=print_score[0], print_res=print_res[0], estimators=estimators[0])

        else:
            logging.warning('No file found for the analysis.\n'
                            + '(see >$ python3 <path-of-js/is_js.py> -help)'
                            + ' to check your files correctness.\n'
                            + 'Otherwise they may not contain enough n-grams)')


if __name__ == "__main__": 
    main_learn()
