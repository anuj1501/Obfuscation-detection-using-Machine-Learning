
import os
import pickle
import argparse
import logging

import utility
import static_analysis


src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def validate(labels_validation, attributes_validation, model, model_name, model_dir, add_trees=100):
    
    if isinstance(model, str):
        model = pickle.load(open(model, 'rb'))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.set_params(warm_start=True)  # RF
    model.n_estimators += add_trees  # RF
    validated = model.fit(attributes_validation, labels_validation)  # RF

    model_path = os.path.join(model_dir, model_name)
    pickle.dump(validated, open(model_path, 'wb'))
    logging.info('The model has been successfully updated in ' + model_path)

    return validated


def parsing_commands():
    

    parser = argparse.ArgumentParser(description='Given a list of directory or file paths, '
                                                 + 'updates a model to classify future '
                                                 + 'JS inputs.')

    parser.add_argument('--d', metavar='DIR', type=str, nargs='+',
                        help='directories to be used to update a model with')
    parser.add_argument('--l', metavar='LABEL', type=str, nargs='+',
                        choices=['benign', 'malicious'],
                        help='labels of the JS directories used to update a model with')
    parser.add_argument('--f', metavar='FILE', type=str, nargs='+',
                        help='files to be used to update a model with')
    parser.add_argument('--lf', metavar='LABEL', type=str, nargs='+',
                        choices=['benign', 'malicious'],
                        help='labels of the JS files used to update a model with')
    parser.add_argument('--m', metavar='OLD-MODEL', type=str, nargs=1,
                        help='path of the old model you wish to update with new JS inputs')
    parser.add_argument('--md', metavar='MODEL-DIR', type=str, nargs=1,
                        default=[os.path.join(src_path, 'Classification')],
                        help='path to store the model that will be produced')
    parser.add_argument('--mn', metavar='MODEL-NAME', type=str, nargs=1,
                        default=['model'],
                        help='name of the model that will be produced')
    parser.add_argument('--at', metavar='NB_TREES', type=int, nargs=1,
                        default=[100], help='number of trees to be added into the forest')
    utility.parsing_commands(parser)

    return vars(parser.parse_args())


arg_obj = parsing_commands()
utility.control_logger(arg_obj['v'][0])


def main_update(js_dirs=arg_obj['d'], js_files=arg_obj['f'], labels_f=arg_obj['lf'],
                labels_d=arg_obj['l'], old_model=arg_obj['m'], model_dir=arg_obj['md'],
                model_name=arg_obj['mn'], n=arg_obj['n'][0], tolerance=arg_obj['t'][0],
                add_trees=arg_obj['at'], dict_not_hash=arg_obj['dnh'][0]):
    

    if js_dirs is None and js_files is None:
        logging.error('Please, indicate a directory or a JS file to be used to update '
                      + 'the old model with')

    elif labels_d is None and labels_f is None:
        logging.error('Please, indicate the labels (either benign or malicious) of the files '
                      + 'used to update the model')

    elif js_dirs is not None and (labels_d is None or len(js_dirs) != len(labels_d)):
        logging.error('Please, indicate as many directory labels as the number '
                      + str(len(js_dirs)) + ' of directories to analyze')

    elif js_files is not None and (labels_f is None or len(js_files) != len(labels_f)):
        logging.error('Please, indicate as many file labels as the number '
                      + str(len(js_files)) + ' of files to analyze')

    elif old_model is None:
        logging.error('Please, indicate the path of the old model you would like to update.\n'
                      + '(see >$ python3 <path-of-clustering/learner.py> -help)'
                      + ' to build a model)')

    else:
        names, attributes, labels = static_analysis.main_analysis\
            (js_dirs=js_dirs, labels_dirs=labels_d, js_files=js_files, labels_files=labels_f,
             tolerance=tolerance, n=n, dict_not_hash=dict_not_hash)

        if names:

            validate(labels, attributes, model=old_model[0], add_trees=add_trees[0],
                     model_name=model_name[0], model_dir=model_dir[0])

        else:
            logging.warning('No file found for the analysis.\n'
                            + '(see >$ python3 <path-of-js/is_js.py> -help)'
                            + ' to check your files correctness.\n'
                            + 'Otherwise they may not contain enough n-grams)')


if __name__ == "__main__":  
    main_update()
