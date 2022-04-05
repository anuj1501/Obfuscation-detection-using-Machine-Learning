import os
import pickle
import logging
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from entropy_handling import entropy

import tokens


CURRENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DICO_PATH = os.path.join(CURRENT_PATH, 'ngrams2int')

def n_grams_list(numbers_list, n):
    
    if numbers_list is not None:
        len_numbers_list = len(numbers_list)
        if n < 1 or n > len_numbers_list:
            logging.warning('The file has less tokens than the length n of an n-gram')

        else:
            range_n = range(n)
            matrix_all_n_grams = []
            range_list = range(len_numbers_list - (n - 1))
            for j in range_list:  # Loop on all the n-grams
                matrix_all_n_grams.append(tuple(numbers_list[j + i] for i in range_n))
            return matrix_all_n_grams
    return None

def count_sets_of_n_grams(input_file, tolerance, n):
    
    numbers_list = tokens.tokens_to_numbers(input_file, tolerance)
    matrix_all_n_grams = n_grams_list(numbers_list, n)

    if matrix_all_n_grams is not None:
        dico_of_n_grams = {}

        for j, _ in enumerate(matrix_all_n_grams):
            if matrix_all_n_grams[j] in dico_of_n_grams:
                dico_of_n_grams[matrix_all_n_grams[j]] += 1
            else:
                dico_of_n_grams[matrix_all_n_grams[j]] = 1

        return [dico_of_n_grams, len(matrix_all_n_grams)]
    return [None, None]

def import_modules(n):
    
    global global_ngram_dict
    global_ngram_dict = pickle.load(open(os.path.join(DICO_PATH, str(n) + '-gram',
                                                      'ast_esprima_simpl'), 'rb'))


def nb_features(n):
    
    ns_features = [19, 361, 1000, 4000, 15000, 40000, 100000]
    if n < 8:
        n_features = ns_features[n - 1]
    else:
        n_features = 200000
    return n_features


def vect_proba_of_n_grams(input_file, tolerance, n, dico_ngram_int):
    
    dico_of_n_grams, nb_n_grams = count_sets_of_n_grams(input_file, tolerance, n)
    if dico_of_n_grams is not None:
        n_features = nb_features(n)
        vect_n_grams_proba = np.zeros(n_features)

        for key, proba in dico_of_n_grams.items():
            map_ngram_int = n_gram_to_int(dico_ngram_int, key, n_features)
            if map_ngram_int is not None:
                vect_n_grams_proba[map_ngram_int] = proba / nb_n_grams

        return vect_n_grams_proba, dico_of_n_grams
    return None,None


def n_gram_to_int(dico_ngram_int, n_gram, n_features):
    
    try:
        i = dico_ngram_int[str(n_gram)]
    except KeyError:  # Key not in dico, we add it. Beware dico referenced as global variable.
        dico_ngram_int[str(n_gram)] = len(dico_ngram_int)
        i = dico_ngram_int[str(n_gram)]
    if i < n_features:
        return i
    else:
        logging.warning('The vector space size of ' + str(n_features) + ' is too small.'
                        + ' Tried to access element ' + str(i)
                        + '. This can be changed in ngrams_handling.nb_features(n)')
        return None


def int_to_n_gram(dico_ngram_int, i):
    

    try:
        ngram = dico_ngram_int[str(i)]
        return ngram
    except KeyError as err:
        logging.warning('The key ' + str(err) + ' is not in the n-gram - int mapping dictionary')


def final_feature_vector(input_file, tolerance, n, dico_ngram_int):

    n_grams_vector,dico_of_n_grams = vect_proba_of_n_grams(input_file, tolerance, n, dico_ngram_int)

    if n_grams_vector is not None:

        entropy_value, length_of_javascript = entropy(input_file)

        entropy_and_n_grams_vector = np.append(n_grams_vector, [entropy_value, length_of_javascript])

        return entropy_and_n_grams_vector

    return None
