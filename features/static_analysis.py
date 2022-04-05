import os
import sys
import pickle
import logging

import ngrams_handling


CURRENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DICO_PATH = os.path.join(CURRENT_PATH, 'ngrams2int')


def main_analysis(js_dirs, js_files, labels_files, labels_dirs, n, tolerance, dict_not_hash):
    
    if js_dirs is None and js_files is None:
        logging.error('Please, indicate a directory or a JS file to be studied')

    else:
        if dict_not_hash:
            ngrams_handling.import_modules(n)

        if js_files is not None:
            files2do = js_files
            if labels_files is None:
                labels_files = ['?' for _, _ in enumerate(js_files)]
            labels = labels_files
        else:
            files2do, labels = [], []
        if js_dirs is not None:
            i = 0
            if labels_dirs is None:
                labels_dirs = ['?' for _, _ in enumerate(js_dirs)]
            for cdir in js_dirs:
                for cfile in os.listdir(cdir):
                    files2do.append(os.path.join(cdir, cfile))
                    if labels_dirs is not None:
                        labels.append(labels_dirs[i])
                i += 1

        tab_res = [[], [], []]

        if not dict_not_hash:
            csr_res = None
            n_features = ngrams_handling.nb_features(n)

        for j, _ in enumerate(files2do):
            if dict_not_hash:
                res = ngrams_handling.final_feature_vector(files2do[j], tolerance, n,
                                                            ngrams_handling.global_ngram_dict)
            else:  # hashes
                res = ngrams_handling.csr_proba_of_n_grams_hash_storage(files2do[j], tolerance,
                                                                        n, n_features)
            if res is not None:
                tab_res[0].append(files2do[j])
                if dict_not_hash:
                    tab_res[1].append(res)
                else:  # hashes
                    csr_res = ngrams_handling.concatenate_csr_matrices(csr_res, res, n_features)
                if labels and labels != []:
                    tab_res[2].append(labels[j])
        if dict_not_hash:
            sys.path.insert(0, os.path.join(DICO_PATH, str(n) + '-gram'))
            pickle.dump(ngrams_handling.global_ngram_dict,
                        open(os.path.join(DICO_PATH, str(n) + '-gram', 'ast_esprima_simpl'), 'wb'))
        else:
            tab_res[1].append(csr_res)
            tab_res[1] = tab_res[1][0]

        return tab_res
