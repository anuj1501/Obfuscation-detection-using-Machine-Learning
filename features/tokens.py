import sys
import os

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(SRC_PATH, 'features', 'tokens2int'))
sys.path.insert(0, os.path.join(SRC_PATH, 'js'))

import parser_esprima_simpl
import is_js

DICO_TOKENS_INT = parser_esprima_simpl.ast_units_dico


def ast_used_esprima(input_file, tolerance):
    
    units = is_js.is_js_file(input_file, syntactical_units=True, tolerance=tolerance)
    if isinstance(units, list):  # otherwise an error code could be returned
        # instead of a list of syntactic units
        return units
    return None


def tokens_to_numbers(input_file, tolerance):
    
    tokens_list = ast_used_esprima(input_file, tolerance)  # List of syntactic units

    if tokens_list is not None and tokens_list != []:
        return list(map(lambda x: DICO_TOKENS_INT[x], tokens_list))
    return None
