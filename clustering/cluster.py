import os
import argparse  # To parse command line arguments
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import utility
import static_analysis


src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def nb_clusters(attributes, fig_dir=os.path.join(src_path, 'Clustering'),
                fig_name='NumberOfClusters.png', min_a=1, max_a=5):
    
    try:
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)


        distorsions = []
        for i in range(min_a, max_a+1):
            kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
            
            kmeans.fit(attributes)  
            distorsions.append(kmeans.inertia_)  

        plt.plot(range(min_a, max_a+1), distorsions, marker='x')
        plt.grid()
        plt.xlabel('Number of clusters')
        plt.ylabel('Total squared distance inside clusters')
        plt.savefig(os.path.join(fig_dir, fig_name), dpi=100)
        plt.clf()  # Otherwise all figures are written on one another

    except ValueError as error:
        logging.exception('Unable to produce more clusters than there is data available: '
                          + str(error))


def clustering(names, attributes, nb_cluster, fig_dir=os.path.join(src_path, 'Clustering'),
               fig_name='ClusteringPca.png', true_labels=None, display_fig=False, annotate=False,
               title='Projection of the n-grams frequency of JavaScript files'):
    
    try:
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        km = KMeans(n_clusters=nb_cluster, init='k-means++', n_init=10, max_iter=300,
                    tol=1e-04, random_state=0)
        

        labels_predicted = km.fit_predict(attributes)  

        utility.get_classification_results(names, labels_predicted)

        if display_fig:
            pca = PCA(n_components=2)  
            attributes = pd.DataFrame(pca.fit_transform(attributes))
            attributes = np.asarray(attributes)

            if true_labels is not None and true_labels and '?' not in true_labels:
                labels_predicted = true_labels
                labels_predicted = np.asarray(labels_predicted)

            colors = ['orange', 'lightblue', 'red', 'lightgreen', 'lightpink', 'darkgoldenrod',
                      'deepskyblue', 'seagreen', 'darkslateblue', 'gainsboro', 'khaki', 'slategray',
                      'darkcyan', 'darkslategrey', 'lawngreen', 'deeppink', 'thistle', 'sandybrown',
                      'mediumorchid', 'orangered', 'paleturquoise', 'coral', 'navy', 'slateblue',
                      'rebeccapurple', 'darkslategray', 'limegreen', 'magenta', 'skyblue',
                      'forestgreen',
                      'blue', 'lavender', 'mediumslateblue', 'aqua', 'mediumvioletred',
                      'lightsteelblue',
                      'cyan', 'mistyrose', 'darkorchid', 'gold', 'chartreuse', 'bisque', 'olive',
                      'darkmagenta', 'darkviolet', 'lightgrey', 'mediumblue', 'indigo',
                      'papayawhip',
                      'powderblue', 'aquamarine', 'wheat', 'hotpink', 'mediumseagreen', 'royalblue',
                      'pink',
                      'mediumaquamarine', 'goldenrod', 'peachpuff', 'darkkhaki', 'silver',
                      'mediumspringgreen', 'yellowgreen', 'cadetblue', 'olivedrab', 'darkgray',
                      'chocolate',
                      'palegoldenrod', 'darkred', 'peru', 'fuchsia', 'darkturquoise', 'cornsilk',
                      'lightgoldenrodyellow', 'lightslategray', 'dimgray', 'white', 'sienna',
                      'orchid',
                      'darkorange', 'darkseagreen', 'steelblue', 'darkgreen', 'violet', 'slategrey',
                      'lightsalmon', 'palegreen', 'yellow', 'lemonchiffon', 'antiquewhite', 'green',
                      'lightslategrey', 'tan', 'honeydew', 'whitesmoke', 'blueviolet',
                      'navajowhite',
                      'darkblue', 'mediumturquoise', 'dodgerblue', 'lightskyblue', 'crimson',
                      'snow',
                      'brown', 'indianred', 'palevioletred', 'plum', 'linen', 'cornflowerblue',
                      'saddlebrown', 'springgreen', 'lightseagreen', 'greenyellow', 'ghostwhite',
                      'rosybrown', 'darkgrey', 'grey', 'lime', 'teal', 'gray', 'mediumpurple',
                      'darkolivegreen', 'burlywood', 'tomato', 'lightcoral', 'purple', 'salmon',
                      'darksalmon', 'dimgrey', 'moccasin', 'maroon', 'ivory', 'turquoise',
                      'firebrick']
            markers = ['s', 'v', 'o', 'd', 'p', '^', '<', '>', '1', '2', '3', '4', '8', 'h', '.',
                       'H',
                       '+',
                       'x', 'D', '|', '_', 's', 'v', 'o', 'd', 'p', '^', '<', '>', '1', '2', '3',
                       '4',
                       '8',
                       'h', '.', 'H', '+', 'x', 'D', '|', '_']

            i = 0
            unique_label = []
            for label in labels_predicted:
                if label not in unique_label:
                    unique_label.append(label)  

            for label in unique_label:
                plt.scatter(attributes[labels_predicted == label, 0],
                            attributes[labels_predicted == label, 1],
                            c=colors[i], marker=markers[i], label='Cluster ' + str(label))
                i += 1

            if annotate:
                for i in range(len(names)):
                    plt.annotate(str(i+1), (attributes[i][0], attributes[i][1]))

            plt.legend()
            plt.grid()
            plt.title(title)
            fig_path = os.path.join(fig_dir, fig_name)
            plt.savefig(fig_path, dpi=100)
            plt.clf()  # Otherwise all figures are written on one another
            logging.info('The graphical representation of the clusters has been successfully '
                         + 'stored in ' + fig_path)

    except ValueError as error:
        logging.exception('Unable to produce more clusters than there is data available: '
                          + str(error))


def parsing_commands_clustering():
    
    parser = argparse.ArgumentParser(description='Given a list of repository or file paths,\
    clusters the JS inputs into several families.')

    parser.add_argument('--d', metavar='DIR', type=str, nargs='+',
                        help='directories containing the JS files to be clustered')
    parser.add_argument('--f', metavar='FILE', type=str, nargs='+', help='files to be analyzed')
    parser.add_argument('--c', metavar='INTEGER', type=int, nargs=1, help='number of clusters')
    parser.add_argument('--g', metavar='BOOL', type=bool, nargs=1, default=[False],
                        help='produces a 2D representation of the files from the JS corpus')
    parser.add_argument('--l', metavar='LABEL', type=str, nargs='+', default=None,
                        help='true labels of the JS directories (used only for display)')
    parser.add_argument('--lf', metavar='LABEL', type=str, nargs='+', default=None,
                        help='true labels of the JS files (used only for display)')
    utility.parsing_commands(parser)

    return vars(parser.parse_args())


arg_obj = parsing_commands_clustering()
utility.control_logger(arg_obj['v'][0])


def main_clustering(js_dirs=arg_obj['d'], js_files=arg_obj['f'], tolerance=arg_obj['t'][0],
                    nb_cluster=arg_obj['c'], n=arg_obj['n'][0], display_fig=arg_obj['g'][0],
                    dict_not_hash=arg_obj['dnh'][0], labels_d=arg_obj['l'], labels_f=arg_obj['lf']):
    
    if js_dirs is None and js_files is None:
        logging.error('Please, indicate a directory or a JS file to be analysed')

    elif nb_cluster is None:
        logging.error('Please, indicate a number of clusters')

    elif js_dirs is not None and labels_d is not None and len(js_dirs) != len(labels_d):
        logging.error('Please, indicate as many directory labels as the number '
                      + str(len(js_dirs)) + ' of directories to analyze')

    elif js_files is not None and labels_f is not None and len(js_files) != len(labels_f):
        logging.error('Please, indicate as many file labels as the number '
                      + str(len(js_files)) + ' of files to analyze')

    else:
        names, attributes, labels = static_analysis.main_analysis \
            (js_dirs=js_dirs, labels_dirs=labels_d, js_files=js_files, labels_files=labels_f,
             tolerance=tolerance, n=n, dict_not_hash=dict_not_hash)

        if names:

            clustering(names=names, attributes=attributes, nb_cluster=nb_cluster[0],
                       display_fig=display_fig, true_labels=labels)

        else:
            logging.warning('No file found for the analysis.\n'
                            + '(see >$ python3 <path-of-js/is_js.py> -help)'
                            + ' to check your files correctness.\n'
                            + 'Otherwise they may not contain enough n-grams)')


if __name__ == "__main__":  
    main_clustering()
