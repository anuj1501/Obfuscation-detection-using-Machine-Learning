import subprocess  
import os 
import argparse  
import logging

OUTPUT_TEXTS = ['valid JavaScript', 'not JavaScript', 'malformed JavaScript']
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def is_js_file(given_file, syntactical_units=False, tolerance='false'):
    

    with open(os.path.join(SRC_PATH, 'is_js.log'), 'w') as my_log:
        try:
            result = subprocess.check_output('node '
                                             + os.path.join(SRC_PATH, 'features',
                                                            'parsing', 'parser.js')
                                             + ' ' + given_file
                                             + ' ' + tolerance, stderr=my_log, shell=True)

            if syntactical_units:
                syntax_part = str(result).split("b'")[1].split('\\n')  
                del syntax_part[len(syntax_part) - 1] 
                return syntax_part 
            return 0

        except subprocess.CalledProcessError as err:
            if err.returncode == 1 or err.returncode == 8:
                if str(err.output) == "b''":  
                    return 1
                return 2  
            elif err.returncode != 0:
                
                logging.exception("Something went wrong with the file <" + given_file + ">: "
                                  + str(err))

        except OSError:  
            logging.exception("System-related error")
            return -1


def main():
    
    parser = argparse.ArgumentParser(description='Given a list of directory, or of file paths,\
    indicates whether the files are either\n\
    valid (\'<fileName>: valid JavaScript\'),\n\
    malformed (\'<fileName>: malformed JavaScript\'),\n\
    or no JavaScript (\'<fileName>: not JavaScript\').')
    
    parser.add_argument('--f', metavar='FILE', nargs='+', help='files to be tested')
    parser.add_argument('--d', metavar='DIR', nargs='+', help='directories to be tested')
    parser.add_argument('--v', metavar='VERBOSITY', type=int, nargs=1, choices=[0, 1, 2, 3, 4, 5],
                        default=[2], help='controls the verbosity of the output, from 0 (verbose) '
                                          + 'to 5 (less verbose)')

    args = vars(parser.parse_args())
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.getLevelName(args['v'][0] * 10))

    if args['f'] is not None:
        files2do = args['f']
    else:
        files2do = []
    if args['d'] is not None:
        for cdir in args['d']:
            files2do.extend(os.path.join(cdir, cfile) for cfile in os.listdir(cdir))
    results = [is_js_file(cfile) for cfile in files2do]
    for cfile, res in zip(files2do, results):
        print("%s: %s" % (cfile, OUTPUT_TEXTS[res]))
    logging.info('\tNumber of correct files: %d', len([i for i in results if i == 0]))


if __name__ == "__main__":  # Executed only if run as a script
    main()
