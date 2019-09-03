# -*- coding: utf-8 -*-
import sys
from time import gmtime, strftime

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log(str0):
    base_str = strftime("[%H:%M:%S]", gmtime())
    output_type = "[Output]: "
    print(base_str+output_type+str0)

def info(str0):
    base_str = strftime("[%H:%M:%S]", gmtime())
    output_type = "[Info]: "
    print(bcolors.OKBLUE+base_str+output_type+str0+bcolors.ENDC)

def sucess(str0):
    base_str = strftime("[%H:%M:%S]", gmtime())
    output_type = "[Success]: "
    print(bcolors.OKGREEN+base_str+output_type+str0+bcolors.ENDC)

def warning(str0):
    base_str = strftime("[%H:%M:%S]", gmtime())
    output_type = "[Warning]: "
    print(bcolors.WARNING+base_str+output_type+str0+bcolors.ENDC)

def error(str0):
    base_str = strftime("[%H:%M:%S]", gmtime())
    output_type = "[Error]: "
    print(bcolors.FAIL+base_str+output_type+str0+bcolors.ENDC)


# Print iterations progress
# https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progress(iteration, total, prefix=strftime("[%H:%M:%S]", gmtime()), suffix='', decimals=1, bar_length=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()