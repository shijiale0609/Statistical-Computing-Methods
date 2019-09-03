import sys
import log
import numpy as np

def readFileData(fileName):
    '''
    Reads in catapillar data from files
    @params:
        fileName(string): File name
    Returns:
        data (np.array): array of data read from file
    '''
    log.info("Reading in data file")
    #Attempt to read text file and extact data into a list
    try:
        file_object  = open(str(fileName), "r").read().splitlines()
        data_list = [[float(x.strip()) for x in my_string.split()] for my_string in file_object]
    except OSError as err:
        log.error("OS error: {0}".format(err))
        return
    except IOError as err:
        log.error("File read error: {0}".format(err))
        return
    except:
        log.error("Unexpected error:"+sys.exc_info()[0])
        return

    data = np.asarray(data_list)
    log.sucess("Catapillar data successfully read from file")
    return data

def npArrayToStrList( arr, str_format='{0:.3f}'):
    '''
    Converts 2D numpy array or list to a list of strings
    Primarily used for outputing tables
    @params:
        arr (np.array or list): 2D list of numbers
        str_format (string): String format string
    Returns:
        str_arr (list): 2D array of strings
    '''
    str_arr = [[str_format.format(float(y)) for y in x] for x in arr]
    return str_arr

def appendListColumn( list0, col, index):
    '''
    Adds on a column to a list at the specified index
    '''
    list_new = []
    for i, row in enumerate(list0):
        row0 = list(row)
        row0.insert(index,col[i])
        list_new.append(row0)
    return list_new

def getGammaIndexes(k, i):
    '''
    Gets the list of model column indexes for the given index
    Used in variable selection
    '''
    frm_str = '{0:0'+str(k)+'b}'
    gamma_b = np.array([int(j) for j in frm_str.format(i)[::-1]]) #Convert index to binary
    gamma_i = np.where(gamma_b == 1)[0]+1
    gamma_i = np.insert(gamma_i,0,0) #Add index for bias term
    q = np.sum(gamma_b)

    return gamma_i, q
