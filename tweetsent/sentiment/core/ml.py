'''
    ML logic.
'''

import os
import pandas as pd


def get_added_name(path, added_name= '_save'):
    '''
        Creates the save file name from the original file name.
        Args    :
            path        : Path to the original file.
            added_name  : The text added to file name to generate the new file name.
        Returns : The new file path
    '''
    basename    = os.path.basename(path)
    dirname     = os.path.dirname (path)
    file_name   = '.'.join(basename.split('.')[:-1])    if '.' in basename else basename
    extension   = ('.'+ basename.split('.')[-1])        if '.' in basename else ''
    return os.path.join(
        dirname                         ,
        file_name+ added_name+ extension)


def creat_numeric_repr(data_set_path, column, use_saved):
    '''
        Creates the numerical represntation from a data set of tweets or text.
        Args    :
            data_set_path   : The path to the data set.
            column          : The cloumn containing the text.
            use_saved       : If True, attemp to use a saved copy(file name + _save).
        Returns : The numeric representation.
    '''
    if use_saved:
        try :
            file_name   = os.path.basename(path)
            save_name   = ''.join(file_name.split('.')[:-1])+'_save'+ file_name.split('.')[-1]
            return pandas.DataFrame.from_csv(data_set_path)
        except:
            pass
