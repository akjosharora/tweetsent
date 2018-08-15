'''
    ML logic.
'''

import os
import pandas as pd

from .clean     import clean_and_stem
from itertools  import chain

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

def get_words_count(df, column= 'tokens', minimum= 4):
    '''
        Gets the number fof occurences for all collected tokens.
        Args    :
            df      : Pandas data frame.
            tokens  : Tokens column name, default is tokens.
            minimum : Tokens with les than "minimum" occurences will not be considered.
        Returns :
            A list of lists [token, number of occurences] ordered by number of occurences.
    '''
    #Flastten the tokens lists into one list
    all_tokens      = list(chain(*df[column].values))

    #Count the occurences of tokens
    occurences      = {}
    for x in all_tokens:
        if not occurences.get(x) :
            count           = all_tokens.count(x)
            #We should add the key even if count < minimum to avoid doing the same operation multiple times.
            occurences[x]   = count

    return list(reversed(
        sorted(
            [[x, occurences[x]] for x in occurences if occurences[x] >=minimum] ,
            key= lambda x: x[1]
            )))

def get_repr_df(df, words_count, tokens_column= 'tokens', category_column= 'Category'):
    '''
        Create a numerical represntation for df using the words counter.
        Args    :
            df              : The tweet or text df.
            words_count     : Words or tokens counts.
            tokens_column   : Column name for tokens.
            category_column : Column name for category.
        Returns : A DataFrame contining tweet ids, category and numerical values. for words.
    '''
    df_repr             = pd.DataFrame()
    df_repr['category'] = df[category_column]

    for word in words_count:
        df_repr[word[0]] = df[tokens_column].apply(lambda x: x.count(word[0]))

    return df_repr

def get_numerical_repr(data_set_path, column, minimum_occ= 4, use_saved= True, save= True):
    '''
        Creates the numerical represntation from a data set of tweets or text.
        Args    :
            data_set_path   : The path to the data set.
            column          : The cloumn containing the text.
            use_saved       : If True, attemp to use a saved copy(file name + _save).
        Returns : The numeric representation.
    '''
    save_path           = get_added_name(data_set_path,'_repr')
    if use_saved :
        return pd.DataFrame.from_csv(save_path)

    #Load the data set
    df                  = pd.DataFrame.from_csv(data_set_path)

    #Clean, tokenize and stem the text
    df['tokens']        = df[column].apply(clean_and_stem)

    #Remove empty tweets
    df                  = df[~df.apply(lambda x: x['tokens'] == [], axis=1)]

    #Get occurences
    occs                = get_words_count(df, 'tokens', minimum_occ)

    #Create the represntation df
    repr_df             = get_repr_df(df, occs)

    #Save for fast loading
    if save :
        with open(save_path, 'w') as f:
            repr_df.to_csv(f)

    return repr_df
