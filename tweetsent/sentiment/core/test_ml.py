'''
    Unit tests for ml.py
'''

from .ml   import *
import os

def test_get_added_name     ():
    '''
        Tests ml.get_added_name
    '''
    paths       = [
        os.path.join('a', 'b', 'c', 'd.ext'),
        os.path.join('a', 'b', 'c')]

    expected    = [
        os.path.join('a', 'b', 'c', 'd_save.ext'),
        os.path.join('a', 'b', 'c_save')]

    results     = [get_added_name(x) for x in paths]

    assert expected == results

def test_get_words_count    ():
    '''
        Tests ml.get_words_count.
    '''
    df  = pd.DataFrame(
        {
            'tokens' :  [
                ['a', 'b', 'c'] ,
                ['b', 'b', 'c'] ,
                ['a', 'a', 'a'] ]})

    assert      get_words_count(df, 'tokens', 0) == \
                [['a', 4], ['b', 3], ['c', 2]]

def testget_repr_df         ():
    '''
        Tests ml.get_repr_df
    '''
    df      = pd.DataFrame(
        {
            'tokens'    :  [
                ['a', 'b', 'c'] ,
                ['b', 'b', 'c'] ,
                ['a', 'a', 'a'] ],
            'Category'  : ['X', 'X', 'X']})
    occs    = get_words_count(df, 'tokens', 0)
    df_repr = get_repr_df(df, occs)

    assert  list(df_repr['a'].values) == [1, 0, 3] and \
            list(df_repr['b'].values) == [1, 2, 0] and \
            list(df_repr['c'].values) == [1, 1, 0]

def test_get_numerical_repr ():
    '''
        Tests ml.get_numerical_repr
    '''
    data_set_path   = os.path.join(
        os.path.dirname(__file__)   ,
        'test_data'                 ,
        'ds.csv'                    )

    assert len(get_numerical_repr(data_set_path, '', 0, True)) == 2

    df_repr = get_numerical_repr(data_set_path, 'Tweets', 0, False)
    
    assert  list(df_repr['product'].values) == [1, 1] and \
            list(df_repr['good'].values)    == [1, 0]
