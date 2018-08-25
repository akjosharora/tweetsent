'''
    ML logic.
'''
from __future__ import print_function

import pandas as pd
import random
import json
import os

from pprint                             import pprint
from clean                              import clean_and_stem
from itertools                          import chain

from sklearn.feature_extraction.text    import TfidfVectorizer, \
                                               CountVectorizer
from sklearn.model_selection            import train_test_split
from sklearn.neural_network             import MLPClassifier
from sklearn.neighbors                  import KNeighborsClassifier
from sklearn.svm                        import SVC
from sklearn.linear_model               import SGDClassifier
from sklearn.gaussian_process           import GaussianProcessClassifier
from sklearn.tree                       import DecisionTreeClassifier
from sklearn.ensemble                   import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes                import GaussianNB, BernoulliNB
from sklearn.metrics                    import accuracy_score




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

def predict(model, text):
    '''
        Predicts the emotion of the tweet using the trained model.
        Args    :
            model       : The trained model.
            tweet       : text to classify.
            words_occ   : words occurences
    '''
    clean       = clean_and_stem(text)
    vectorizer  = CountVectorizer(analyzer    = 'word', lowercase   = True)
    mask        = vectorizer.fit_transform([clean])

    return model.predict([mask])[0]

class Classifier():
    '''
        A custom classifier class
        Args    :
            classifiers     : Sklearn classifier classes.
            ds_path         : Data set path.
            clean_data      : Uses clean.py to clean data
            min_df          : Minimum number of a token occurences.
            data_size       : Used to extract a sub set for training, only the generated sub set will be used.
            tfidf           : Uses a tfidf vectorizer.
            text_column     : Text column.
            category_column : Category column.
            encoding        : Data set encoding.
            header          : Data set header row index.
            index_col       : The index column.
    '''
    def __init__(
        self            ,
        classifiers     ,
        ds_path         ,
        clean_data      ,
        min_df          ,
        data_size       ,
        train_size      ,
        tfidf           ,
        text_column     ,
        category_column ,
        encoding        ,
        header          ,
        index_col       ,
        ):

        self.classifiers            = classifiers
        self.ds_path                = ds_path
        self.clean_data             = clean_data
        self.min_df                 = min_df
        self.data_size              = data_size
        self.train_size             = train_size
        self.tfidf                  = tfidf
        self.text_column            = text_column
        self.category_column        = category_column
        self.encoding               = encoding
        self.header                 = header
        self.index_col              = index_col

        print('>>> Loading: {} ...'.format(self.ds_path))
        self.df, self.df_remaining  = self.load_ds()
        self.labels                 = self.df[str(self.category_column) if clean_data else self.category_column].values
        print('>>> Vectorizing ...')
        self.vectorized             = self.vectorize()

        for x in self.classifiers:
            xd  = classifiers[x]
            print('Training: {}'.format(x.__class__.__name__))
            train_data  = self.split(xd.get('toarray'))
            model       = x.fit(train_data[0], train_data[2])
            predictions = model.predict(train_data[1])
            accuracy    = accuracy_score(train_data[3], predictions)

            self.classifiers[x]['accuracy' ]= accuracy
            self.classifiers[x]['model']    = model



    def load_ds(self):
        '''
            Loads the data set from disk.
            Args    :
                ds_path : The path to the data set.
                encoding: Data set encoding.
        '''
        #Load the original data set
        df = pd.read_csv(
            self.ds_path                ,
            encoding    = self.encoding ,
            header      = self.header   ,
            index_col   = self.index_col)

        #Check if there is a clean data set
        clean_path  = get_added_name(self.ds_path, '_clean')
        if not os.path.isfile(clean_path):

            df['clean'] = df[self.text_column].apply(clean_and_stem)

            df = df[df['clean'] != '']
            with open(clean_path, 'w') as f:
                df[[self.category_column, 'clean']].to_csv(f)

        df      = pd.read_csv(
            clean_path if self.clean_data else self.ds_path             ,
            encoding    = self.encoding                                 ,
            header      = 0     if self.clean_data else self.header     ,
            index_col   = None  if self.clean_data else self.index_col  )
        return df.sample(frac= 1)[:self.data_size], df.sample(frac= 1)[self.data_size:]

    def vectorize(self):
        '''
            Vectorizes the data set.
        '''
        values  = self.df['clean' if self.clean_data else self.text_column ].values
        if self.tfidf :
            return TfidfVectorizer(
                analyzer    = 'word'                                                ,
                stop_words  = 'english'                                             ,
                lowercase   = True                                                  ,
                tokenizer   = None if not self.clean_data else lambda x: x.split()  ,
                min_df      = self.min_df                                           ).fit_transform(values)
        else :
            return CountVectorizer(
                analyzer    = 'word'                                                ,
                stop_words  = 'english'                                             ,
                lowercase   = True                                                  ,
                tokenizer   = None if not self.clean_data else lambda x: x.split()  ,
                min_df      = self.min_df                                           ).fit_transform(values)

    def split(self, toarray):
        '''
            Split the data set into test and train data.
            Args    :
                toarray : Convert to array.
        '''
        return  train_test_split(
            self.vectorized.toarray() if toarray else self.vectorized   ,
            self.labels                                                 ,
            train_size   = self.train_size                              ,
            random_state = random.randint(0,1000)                       )

    def save(self):
        '''
            Save the trained classifiers.
        '''
        saved   = {}

        for x in self.classifiers:
            saved['classifier']     = x.__class__.name
            saved['clean_data']     = x.__class__.name

        return
