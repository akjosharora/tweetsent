'''
    Cleans the text and set it for processing.
'''

import  re
import  nltk

from    nltk.corpus     import wordnet  as wn
try :
    nltk.download('stopwords')
except :
    pass

ANTO_DICT   = {}
for i in wn.all_synsets():
    for j in i.lemmas():
        if j.antonyms():
            ANTO_DICT[j.name()] = j.antonyms()[0].name()
STOP_WORDS     = nltk.corpus.stopwords.words('english')
STOP_WORDS.remove('not')

#Contains regex extressions for cleaning
CLEAN_REGEX = {
    'url'       : 'http.?://[^\s]+[\s]?'                    ,
    'username'  : '@[^\s]+[\s]?'                            ,
    'tag'       : '#[^\s]+[\s]?'                            ,
    'empty'     : 'Not Available'                           ,
    'number'    : '\s?\d+\.?\d*'                            ,
    'special'   : '[^\w+\s]'                                ,
    }

def clean_text(text):
    '''
        Cleans a text from Urls, Usernames, Empty texts, Special Characters, Numbers and Hashtags.
        Args    :
            text   : The text.
        Returns :   The clean text.
    '''
    try :
        #Create the combined expression to eliminate all unwanted strings
        clean_ex    = '|'.join(['(?:{})'.format(x) for x in CLEAN_REGEX.values()])
        result      = re.sub(clean_ex, '', text).strip()

        #Remove character sequences
        result      = re.sub('([a-z])\\1+', '\\1\\1', result)
        result      = re.sub('[ ]+', ' ', result).strip()

        return result
    except :
        #Some times the DataFrame contains Nan
        return ''

def tokenize_text(text):
    '''
        Tokenization.
        Args    :
            text   : The text.
        Returns :   A list of tokens.
    '''
    #Using SnowballStemmer for english
    return nltk.word_tokenize(text,'english')

def replace_antonyms(text):
    '''
        Replace a word preceded by "not" or "no" with its antonym.
        Args    :
            text   : The text to process.
        Returns :   A text with each preceded by "not" or "no" replaced by its antonym.
    '''
    #Replace all words that end with n't with  not
    nt_expression   = '\w+n\'t'
    text           = re.sub(nt_expression, 'not', text).strip()

    expression      = '((?:not|no) (\w+))'
    not_words       = re.compile(expression).findall(text)

    for x in not_words :
        if ANTO_DICT.get(x[1]) :
            text = text.replace(x[0], ANTO_DICT.get(x[1]))

    return text

def remove_stop_words(tokens):
    '''
        Removes stop words from a text
        Args    :
            tokens      : The tokenized text.
        Returns :   Tokens that are not a stop word.
    '''
    return [x for x in tokens if x not in STOP_WORDS]

def stem_text(tokens):
    '''
        Stemming is the process of reducing a derived word to it's original word.
        Args    :
            tokens  : The original non stemmed tokens.
        Returns :   The stemmed tokens.
    '''
    #Using SnowballStemmer for english
    stemmer     = nltk.SnowballStemmer('english')
    return [stemmer.stem(x) for x in tokens]

def clean_and_stem(text):
    '''
        Clean and stem the text.
        Args    :
            text   : The text to process.
        Returns :   A list of stemmed clean tokens.
    '''
    text    = replace_antonyms  (text)
    text    = clean_text        (text)
    tokens  = tokenize_text     (text)
    tokens  = remove_stop_words (tokens)
    tokens  = stem_text         (tokens)

    return tokens
