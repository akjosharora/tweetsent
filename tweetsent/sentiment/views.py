from django.shortcuts   import render, redirect
from .core.ml           import *
from .core.twitter      import *

# Create your views here.


def classifiers(request):
    models = load_models('models')
    for x in models:
        del x[1]['model']
    return render(request, 'classifiers.html',{'models': models})

def delete_classifier(request):
    delete(request.GET.get('id'))
    return redirect('/classifiers')

def add_classifier(request):
    params = [
        'classifier'        ,
        'ds_path'           ,
        'clean_data'        ,
        'min_df'            ,
        'data_size'         ,
        'train_size'        ,
        'tfidf'             ,
        'text_column'       ,
        'category_column'   ,
        'encoding'          ,
        'header'            ,
        'index_col'         ,
        'toarray'           ,
        'max_features'
        ]
    create_from_params(
        **{x: request.GET.get(x) for x in params}
        )
    return redirect('/classifiers')

def tweets(request):
    tag     = request.GET.get('tag', 'bitcoin')
    tweets  = get_last_tweets(tag, auth())
    result  = predict_all(tweets)
    print(len(result))
    return render(request, 'tweets.html',{'result': result})
