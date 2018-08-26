from django.shortcuts import render, redirect
from .core.ml import *


# Create your views here.


def classifiers(request):
    models = load_models('test_models')
    for x in models:
        del x['model']
    return render(request, 'classifiers.html',{'models': models})


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
        ]
    create_from_params(
        **{x: request.GET.get(x) for x in params}
        )
    return redirect('/classifiers')

def tweets(request):
    render(request, 'tweets.html',{})
