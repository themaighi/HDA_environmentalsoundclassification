import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def precision_plot(model_dict, figsize):

    precision_dt = pd.DataFrame()
    for k, v in model_dict.items():
        precision_dt = pd.concat([precision_dt,
                pd.DataFrame(v['classification_rep']).iloc[:, :-2].transpose()[['precision']].rename(columns={'precision': k})], axis=1)
    # precision_dt = precision_dt.reset_index(names='category')
    precision_dt.plot(kind='bar', figsize=figsize)

def recall_plot(model_dict, figsize):
    recall_dt = pd.DataFrame()
    for k, v in model_dict.items():
        recall_dt = pd.concat([recall_dt,
                pd.DataFrame(v['classification_rep']).iloc[:, :-2].transpose()[['recall']].rename(columns={'recall': k})], axis=1)
    recall_dt.plot(kind='bar', figsize=figsize)

def accuracy_plot(model_dict):
    return NotImplemented

def probability_overview(model, classes, figsize=(18, 8)):

    y_proba = model['y_proba']
    y_true = model['y_true']


    prob_dt = pd.DataFrame(y_proba, columns=classes)
    prob_dt['y_true'] = y_true.reset_index(drop=True)

    fig, axs = plt.subplots(len(prob_dt.y_true.unique()), 1, figsize=figsize)
    ax_flatten = axs.flatten()
    for i, category in enumerate(prob_dt.y_true.unique()):
        plot_dt = prob_dt[prob_dt.y_true == category].copy()
        plot_dt.boxplot(ax=ax_flatten[i])
        ax_flatten[i].title.set_text(category)
    
    plt.tight_layout()



if __name__ == '__main__':
    from models import random_forest_model
    from processing import general_processing, bayes_classification_processing
    dt = pd.read_pickle('data/imported_audio_original.pkl')
    dt = dt[dt.esc10].reset_index()
    dt = general_processing(dt)
    X_train, y_train = bayes_classification_processing(dt[dt.train == 1])
    X_test, y_test = bayes_classification_processing(dt[dt.train == 0].reset_index(drop=True))

    model = random_forest_model(X_train, y_train, X_test, y_test)
    model_dict = {'rf1': model,
                'rf2': model}
    precision_plot(model_dict, figsize=(18,12))
    probability_overview(model_dict['rf1'], classes = model_dict['rf1']['model'].classes_)


