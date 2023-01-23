import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def precision_plot(model_dict):

    precision_dt = pd.DataFrame()
    for k, v in model_dict.items():
        precision_dt = pd.concat([precision_dt,
                pd.DataFrame(v['classification_rep']).transpose()[['precision']].rename(columns={'precision': k})], axis=1)
    # precision_dt = precision_dt.reset_index(names='category')
    precision_dt.plot(kind='bar')

def recall_plot(model_dict):
    recall_dt = pd.DataFrame()
    for k, v in model_dict.items():
        recall_dt = pd.concat([recall_dt,
                pd.DataFrame(v['classification_rep']).transpose()[['recall']].rename(columns={'recall': k})], axis=1)
    recall_dt.plot(kind='bar')

def accuracy_plot(model_dict):
    return NotImplemented

def probability_overview(model_dict):
    return NotImplemented


if __name__ == '__main__':
    from models import random_forest_model
    from processing import general_processing, bayes_classification_processing
    dt = pd.read_pickle('data/imported_audio.pkl')
    dt = dt[dt.esc10].reset_index()
    dt = general_processing(dt)
    X, y = bayes_classification_processing(dt)
    model = random_forest_model(X, y)
    model_dict = {'rf1': model,
                'rf2': model}
    precision_plot(model_dict)


