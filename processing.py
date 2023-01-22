import pandas as pd
import numpy as np


def general_processing(dt):

    for j, audio in enumerate(dt['audio']):
        mfcc = pd.DataFrame(audio.mfcc, columns=['mfcc_' + str(k) for k in range(audio.mfcc.shape[1])])
        zcr = pd.DataFrame({'zcr': audio.zcr})
        var_dt = pd.concat([mfcc, zcr], axis=1)
        dt['audio'][j].x_df = var_dt

    return dt


def bayes_classification_processing(dt):
    mfcc_average_list = []
    mfcc_std_list = []
    for i in range(dt.shape[0]):
        mfcc_average_list.append(np.mean(dt['audio'][i].mfcc, axis=0))
        mfcc_std_list.append(np.std(dt['audio'][i].mfcc, axis=0))

    avg_mfcc_dt = pd.DataFrame(mfcc_average_list, columns=['mfcc_avg_' + str(i) for i in range(mfcc_average_list[0].shape[0])]) 
    std_mfcc_dt = pd.DataFrame(mfcc_std_list, columns=['mfcc_avg_' + str(i) for i in range(mfcc_average_list[0].shape[0])]) 

    X = pd.concat([avg_mfcc_dt, std_mfcc_dt], axis=1)
    y = dt['category']

    return X, y


if __name__ == '__main__':
    path = 'data/imported_audio.pkl'
    dt = pd.read_pickle(path)
    dt = dt[dt.esc10].reset_index()
    general_processing(dt)
