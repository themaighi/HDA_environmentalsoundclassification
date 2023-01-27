import pandas as pd
import pandas as pd
import numpy as np


def general_processing(dt):

    for j, audio in enumerate(dt['audio']):
        mfcc = pd.DataFrame(audio.mfcc, columns=['mfcc_' + str(k) for k in range(audio.mfcc.shape[1])])
        zcr = pd.DataFrame({'zcr': audio.zcr})
        energy = pd.DataFrame({'energy': audio.energy})
        energy_delta = pd.DataFrame({'energy_delta': audio.energy_delta})
        energy_delta_delta = pd.DataFrame({'energy_delta2': audio.energy_delta_delta})
        delta = pd.DataFrame(audio.delta, columns=['delta_' + str(k) for k in range(audio.delta.shape[1])])
        delta2 = pd.DataFrame(audio.delta, columns=['delta2_' + str(k) for k in range(audio.delta_delta.shape[1])])
        var_dt = pd.concat([mfcc, zcr, energy, delta, delta2, energy_delta, energy_delta_delta], axis=1)
        dt['audio'][j].x_df = var_dt

    return dt


def bayes_classification_processing(dt):
    mfcc_average_list = []
    mfcc_std_list = []
    zcr_average_list = []
    zcr_std_list = []
    energy_average_list = []
    energy_std_list = []
    energy_delta_average_list = []
    energy_delta_std_list = []
    delta_average_list = []
    delta_std_list = []


    for i in range(dt.shape[0]):
        mfcc_average_list.append(np.mean(dt['audio'][i].mfcc, axis=0))
        mfcc_std_list.append(np.std(dt['audio'][i].mfcc, axis=0))
        zcr_average_list.append(np.mean(dt['audio'][i].zcr, axis=0))
        zcr_std_list.append(np.std(dt['audio'][i].zcr, axis=0))
        energy_average_list.append(np.mean(dt['audio'][i].energy, axis=0))
        energy_std_list.append(np.std(dt['audio'][i].energy, axis=0))
        energy_delta_average_list.append(np.mean(dt['audio'][i].energy_delta, axis=0))
        energy_delta_std_list.append(np.std(dt['audio'][i].energy_delta, axis=0))
        delta_average_list.append(np.mean(dt['audio'][i].delta, axis=0))
        delta_std_list.append(np.std(dt['audio'][i].delta, axis=0))

    avg_mfcc_dt = pd.DataFrame(mfcc_average_list, columns=['mfcc_avg_' + str(i) for i in range(mfcc_average_list[0].shape[0])]) 
    std_mfcc_dt = pd.DataFrame(mfcc_std_list, columns=['mfcc_std_' + str(i) for i in range(mfcc_average_list[0].shape[0])])
    avg_zcr_dt = pd.DataFrame(zcr_average_list, columns=['zcr_avg']) 
    std_zcr_dt = pd.DataFrame(zcr_std_list, columns=['zcr_std']) 
    avg_energy_dt = pd.DataFrame(energy_average_list, columns=['energy_avg']) 
    std_energy_dt = pd.DataFrame(energy_std_list, columns=['energy_std']) 
    avg_energy_delta_dt = pd.DataFrame(energy_delta_average_list, columns=['delta_energy_avg']) 
    std_energy_delta_dt = pd.DataFrame(energy_delta_std_list, columns=['delta_energy_std'])
    avg_delta_dt = pd.DataFrame(delta_average_list, columns=['delta_mfcc_avg_' + str(i) for i in range(delta_average_list[0].shape[0])]) 
    std_delta_dt = pd.DataFrame(delta_std_list, columns=['delta_mfcc_std_' + str(i) for i in range(mfcc_average_list[0].shape[0])])


    X = pd.concat([avg_mfcc_dt, std_mfcc_dt, avg_zcr_dt, std_zcr_dt,
                    avg_energy_dt, std_energy_dt, avg_energy_delta_dt,
                    std_energy_delta_dt, avg_delta_dt, std_delta_dt], axis=1)
    y = dt['category']

    return X, y


def rnn_classification_processing(dt):
    ##  I need to use numerical values for the categories, but I should use a predefined cetegory values
    
    X = []
    for i in range(dt.shape[0]):
        

        X.append(np.concatenate([dt.audio[i].logamplitude.transpose()[:,:13],
                        dt.audio[i].zcr.reshape(-1,1),
                        dt.audio[i].melspectrogram_power1.transpose()[:,:13],
                        dt.audio[i].delta[:,:13]], axis=1))
        # X.append(np.concatenate([dt.audio[i].mfcc,
        #                 dt.audio[i].zcr.reshape(-1,1),
        #                 np.array(dt.audio[i].energy).reshape(-1,1),
        #                 np.array(dt.audio[i].energy_delta).reshape(-1,1),
        #                 dt.audio[i].delta], axis=1))
        # X.append(dt.audio[i].mfcc)


    X = np.array(X)
    y = dt['category']

    return X, y


if __name__ == '__main__':
    path = 'data/imported_audio.pkl'
    dt = pd.read_pickle(path)
    dt = dt[dt.esc10].reset_index()
    dt = general_processing(dt)
    # bayes_classification_processing(dt)
    rnn_classification_processing(dt)
