from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scripts.processing import bayes_classification_processing, general_processing, rnn_classification_processing
import pandas as pd
from scripts.audio_importer import Clip
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import copy

## I can use the shape value to understand the actual 
## I could test the robustness of the method by adding noise to the series and try to fit it
## Always use stratification when you sample 
## Normalize the input before adding them
## Add noise to the input
## Add hyperparameter tuning for the random forest
## Add k-fold estimation? Maybe not
## Using an Elbow method to clasterize the groups before and then create N model based on how many cluster exists

def random_forest_model(X_train, y_train, X_test, y_test, variables=None, param_grid={}):


    
    if variables is not None:
        X_train=X_train[variables]
        X_test = X_test[variables]

       
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    rfc = RandomForestClassifier(random_state=0)
    rf_random = RandomizedSearchCV(estimator = rfc,
                                     param_distributions = param_grid,
                                     n_iter = 100,
                                      cv = 3,
                                       verbose=2,
                                        random_state=42,
                                         n_jobs = -1,
                                         scoring='accuracy')
    rf_random.fit(X_train, y_train)
    rfc = copy.deepcopy(rf_random.best_estimator_)
    rfc.fit(X_train, y_train)

    output_dict = {}
    output_dict['model'] = rf_random.best_estimator_
    output_dict['y_proba'] = rfc.predict_proba(X_test)
    output_dict['y_pred'] = rfc.predict(X_test)
    output_dict['y_true'] = y_test
    output_dict['confusion_mat'] = confusion_matrix(y_true=y_test, y_pred=output_dict['y_pred'])
    output_dict['classification_rep'] = classification_report(y_test, output_dict['y_pred'], digits=4, output_dict=True)
    output_dict['X_test'] = X_test
    output_dict['y_test'] = y_test
    return output_dict


def rnn_model(X_train, y_train, X_test, y_test):

    # X = X[:,:,1:]
    le = OneHotEncoder()
    y_train = le.fit_transform(y_train.values.reshape(-1,1)).toarray()
    y_test = le.transform(y_test.values.reshape(-1,1)).toarray()

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123, stratify=y_train)

    minmax_sc = StandardScaler()
    X_train = minmax_sc.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = minmax_sc.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    X_val = minmax_sc.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    input_shape=(431,14)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, return_sequences=False, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(24, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.summary()
 
    model.compile(optimizer='adam',loss='CategoricalCrossentropy',metrics=['acc'])
    history = model.fit(np.array(X_train), y_train, epochs=50, batch_size=50, 
                    validation_data=(np.array(X_val), y_val), shuffle=False)

    predict_proba = model.predict(X_test)
    y_predicted = np.zeros(predict_proba.shape)
    y_predicted[np.arange(len(y_predicted)), predict_proba.argmax(axis=-1)] = 1

    output_dict = {}
    output_dict['model'] = model
    output_dict['y_proba'] = model.predict(X_test)
    output_dict['y_pred'] = le.inverse_transform(y_predicted).reshape(-1)
    output_dict['y_true'] = le.inverse_transform(y_test).reshape(-1)
    output_dict['confusion_mat'] = confusion_matrix(y_true=output_dict['y_true'], y_pred=output_dict['y_pred'])
    output_dict['classification_rep'] = classification_report(output_dict['y_true'], output_dict['y_pred'], digits=4, output_dict=True)
    output_dict['history'] = history
    output_dict['encoder'] = le
    output_dict['scaler'] = minmax_sc

    return output_dict



def cnn_rnn_model(X_train, y_train, X_test, y_test):

    # X = X[:,:,1:]
    le = OneHotEncoder()
    y_train = le.fit_transform(y_train.values.reshape(-1,1)).toarray()
    y_test = le.transform(y_test.values.reshape(-1,1)).toarray()

    # for i in range(X.shape[0]):
    #     mu = np.mean(X[i,:,:])
    #     std = np.std(X[i,:,:])
    #     X[i,:,:] = (X[i,:,:] - mu)/std

    # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=123, stratify=y_encoded)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123, stratify=y_train)

    minmax_sc = StandardScaler()
    X_train = minmax_sc.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = minmax_sc.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    X_val = minmax_sc.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    input_shape=(431,14)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=15, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=8))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=4))
    # model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.LSTM(128, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(24, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dense(64, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.Dense(24, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.summary()
 
    model.compile(optimizer='adam',loss='CategoricalCrossentropy',metrics=['acc'])
    history = model.fit(np.array(X_train), y_train, epochs=50, batch_size=50, 
                    validation_data=(np.array(X_val), y_val), shuffle=False)

    predict_proba = model.predict(X_test)
    y_predicted = np.zeros(predict_proba.shape)
    y_predicted[np.arange(len(y_predicted)), predict_proba.argmax(axis=-1)] = 1

    output_dict = {}
    output_dict['model'] = model
    output_dict['y_proba'] = model.predict(X_test)
    output_dict['y_pred'] = le.inverse_transform(y_predicted).reshape(-1)
    output_dict['y_true'] = le.inverse_transform(y_test).reshape(-1)
    output_dict['confusion_mat'] = confusion_matrix(y_true=output_dict['y_true'], y_pred=output_dict['y_pred'])
    output_dict['classification_rep'] = classification_report(output_dict['y_true'], output_dict['y_pred'], digits=4, output_dict=True)
    output_dict['history'] = history
    output_dict['encoder'] = le
    output_dict['scaler'] = minmax_sc

    return output_dict


if __name__ == '__main__':
    path = 'data/imported_audio.pkl'
    dt = pd.read_pickle(path)
    dt = dt[dt.esc10].reset_index()
    dt = general_processing(dt)
    # X_train, y_train = bayes_classification_processing(dt[dt.train == 1])
    # X_test, y_test = bayes_classification_processing(dt[dt.train == 0])


    # random_forest_model(X_train, y_train, X_test, y_test, variables=X.filter(regex='^mfcc_avg_').columns,
    #  param_grid = {'max_depth': [None, 10, 20, 30],
    #                 'n_estimators': [10, 50, 100, 200]} )
    X_train, y_train = rnn_classification_processing(dt[dt.train == 1])
    X_test, y_test = rnn_classification_processing(dt[dt.train == 0])

    rnn_model(X_train,y_train, X_test, y_test)

