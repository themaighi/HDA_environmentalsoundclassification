from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from processing import bayes_classification_processing, general_processing, rnn_classification_processing
import pandas as pd
from scripts.audio_importer import Clip
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np

## I can use the shape value to understand the actual 
## I could test the robustness of the method by adding noise to the series and try to fit it
## Always use stratification when you sample 
## Normalize the input before adding them
## Add noise to the input


def random_forest_model(X, y, variables=None):
    
    if variables is not None:
        X=X[variables]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(X_train, y_train)
    output_dict = {}
    output_dict['model'] = rfc
    output_dict['y_proba'] = rfc.predict_proba(X_test)
    output_dict['y_pred'] = rfc.predict(X_test)
    output_dict['y_true'] = y_test
    output_dict['confusion_mat'] = confusion_matrix(y_true=y_test, y_pred=output_dict['y_pred'])
    output_dict['classification_rep'] = classification_report(y_test, output_dict['y_pred'], digits=4, output_dict=True)

    return output_dict


def rnn_model(X, y):

    # X = X[:,:,1:]
    le = OneHotEncoder()
    y_encoded = le.fit_transform(y.values.reshape(-1,1)).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=123, stratify=y_encoded)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123, stratify=y_train)

    minmax_sc = StandardScaler()
    X_train = minmax_sc.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = minmax_sc.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    X_val = minmax_sc.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    input_shape=(431,128)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(248, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(24, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.summary()
 
    model.compile(optimizer='adam',loss='CategoricalCrossentropy',metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=50, 
                    validation_data=(X_val, y_val), shuffle=False)

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
    # X, y = bayes_classification_processing(dt)
    # random_forest_model(X, y, variables=X.filter(regex='^mfcc_avg_').columns)
    X, y = rnn_classification_processing(dt)
    rnn_model(X,y)

