from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from processing import bayes_classification_processing, general_processing
import pandas as pd
from scripts.audio_importer import Clip
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

## I can use the shape value to understand the actual 


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
    

if __name__ == '__main__':
    path = 'data/imported_audio.pkl'
    dt = pd.read_pickle(path)
    dt = dt[dt.esc10].reset_index()
    dt = general_processing(dt)
    X, y = bayes_classification_processing(dt)
    random_forest_model(X, y, variables=X.filter(regex='^mfcc_avg_').columns)

