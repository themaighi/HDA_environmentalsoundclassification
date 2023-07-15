import os
import shutil
import urllib
import zipfile
import glob
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
import librosa
from scripts.audio_importer import Clip
import os
import numpy as np
from sklearn.model_selection import train_test_split
import requests



def download_dataset(name):
    """Download the dataset into current working directory."""
    if not os.path.exists(name):
        os.makedirs(name, exist_ok=True)
        urllib.request.urlretrieve('https://github.com/karoldvl/{0}/archive/master.zip'.format(name), '{0}/{0}.zip'.format(name))

        # request = requests.get('https://github.com/karoldvl/{0}/archive/master.zip'.format(name), timeout=10, stream=True)
        # with open('{0}/{0}.zip'.format(name), 'wb') as fh:
        #     # Walk through the request response in chunks of 1024 * 1024 bytes, so 1MiB
        #     for i, chunk in enumerate(request.iter_content(1024 * 1024)):
        #         print(f'Downloading part {str(i)} ---------')
        #         # Write the chunk to the file
        #         fh.write(chunk)

        with zipfile.ZipFile('{0}/{0}.zip'.format(name)) as package:
            package.extractall('{0}/'.format(name))

        os.unlink('{0}/{0}.zip'.format(name))   


def load_dataset(path, augmentation=False):
    """Load all dataset recordings into a nested list."""
    
    
    reference_table = pd.read_csv('ESC-50/ESC-50-master/meta/esc50.csv')
    reference_table = reference_table[reference_table.esc10]

    X_train, X_test = train_test_split(reference_table['filename'].values, test_size=0.25, stratify=reference_table['category'].values)
    
    train_clips = {}
    for directory in X_train:
        print(f'---------------{directory}--------------')
        list_clips = []
        list_clips.append(Clip('{0}/{1}'.format(path, directory)))
        if augmentation:
            for i in range(5):
                list_clips.append(Clip('{0}/{1}'.format(path, directory),
                                    timedelay={'shift_seconds': 2,
                                                'direction': 'both'}))
            for i in range(5):
                list_clips.append(Clip('{0}/{1}'.format(path, directory),
                                    pitchshift={'pitch_range_low': -2,
                                                'pitch_range_high': 2}))
            
            for i in range(5):
                list_clips.append(Clip('{0}/{1}'.format(path, directory),
                                    speed_change={'speed_factor': 2}))
        train_clips[directory] = list_clips

    test_clips = {}

    for directory in X_test:
        print(f'---------------{directory}--------------')
        list_clips = []
        list_clips.append(Clip('{0}/{1}'.format(path, directory)))
        test_clips[directory] = list_clips

    print('All {0} recordings loaded.'.format(path))  

    try:
        train_dt = pd.DataFrame(train_clips, index=[0]).melt(var_name='filename', value_name='audio')
    except:
        train_dt = pd.DataFrame(train_clips).melt(var_name='filename', value_name='audio')

    test_dt = pd.DataFrame(test_clips, index=[0]).melt(var_name='filename', value_name='audio')          
    train_dt['train'] = 1
    test_dt['train'] = 0
    dt = pd.concat([train_dt, test_dt])
    dt = dt.merge(reference_table, on='filename', how='left')
    
    return dt

def save_data(dt, path='data/imported_audio_original.pkl'):
    dt.to_pickle(path)


if __name__ == '__main__':
    # download_dataset('ESC-50')s
    dt = load_dataset('ESC-50/ESC-50-master/audio/')
    save_data(dt)