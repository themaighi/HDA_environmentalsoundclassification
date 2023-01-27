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


def download_dataset(name):
    """Download the dataset into current working directory."""
    if not os.path.exists(name):
        os.mkdir(name)
        urllib.request.urlretrieve('https://github.com/karoldvl/{0}/archive/master.zip'.format(name), '{0}/{0}.zip'.format(name))

        with zipfile.ZipFile('{0}/{0}.zip'.format(name)) as package:
            package.extractall('{0}/'.format(name))

        os.unlink('{0}/{0}.zip'.format(name))   


def load_dataset(path):
    """Load all dataset recordings into a nested list."""
    clips = {}
    
    for directory in sorted(os.listdir('{0}/'.format(path))):
        print(f'---------------{directory}--------------')
        list_clips = []
        list_clips.append(Clip('{0}/{1}'.format(path, directory)))
        # for i in range(5):
        #     list_clips.append(Clip('{0}/{1}'.format(path, directory),
        #                         timedelay={'shift_seconds': 2,
        #                                     'direction': 'both'}))
        # for i in range(5):
        #     list_clips.append(Clip('{0}/{1}'.format(path, directory),
        #                         pitchshift={'pitch_range_low': -4,
        #                                     'pitch_range_high': 4}))
        
        # for i in range(5):
        #     list_clips.append(Clip('{0}/{1}'.format(path, directory),
        #                         speed_change={'speed_factor': 2}))
        clips[directory] = list_clips

    print('All {0} recordings loaded.'.format(path))            
    
    return clips

def save_data(dt, path='data/imported_audio.pkl'):
    output = pd.DataFrame(dt, index=[0]).melt(var_name='filename', value_name='audio')
    reference_table = pd.read_csv('ESC-50/meta/esc50.csv')
    output = output.merge(reference_table, on='filename', how='left')
    output.to_pickle(path)


if __name__ == '__main__':
    # download_dataset('ESC-50')
    dt = load_dataset('ESC-50/audio/')
    save_data(dt)