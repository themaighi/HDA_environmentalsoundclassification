import numpy as np
import pydub
import librosa
import os
import librosa.display

class Clip:
    """A single 5-sec long recording."""

    RATE = 44100   # All recordings in ESC are 44.1 kHz
    FRAME = 512    # Frame size in samples
    
    class Audio:
        """The actual audio data of the clip.
        
            Uses a context manager to load/unload the raw audio data. This way clips
            can be processed sequentially with reasonable memory usage.
        """
        
        def __init__(self, path):
            self.path = path
        
        def __enter__(self):
            # Actual recordings are sometimes not frame accurate, so we trim/overlay to exactly 5 seconds
            self.data = pydub.AudioSegment.silent(duration=5000)
            self.data = self.data.overlay(pydub.AudioSegment.from_file(self.path)[0:5000])
            self.raw = (np.fromstring(self.data._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)   # convert to float
            # The length of self.raw is he Rate * the second (makes sense)
            return(self)
        
        def __exit__(self, exception_type, exception_value, traceback):
            if exception_type is not None:
                print(exception_type, exception_value, traceback)
            del self.data
            del self.raw
        
    def __init__(self, filename, pitchshift=None, timedelay=None, noise=None, speed_change=None):
        self.filename = os.path.basename(filename)
        self.path = os.path.abspath(filename)        
        self.directory = os.path.dirname(self.path)
        self.category = self.filename.split('.')[0].split('-')[-1]
        
        self.audio = Clip.Audio(self.path)
        
        with self.audio as audio:
            if pitchshift != None:
                self._pitch_shift(pitchshift)
            if timedelay != None:
                self._shift_time(timedelay)
            if noise != None:
                NotImplemented
            if speed_change != None:
                self._speed_manipulation(speed_change)
            self._compute_mfcc(audio)    
            self._compute_zcr(audio)
            self._compute_energy(audio)
            self._compute_delta(self.mfcc)
            self._compute_delta_delta(self.mfcc)
            self._compute_delta_energy(self.energy)
            self._compute_delta_delta_energy(self.energy)
            
    def _compute_mfcc(self, audio):
        # MFCC computation with default settings (2048 FFT window length, 512 hop length, 128 bands)
        ## Play around with these settings (make the function to get more inputs)
        ## We can make this to be an input, so that we can always try different transformations 
        # the 431 is the length of the series divided by the hop length 220500/512
        # The 128 is the amount of filters that are used, so there is a good approximation of the high frequency
        self.melspectrogram = librosa.feature.melspectrogram(audio.raw, sr=Clip.RATE, hop_length=Clip.FRAME)
        self.logamplitude = librosa.amplitude_to_db(self.melspectrogram)
        self.mfcc = librosa.feature.mfcc(S=self.logamplitude, n_mfcc=13).transpose()
            
    def _compute_zcr(self, audio):
        # Zero-crossing rate
        self.zcr = []
        frames = int(np.ceil(len(audio.data) / 1000.0 * Clip.RATE / Clip.FRAME))
        
        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            self.zcr.append(np.mean(0.5 * np.abs(np.diff(np.sign(frame)))))

        self.zcr = np.asarray(self.zcr)
    
    def _compute_energy(self, audio):
        self.energy = []
        frames = int(np.ceil(len(audio.data) / 1000.0 * Clip.RATE / Clip.FRAME))
        
        for i in range(0, frames):
            frame = Clip._get_frame(audio, i)
            self.energy.append(sum((frame**2)))
    
    def _shift_time(self, timedelay):
        shift = np.random.randint(Clip.RATE * timedelay['shift_seconds'])
        if timedelay['direction'] == 'right':
            shift = -shift
        elif timedelay['direction'] == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(self.audio.raw, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        self.audio.raw = augmented_data

    def _pitch_shift(self,pitchshift):
        pitch_factor = np.random.uniform(pitchshift['pitch_range_low'], pitchshift['pitch_range_high'])
        self.audio.raw = librosa.effects.pitch_shift(self.audio.raw, sr=Clip.RATE, n_steps=pitch_factor)

    def _speed_manipulation(self, speed_change):
        speed_shift = np.random.uniform(1, speed_change['speed_factor'])
        random_choice = np.random.binomial(1,p=0.5)
        if random_choice == 1:
            speed_factor = 1/speed_shift
        if random_choice == 0:
            speed_factor = speed_shift

        augmented_data = librosa.effects.time_stretch(self.audio.raw, rate=speed_factor)
        augmented_data = librosa.util.fix_length(augmented_data, size=self.audio.raw.shape[0])
        self.audio.raw = augmented_data


    def _compute_delta(self, mfcc):
        self.delta = librosa.feature.delta(mfcc.transpose(), order=1).transpose()
    
    def _compute_delta_delta(self, mfcc):
        self.delta_delta = librosa.feature.delta(mfcc.transpose(), order=2).transpose()

    def _compute_delta_energy(self, energy):
        self.energy_delta = librosa.feature.delta(energy, order=1)

    def _compute_delta_delta_energy(self, energy):
        self.energy_delta_delta = librosa.feature.delta(energy, order=2)

    @classmethod
    def _get_frame(cls, audio, index):
        if index < 0:
            return None
        return audio.raw[(index * Clip.FRAME):(index+1) * Clip.FRAME]
    
    def __repr__(self):
        return '<{0}/{1}>'.format(self.category, self.filename)