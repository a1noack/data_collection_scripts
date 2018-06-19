import sys
sys.path.append("/Users/adamnoack/Desktop/thesis_research/py35/mindwavemobilepy35")
from mindwavemobile.MindwaveDataPointReader import *
from mindwavemobile.MindwaveDataPoints import RawDataPoint
import time, csv, pandas as pd, numpy as np

class Recorder():
    def __init__(self, data_type='raw', mwm_reader=None, episode_duration=5):
        #aggregated data: the certain wave types are recorded instead of raw frequencies
        #raw: only the raw eeg readings are recorded
        self.episode_duration = episode_duration
        self.trim_size = self.episode_duration*505 #all episode data should be one size
        if(data_type == 'aggregated'):  #these are the outputs that we'd like a ML algorithm to predict
            self.data_type = 1
        else:
            self.data_type = 0
        self.mwm_reader = mwm_reader
        self._episode_data_points = []
        self.episode_data_data_frame = pd.DataFrame()

    def get_episode_data(self):
        return self.episode_data_data_frame

    def record_next_data_point(self):
        #if the next datapoint is raw, append it to the episode cache
        data_point = self.mwm_reader.readNextDataPoint()
        if (data_point.__class__ is RawDataPoint):
            raw_value = data_point._readRawValue()
            self._episode_data_points.append(raw_value)
        # elif(self.data_type != 'raw'):
        #     value = data_point.

    def save_episode_data(self, episode_type):
        #append episode data to the dataframe for this trial of n episodes and clear episode data cache
        self.trim_episode_data()
        self.episode_data_data_frame = self.episode_data_data_frame.append([self._episode_data_points])
        self.episode_data_data_frame['episode_type'] = np.array([episode_type])
        # self._episode_data_points.append(episode_type)
        self._episode_data_points.clear()

    def trim_episode_data(self):
        #make sure the length of the data from all episodes is similar
        print("trimming ", len(self._episode_data_points)-int(self.trim_size), "readings")
        self._episode_data_points = self._episode_data_points[:int(self.trim_size)]

    def record_one_episode(self, episode_type, start_time, display_freq):
        end_time = start_time + self.episode_duration
        current_time = start_time
        counter = 0
        record_times = []
        print("\tCommencing data acquisition at ", time.time())
        if (self.mwm_reader.isConnected()):
            while(current_time < end_time):
                self.record_next_data_point()
                current_time = time.time()
                if(display_freq):
                    record_times.append(current_time)
                    counter += 1
                    if (counter == 240):
                        l = np.average(np.diff(record_times))
                        print("\t\tmwm:", 1 / l, "Hz")
                        record_times = []
                        counter = 0
        self.save_episode_data(episode_type)
        self.mwm_reader.end()
        print("\tClosing data stream at", time.time())

    def record_trial(self, num_episodes):
        current_episode = 1
        while(current_episode <= num_episodes):
            self.record_one_episode()
        return self.episode_data_data_frame












