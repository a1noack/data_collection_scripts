import sys
sys.path.append("/Users/adamnoack/Desktop/thesis_research/py35/mindwavemobilepy35")
import environment.recorder as rcrdr
from mindwavemobile.MindwaveDataPointReader import MindwaveDataPointReader
from mindwavemobile.MindwaveDataPoints import RawDataPoint, StartCommand, EEGPowersDataPoint
import multiprocessing, time, bluetooth
import pandas as pd

def read(queue):
    global results
    mwm_reader = MindwaveDataPointReader()
    mwm_reader.start()
    recorder = rcrdr.Recorder(episode_duration=episode_duration, mwm_reader=mwm_reader)
    time_out_time = time.time() + 10
    while time.time() < time_out_time:
        data_point = mwm_reader.readNextDataPoint()
        if data_point.__class__ is StartCommand:
            print("start sequence received, commencing episode...")
            break
    recorder.record_one_episode(1, time.time())
    print(recorder.get_episode_data())
    queue.put(recorder.get_episode_data())
    # Just the bluetooth connection example commented out below
    #
    # end_time = time.time() + episode_duration
    # while time.time() < end_time:
    #     data_point = mwm_reader.readNextDataPoint()
    #     if (data_point.__class__ is RawDataPoint:
    # socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    # try:
    #     socket.connect(('B0:B4:48:F6:38:A1', 1))
    #     print("\tconnected!!")
    # except Exception as e:
    #     print("\tfailed :(")


def wait(wait_time, iteration):
    print("hehe")


episode_duration = 2
num_episodes = 3


results = pd.DataFrame()

if __name__ == '__main__':
    # multiprocessing.freeze_support()
    for i in range(1, num_episodes + 1):
        start_time = time.time()
        # the default method is 'fork'. fork makes a replica of the parent process while 'spawn' creates a new, blank default process
        multiprocessing.set_start_method('spawn', force=True)

        queue = multiprocessing.Queue()

        process1 = multiprocessing.Process(target=read, args=(queue,))
        process2 = multiprocessing.Process(target=wait, args=(episode_duration, i,))

        process1.start()
        process2.start()

        results = pd.concat([results, queue.get()])

        process1.join()
        process2.join()
    print(results)

