import sys
import multiprocessing, time
sys.path.append("/Users/adamnoack/Desktop/thesis_research/py35/mindwavemobilepy35")
from mindwavemobile.MindwaveDataPointReader import MindwaveDataPointReader
from mindwavemobile.MindwaveDataPoints import RawDataPoint, StartCommand, EEGPowersDataPoint
from environment.environment import PacmanEnv
import pandas as pd, numpy as np
from random import randint

#essentially a wrapper around recorder's record episode fcn. Used to synchronize data gathering and animation
def read(queue, mwm_pipeend, display_freq, episode_duration):
    #used for data acquisition timing
    counter = 0
    record_times = []

    #mwm collects data at approximately 512 Hz. Can vary though. That's why we need the trim.
    trim_size = (episode_duration - .4) * 508
    print("episode_duration:",episode_duration)
    episode_data = []
    #creates a new mindwave mobile object
    mwm_reader = MindwaveDataPointReader()
    mwm_reader.start()

    #finds the start signal
    while True:
        data_point = mwm_reader.readNextDataPoint()
        if data_point.__class__ is StartCommand:
            print("Start sequence received, signaling animation...")
            #once the start command is received from the mwm, the start signal is sent through the pipe to the animation process
            mwm_pipeend.send([True])
            break
    epi_type = mwm_pipeend.recv()[0]

    print("\tCommencing data acquisition at", time.time())
    #for as long as the episode is happening, read data from mwm. For some reason, mwm closes too late. Need early stopping
    #that's why I subtract .4 seconds from the end_time here
    end_time = (time.time() + episode_duration - .4) - .1
    while time.time() < end_time:
        data_point = mwm_reader.readNextDataPoint()
        if(data_point.__class__ is RawDataPoint):
            episode_data.append(data_point._readRawValue())
            if (display_freq):
                current_time = time.time()
                record_times.append(current_time)
                counter += 1
                if (counter == 240):
                    l = np.average(np.diff(record_times))
                    print("\t\tmwm:", 1 / l, "Hz")
                    record_times = []
                    counter = 0
    print("\tClosing mwm stream at", time.time())

    #get the episode type from the environment
    # print("trimming ", len(episode_data) - int(trim_size), "readings")
    # trim the episode data so it's always a constant length
    print("trimming:",len(episode_data) - trim_size)
    episode_data = episode_data[:int(trim_size)]
    episode_data_df = pd.DataFrame().append([episode_data])
    episode_data_df['episode_type'] = np.array([epi_type])

    #close and delete mwm
    mwm_reader.end()
    del mwm_reader
    #when the episode is finished, the data gathered by the recorder device is accessed and placed in the queue so the parent process can obtain it
    queue.put(episode_data_df)

#update stored data
def update_data_base(trial_results):
    try:
        existing_data = pd.read_pickle(data_file_path)
        updated_data = pd.concat([existing_data, trial_results])
    except FileNotFoundError:
        print("No data base found. Creating new data file...\n")
        updated_data = trial_results
    updated_data.reset_index(inplace=True)
    updated_data.index.name = 'index'
    del updated_data['index']
    updated_data.to_pickle(data_file_path)
    print("TRIAL DATA SUCCESSFULLY SAVED :\n",updated_data)

#obtain most recent trial number
def get_trial_num():
    try:
        return pd.read_pickle(data_file_path).tail(n=1).reset_index().at[0,'trial_num']+1
    except FileNotFoundError:
        return 1

#this idiom is necessary. If objects or variables are instantiated outside of this area, things get weird.
if __name__ == '__main__':
    save_data = True
    # the results from this trial are appended to this data frame
    trial_results = pd.DataFrame()
    # display the frequencies that each process is running at
    display_freq = True
    trial_type = -1
    current_score = 0
    cur_epi = 0
    # 1 if the environment is controlled. 0 if completely random.
    control = 1
    data_file_path = 'data/data_current.pkl'
    # 3 episodes take 27sec. 20 episodes take roughly 3min 10sec.
    num_episodes = 30
    deviate = 0

    while not trial_type in [0,1]:
        trial_type = int(input("ENTER TRIAL TYPE: "))
        deviate = int(input("ENTER DEV: "))
    print("\n\n\n")

    if(deviate == -100):
        deviate = randint(-2,2)


    # the amount of time the participant has to notice the initial state of the environment. Happens before Pacman indicates which direction he'll take
    hangtime = 2  # seconds
    # create the environment with the probabilities determining each epi_type
    env = PacmanEnv(num_episodes=num_episodes, reward=100, punishment=150, scale=10, move_left=.5, is_beneficial=.5,
                    update_freq=20, hangtime=hangtime, speed=7, deviate=deviate, win_value=600)
    # return the calculated episode duration from the environment
    episode_duration = env.episode_duration
    sequence = env.sequence
    print(sequence)


    #for some reason the fork start method doesn't work. Spawn must be used. Also, it won't switch to spawn unless force is set to true
    multiprocessing.set_start_method('spawn', force=True)

    while(cur_epi < num_episodes):
        print("Episode", cur_epi, "/", num_episodes-1)

        epi_rem = num_episodes - cur_epi

        #get the episode type from the environment. The episode type is related to where the reward is and where Pacman is heading
        epi_type = env.epi_type

        #this queue is used to send data back to this main, parent process from the data gathering function/process
        queue = multiprocessing.Queue()
        score_queue = multiprocessing.Queue()
        #this pipe is used to sync the data acquisition process and the animation display process.
        mwm_pipeend, ani_pipeend = multiprocessing.Pipe()

        win_lose = sequence[cur_epi]

        #both of the processes are created, started, and then join (wait for the other one to finish)
        p1 = multiprocessing.Process(target=read, args=(queue, mwm_pipeend, display_freq, episode_duration))
        p2 = multiprocessing.Process(target=env.simulate_one_epi, args=(ani_pipeend, display_freq, epi_rem, cur_epi, control, win_lose, score_queue, current_score))


        p1.start()
        p2.start()

        #gets the episode data from read()
        trial_results = pd.concat([trial_results, queue.get()])
        #get the current score from the environment because apparently it doesn't keep track on its own
        current_score = score_queue.get()[0]

        p1.join()
        p2.join()

        p1.terminate()
        p2.terminate()

        print("Episode finished. \n\n")

        #rebuilds the environment in a random fashion
        env.reset_env()

        cur_epi += 1

    trial_results['episode_num'] = np.array(list(range(num_episodes)))

    trial_num = get_trial_num()
    trial_results['trial_type'] = pd.Series(np.array([trial_type]*num_episodes))
    trial_results['trial_num'] = pd.Series(np.array([trial_num]*num_episodes))

    trial_results.reset_index(inplace=True)
    trial_results.index.name = 'index'
    del trial_results['index']
    #appends this trial's data to saved data and resaves it
    if(save_data):
        update_data_base(trial_results)
    else:
        print("UNSAVED TRIAL RESULTS:", trial_results)









