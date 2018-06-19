from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import multiprocessing, time
sys.path.append("/Users/adamnoack/Documents/GitHub/Noack-Adam-Work/fall17_spring18/py35/mindwavemobilepy35")
sys.path.append(os.getcwd())
from mindwavemobile.MindwaveDataPointReader import MindwaveDataPointReader
from mindwavemobile.MindwaveDataPoints import RawDataPoint, StartCommand, EEGPowersDataPoint
from environment.environment import PacmanEnv

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf
sys.path.append('..')
from random import randint


tf.logging.set_verbosity(tf.logging.INFO)

#essentially a wrapper around recorder's record episode fcn. Used to synchronize data gathering and animation
def read(queue, mwm_pipeend, display_freq, episode_duration, num_episodes): #num_episodes
    counter = 0
    record_times = []
    current_episode = 0
    trim_size = (episode_duration - .4) * 508 - 20
    trial_data_df = pd.DataFrame()

    # creates a new mindwave mobile object
    mwm_reader = MindwaveDataPointReader()
    mwm_reader.start()

    # finds the start signal
    while True:
        data_point = mwm_reader.readNextDataPoint()
        if data_point.__class__ is StartCommand:
            print("Start sequence received.")
            break

    start_time = time.time()
    #idles for 3 seconds to jump past annoying start spike
    while (time.time() < start_time+12):
        mwm_reader.readNextDataPoint()
        if (display_freq):
            current_time = time.time()
            record_times.append(current_time)
            counter += 1
            if (counter == 512):
                l = np.average(np.diff(record_times))
                print("\t\tmwm idling")
                record_times = []
                counter = 0

    # once the start command is received from the mwm, the start signal is sent through the pipe to the animation process
    mwm_pipeend.send(["start_trial"])
    print("Startup spike passed. Trial begins.\n")

    while(current_episode < num_episodes):
        counter = 0
        record_times = []
        #used for data acquisition timing
        episode_data = []
        episode_reading_times = []
        epi_type = mwm_pipeend.recv()[0]
        mwm_reader.clearBuffer()
        # mwm_reader.clearBufferV2()
        mwm_pipeend.send(["buffer_cleared"])
        #for as long as the episode is happening, read data from mwm. For some reason, mwm closes too late. Need early stopping
        #that's why I subtract .4 seconds from the end_time here
        end_time = (time.time() + episode_duration - .4) - .15
        print("\tCommencing data acquisition at", time.time())
        #record data for length of episode
        while time.time() < end_time:
            data_point = mwm_reader.readNextDataPoint()
            if(data_point.__class__ is RawDataPoint):
                episode_data.append(data_point._readRawValue())
                episode_reading_times.append(time.time())
                if (display_freq):
                    current_time = time.time()
                    record_times.append(current_time)
                    counter += 1
                    if (counter == 240):
                        l = np.average(np.diff(record_times))
                        print("\t\tmwm:", int(1 / l), "Hz")
                        record_times = []
                        counter = 0
        print("\tEnding data acquisition at", time.time())

        correct_index = 1024 #this value represents the index at which Pacman initially moves
        #receives the time at which pacman makes his move from the environment
        action_time = mwm_pipeend.recv()[0]
        #finds the index closest in time to pacman's move time
        action_index = episode_reading_times.index(min(episode_reading_times, key=lambda x: abs(x - action_time)))

        #adds dummy data to front of episode_data or removes offset number of readings so that the critical points are aligned
        offset = action_index - correct_index
        if(offset < 0):
            dummy = [0]*abs(offset)
            episode_data = dummy + episode_data
        elif(offset > 0):
            episode_data = episode_data[offset:]

        to_trim = int(len(episode_data) - trim_size)
        print("\t\ttrimming:", to_trim, "readings")
        episode_data = episode_data[:int(trim_size)]
        episode_data_df = pd.DataFrame().append([episode_data])
        episode_data_df['episode_type'] = np.array([epi_type])
        episode_data_df['episode_num'] = np.array([current_episode])
        episode_data_df['action_index'] = np.array([action_index])
        episode_data_df['trimmed_data'] = np.array([to_trim])
        episode_data_df['win'] = np.array([epi_type in [0,3]])

        print("\tpredicting...")
        prediction = predict(episode_data_df)
        mwm_pipeend.send(prediction)
        print("prediction sent")



        #predict if episode data df is corrupt
        if(action_index < 950 or action_index > 1100):
            episode_data_df['is_corrupt'] = np.array([1])
        elif(to_trim > 100 or to_trim < 0):
            episode_data_df['is_corrupt'] = np.array([1])
        else:
            episode_data_df['is_corrupt'] = np.array([0])

        # append to trial data df
        trial_data_df = pd.concat([trial_data_df, episode_data_df])
        print("Episode finished. \n\n")
        #idle until animation sends finished animation signal
        # while(not mwm_pipeend.recv()):
        #     mwm_reader.readNextDataPoint()
        #     if (display_freq):
        #         current_time = time.time()
        #         record_times.append(current_time)
        #         counter += 1
        #         if (counter == 240):
        #             l = np.average(np.diff(record_times))
        #             print("\t\tmwm:", 1 / l, "Hz")
        #             record_times = []
        #             counter = 0
        current_episode += 1
    #close and delete mwm
    mwm_reader.end()
    del mwm_reader
    #when the episode is finished, the data gathered by the recorder device is accessed and placed in the queue so the parent process can obtain it
    queue.put(trial_data_df)

#display data so far
def display(data_frame):
    df = data_frame.drop(["episode_num", "trial_type", "trial_num", "action_index", "trimmed_data","is_corrupt"], axis = 1)

    df_win = df.loc[df['episode_type'].isin([0,3])].drop(["episode_type"], axis=1)
    df_loss = df.loc[df['episode_type'].isin([1,2])].drop(["episode_type"], axis=1)
    df_both = df.drop(["episode_type"], axis=1)

    ave_win = df_win.mean()
    ave_loss = df_loss.mean()
    ave_both = df_both.mean()

    sns.set(style="darkgrid")

    plt.plot(ave_win)
    plt.plot(ave_loss)
    plt.plot(ave_both)

    plt.xlabel("reading number")
    plt.ylabel("milli-volts")
    plt.rcParams["figure.figsize"] = [30, 10]
    plt.show()

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
    # display(trial_results)

#obtain most recent trial number
def get_trial_num():
    try:
        dff = pd.read_pickle(data_file_path).tail(n=1).reset_index()
        next_trial_num = dff.at[0, 'trial_num'] + 1
        return next_trial_num
    except FileNotFoundError:
        return 1

eeg_length = -1# make eeg_length global

# shallow convnet model
#params is a dict of the hyperparameters for the model
def this_cnn(features, labels, mode, params):
    global eeg_length

    # reshape input to account for minibatch size. Also, the number of channels (1 in this case) is specified.
    x = tf.reshape(features["x"], [-1, eeg_length, 1])

    # temporal convolution
    with tf.name_scope("conv1_pool1"):
        x = tf.layers.conv1d(
            inputs=x,
            filters=params["conv1_kernel_num"],
            kernel_size=params["conv1_filter_size"],
            strides=params["conv1_stride"],
            padding='valid',
            use_bias=True,
            activation=None,
            name="conv1")
        # with some luck, conv1 will have shape [-1,289,24] ie [batch_size, readings, num_kernels]
        # pooling layer
        x = tf.layers.max_pooling1d(
            inputs=x,
            pool_size=params["pool1_size"],
            strides=params["pool1_stride"],
            name="pool1")

    #only perform these ops if two layers of convolutions are specified
    if(params["use_two_cv"]):#60/100   .4(.4) + .6(.6) =
        with tf.name_scope("conv2_pool2"):
            x = tf.layers.conv1d(
                inputs=x,
                filters=params["conv2_kernel_num"],
                kernel_size=params["conv2_filter_size"],
                strides=params["conv2_stride"],
                padding='valid',
                use_bias=True,
                activation=None,
                name="conv2")
            # pooling layer
            x = tf.layers.max_pooling1d(
                inputs=x,
                pool_size=params["pool2_size"],
                strides=params["pool2_stride"],
                name="pool2")

    #flatten the data to pass through the dense layers
    x_shape = x.shape.as_list()
    x = tf.reshape(x, [-1, x_shape[1] * x_shape[2]])

    if(params["num_fc"] >= 1):
        with tf.name_scope("dense1"):
            x = tf.layers.dense(
                inputs=x,
                units=params["dense1_neurons"],
                activation=params["activations"],
                name="dense1")

    #only perform this if two fully connected layers are specified
    if(params["num_fc"] >= 2):
        with tf.name_scope("dense2"):
            x = tf.layers.dense(
                inputs=x,
                units=params["dense2_neurons"],
                activation=params["activations"],
                name="dense2")

    # rate = .4 => 40% of the output from the dense layers will be randomly dropped during training
    # training checks to see if the CNN is in training mode. if so, the dropout proceeds.
    with tf.name_scope("dropout"):
        x = tf.layers.dropout(
            inputs=x,
            rate=params['dropout_rate'],
            training=mode == tf.estimator.ModeKeys.TRAIN,
            name="dropout")

    # logits output has shape [batch_size, 2]
    with tf.name_scope("logits"):
        logits = tf.layers.dense(
            inputs=x,
            units=2,
            name="logits")

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # if in TRAIN or EVAL mode, calculate loss and backpropagate
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # record metrics for display in tensorboard
    with tf.name_scope("metrics"):
        precision = tf.metrics.precision(
            labels=labels,
            predictions=predictions["classes"])

        recall = tf.metrics.recall(
            labels=labels,
            predictions=predictions["classes"])
        tf.summary.scalar('train_cross_entropy', loss)

        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"])
        tf.summary.scalar('train_accuracy', accuracy[1])

        tf.summary.scalar('train_precision', precision[1])

        tf.summary.scalar('train_recall', recall[1])

        predictions_float = tf.cast(predictions['classes'], dtype=tf.float32)
        win_prediction_freq = tf.reshape(tf.reduce_mean(predictions_float), shape=()) #convert to scalar
        tf.summary.scalar('win_prediction_freq', win_prediction_freq)

        tf.summary.histogram('train_prediction_distributions', predictions['probabilities'])

        # prediction_distribution = np.mean(tf.Session().run(predictions['probabilities']), axis=0)
        # print(predictions['probabilities'])
        # predict_win = prediction_distribution[1]
        # predict_loss = prediction_distribution[0]

        # tf.summary.scalar('predict_win', predict_win)
        # tf.summary.scalar('predict_loss', predict_loss)

    # if in training mode, calculate loss and step to take
    if mode == tf.estimator.ModeKeys.TRAIN:
        if(params["optimizer"] == "GD"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learn_rate"])
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=params["learn_rate"])
        #i think tf.train.get_global_step() is self explanatory here
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # if in eval mode, eval!
    eval_metric_ops = {"eval_accuracy": accuracy,
                       "eval_precision": precision,
                       "eval_recall": recall}
                       # "predict_win": predict_win,
                       # "predict_loss": predict_loss}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def predict(episode_data):
    global eeg_length

    labels = ["episode_type", "trial_type", "trial_num",
              "episode_num", "is_corrupt", "action_index",
              "trimmed_data", "left", "subject_num", "win"]

    episode_data_x = episode_data.drop(labels, axis=1)

    eeg_length = episode_data_x.count(axis=1).values[0]

    hyp_params = {"optimizer":"Adam",
                  "use_f1_as_loss":False,
                  "learn_rate":.00005,
                  "batch_size":25,
                  "steps":100,
                  "use_two_cv":False,
                  "num_fc":0,
                  "conv1_filter_size":25,
                  "conv1_kernel_num":10,
                  "conv1_stride":3,
                  "pool1_size":20,
                  "pool1_stride":3,
                  "conv2_filter_size":15,
                  "conv2_kernel_num":80,
                  "conv2_stride":3,
                  "pool2_size":5,
                  "pool2_stride":2,
                  "dense1_neurons":50,
                  "dense2_neurons":200,
                  "activations":tf.nn.relu,
                  "dropout_rate":.4}

    model_dir = "intrasubject_final/sbj_num_6_cvf_25_dr_0.4_lr_1e-05_cvk_10_cvs_3_pos_20_bs_25"
    classifier = tf.estimator.Estimator(
        params = hyp_params,
        model_fn = this_cnn,
        model_dir = model_dir)


    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": episode_data_x},
        num_epochs=1,
        shuffle=False)

    prediction = classifier.predict(input_fn=predict_input_fn)

    return prediction[0]

#this idiom is necessary. If objects or variables are instantiated outside of this area, things get weird.
if __name__ == '__main__':
    save_data = True
    # the results from this trial are appended to this data frame
    trial_results = pd.DataFrame()
    # display the frequencies that each process is running at
    display_freq = True
    current_score = 0
    cur_epi = 0
    trial_type = -1
    # 1 if the environment is controlled. 0 if completely random.
    control = 1
    data_file_path = 'data/data__11*.pkl'
    # 3 episodes take 27sec. 20 episodes take roughly 3min 10sec.
    num_episodes = 30

    while not trial_type in [0,1]:
        subject_num = int(input("ENTER SUBJECT_NUM: "))
        trial_type = int(input("ENTER TRIAL TYPE: "))
        deviate = int(input("ENTER DEV: "))
    print("\n\n\n")
    trial_start_time = time.time()

    if(deviate == -100):
        deviate = randint(-2,2)


    # the amount of time the participant has to notice the initial state of the environment. Happens before Pacman indicates which direction he'll take
    hangtime = 2  # seconds
    # create the environment with the probabilities determining each epi_type
    env = PacmanEnv(num_episodes=num_episodes, reward=100, punishment=100, scale=10, move_left=.5, is_beneficial=.5,
                    update_freq=20, hangtime=hangtime, speed=7, deviate=deviate, win_value=500)
    # return the calculated episode duration from the environment
    episode_duration = env.episode_duration
    # sequence = env.sequence
    # print(sequence)


    #for some reason the fork start method doesn't work. Spawn must be used. Also, it won't switch to spawn unless force is set to true
    multiprocessing.set_start_method('spawn', force=True)


    #this queue is used to send data back to this main, parent process from the data gathering function/process
    data_queue = multiprocessing.Queue()
    # score_queue = multiprocessing.Queue()
    #this pipe is used to sync the data acquisition process and the animation display process.
    mwm_pipeend, ani_pipeend = multiprocessing.Pipe()

    #both of the processes are created, started, and then join (wait for the other one to finish)
    p1 = multiprocessing.Process(target=read, args=(data_queue, mwm_pipeend, display_freq, episode_duration, num_episodes))
    p2 = multiprocessing.Process(target=env.simulate_multi_epi, args=(ani_pipeend, display_freq, control))


    p1.start()
    p2.start()

    #gets the episode data from read()
    trial_results = pd.concat([trial_results, data_queue.get()])

    p1.join()
    p2.join()

    p1.terminate()
    p2.terminate()

    trial_end_time = time.time()

    trial_duration = trial_end_time - trial_start_time

    print("TOTAL TRIAL TIME:",int(trial_duration/60),"min", int(trial_duration%60),"sec\nSECONDS PER EPISODE:",round((trial_duration/num_episodes),1))
    cur_epi += 1

    trial_results['episode_num'] = np.array(list(range(num_episodes)))

    trial_num = get_trial_num()
    trial_results['trial_type'] = pd.Series(np.array([trial_type]*num_episodes))
    trial_results['trial_num'] = pd.Series(np.array([trial_num]*num_episodes))
    trial_results['subject_num'] = pd.Series(np.array([subject_num]*num_episodes))

    trial_results.reset_index(inplace=True)
    trial_results.index.name = 'index'
    del trial_results['index']
    #appends this trial's data to saved data and resaves it
    if(save_data):
        update_data_base(trial_results)
    else:
        print("UNSAVED TRIAL RESULTS:", trial_results)









