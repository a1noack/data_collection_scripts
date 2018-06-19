import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import sys,csv
sys.path.append("..")
import time
from py35.mindwavemobilepy35.mindwavemobile.MindwaveDataPoints import RawDataPoint, PoorSignalLevelDataPoint
from py35.mindwavemobilepy35.mindwavemobile.MindwaveDataPointReader import MindwaveDataPointReader
style.use('fivethirtyeight')
import _thread

#define plot and axis
fig = plt.figure(figsize=(15, 4))
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('reading number')
ax1.set_ylabel(u'\u03bc'+'V')
ax1.set_title('Raw EEG Readings')
# text = ax1.text('Sampling rate: ')
plt.ion()
plt.show()

#define plot ranges
domain=2500
xs = [x for x in range(0,domain)] #if the raw values are received at 512Hz, then this will be enough to show 20 seconds worth of data
ys = [0]*domain
line1, = ax1.plot(xs, ys, 'm',linewidth=1)

#create csv file for saving output
file = open('raw_eeg.csv','w', newline='')
writer = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

#start connection to mwm
mindwaveDataPointReader = MindwaveDataPointReader()
mindwaveDataPointReader.start()

#the headset starts on the participant's head
on_head = True

def animate(on_head):
    if (mindwaveDataPointReader.isConnected()):
        counter = 0
        record_times = []
        while(on_head == True):
            if(counter == 500):
                l = np.average(np.diff(record_times))
                print("average receive frequency:",1/l,"Hz")
                record_times = []
                counter = 0
            data_point = mindwaveDataPointReader.readNextDataPoint()
            if (data_point.__class__ is RawDataPoint):
                raw_value = data_point._readRawValue()
                current_time = time.time()
                record_times.append(current_time)
                counter += 1
                #write data to csv file
                #writer.writerow([current_time, raw_value])

                #update x and y values for plot. Add new value to end of each and remove element in position 0 from both.
                xs.append(xs.pop(0) + domain + 1)
                ys.pop(0)
                ys.append(raw_value)

                #reset graph domain and range to fit updated values
                plt.xlim(xs[0], xs[domain-1])
                plt.ylim(min(ys), max(ys))

                #update figure and draw
                line1.set_ydata(ys)
                line1.set_xdata(xs)
                fig.canvas.draw()
            elif(data_point.__class__ is PoorSignalLevelDataPoint):
                #check if headset is still on head
                on_head = data_point.headSetHasContactToSkin()
        file.close()
        print("Headset removed from head. Animation closed.")

animate(on_head)