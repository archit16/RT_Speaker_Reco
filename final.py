from python_speech_features import mfcc
import numpy as np
import scipy.io.wavfile as wav
import os
import struct
import pyaudio
from matplotlib import pyplot
from joblib import dump, load
from sklearn import svm
from sklearn import preprocessing
import pandas as pd


WIDTH = 2           # bytes per sample
CHANNELS = 1        # mono
RATE = 8000         # frames per second
BLOCKLEN = 1024     # block length in samples
DURATION = 10       # Duration in seconds

K = int( DURATION * RATE / BLOCKLEN )   # Number of blocks

print('Block length: %d' % BLOCKLEN)
print('Number of blocks to read: %d' % K)
print('Duration of block in milliseconds: %.1f' % (1000.0 * BLOCKLEN/RATE))

train_file = pd.read_csv('speak_reco_DSP_new.csv')
test_file = pd.read_csv('speak_reco_DSP_test.csv')
features = ['feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','feat_8','feat_9','feat_10','feat_11','feat_12','feat_13']

X_train = train_file[features].values

# pyplot.ion()           # Turn on interactive mode
# pyplot.figure(1)
# [g1] = pyplot.plot([], [], 'blue')  # Create empty line
# [g2] = pyplot.plot([], [], 'red')

# n = range(0, BLOCKLEN)
# pyplot.xlim(0, BLOCKLEN)         # set x-axis limits
# pyplot.xlabel('Time (n)')
# g1.set_xdata(n)                   # x-data of plot (discrete-time)
# g2.set_xdata(n)

# pyplot.ylim(-10, 10)        # set y-axis limits
SVM_clf = load("speakereco_model.joblib")
p = pyaudio.PyAudio()
PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(
    format = PA_FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input = True,
    output = False)

for i in range(K):

    # Read audio input stream
    input_bytes = stream.read(BLOCKLEN)

    signal_block = struct.unpack('h' * BLOCKLEN, input_bytes)  # Convert
    signal_block = np.array(signal_block)
    output = np.mean(mfcc(signal_block, RATE), axis=0)
    np.reshape(output, (1,13))
    output = [output]
    std_scaler_x = preprocessing.StandardScaler().fit(X_train)
    X_train_std = std_scaler_x.transform(output)
    #print(SVM_clf.predict(output))
    y_pred = SVM_clf.predict_proba(X_train_std)
    if(y_pred[:,0]>0.8):
        print("Speaker -> Deep")
    elif(y_pred[:,1]>0.8):
        print("Speaker -> Archit")
    else:
        print("No speaker detected")


stream.stop_stream()
stream.close()
p.terminate()
#     # g1.set_ydata(signal_block/1000)   # Update y-data of plot
#     #g2.set_ydata(output)
#     # pyplot.pause(0.0001)
