from python_speech_features import mfcc
import numpy as np
import scipy.io.wavfile as wav
import os
import struct
import pyaudio
from matplotlib import pyplot


# WIDTH = 2           # bytes per sample
# CHANNELS = 1        # mono
# RATE = 8000         # frames per second
# BLOCKLEN = 1024     # block length in samples
# DURATION = 10       # Duration in seconds

# K = int( DURATION * RATE / BLOCKLEN )   # Number of blocks

# print('Block length: %d' % BLOCKLEN)
# print('Number of blocks to read: %d' % K)
# print('Duration of block in milliseconds: %.1f' % (1000.0 * BLOCKLEN/RATE))

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

# p = pyaudio.PyAudio()
# PA_FORMAT = p.get_format_from_width(WIDTH)
# stream = p.open(
#     format = PA_FORMAT,
#     channels = CHANNELS,
#     rate = RATE,
#     input = True,
#     output = False)

# (rate, sig) = wav.read('author.wav')
# print(type(sig))
# for i in range(K):

#     # Read audio input stream
#     input_bytes = stream.read(BLOCKLEN)

#     signal_block = struct.unpack('h' * BLOCKLEN, input_bytes)  # Convert
#     signal_block = np.array(signal_block)
#     #output = np.mean(np.mean(mfcc(signal_block, RATE), axis=0))
#     output = mfcc(signal_block, RATE)
#     print(np.shape(output))
#     # print(output)

#     # g1.set_ydata(signal_block/1000)   # Update y-data of plot
#     #g2.set_ydata(output)
#     # pyplot.pause(0.0001)



ds=np.empty((0,13))
i=-1
for filename in os.listdir('/Users/archit/Documents/Notes/NYU/Sem3/DSPLAB/project/dataset'):
    print(filename)
    (rate, sig)= wav.read('/Users/archit/Documents/Notes/NYU/Sem3/DSPLAB/project/dataset'+'/'+filename)
    mfcc_feat = mfcc(sig, rate) 
    avg=np.mean(mfcc_feat, axis=0)
    print(np.shape(avg))
    np.reshape(avg, (1,13))
    # avg=np.hstack((avg,i))
    #print(avg)
    ds=np.append(ds, [avg], axis=0)
    #print(ds.shape)
np.savetxt("speak_reco_DSP_new.csv", ds, delimiter=",")

# stream.stop_stream()
# stream.close()
# p.terminate()

# # pyplot.ioff()           # Turn off interactive mode
# # pyplot.show()           # Keep plot showing at end of program

# # pyplot.close()
# print('* Finished')
