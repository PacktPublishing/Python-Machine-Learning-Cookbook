import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Read the input file
sampling_freq, audio = wavfile.read('input_read.wav')

# Print the params
print '\nShape:', audio.shape
print 'Datatype:', audio.dtype
print 'Duration:', round(audio.shape[0] / float(sampling_freq), 3), 'seconds'

# Normalize the values
audio = audio / (2.**15)

# Extract first 30 values for plotting
audio = audio[:30]

# Build the time axis
x_values = np.arange(0, len(audio), 1) / float(sampling_freq)

# Convert to seconds
x_values *= 1000

# Plotting the chopped audio signal
plt.plot(x_values, audio, color='black')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Audio signal')
plt.show()

