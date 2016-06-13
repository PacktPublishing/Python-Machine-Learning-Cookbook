import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# File where the output will be saved
output_file = 'output_generated.wav'

# Specify audio parameters
duration = 3  # seconds
sampling_freq = 44100  # Hz
tone_freq = 587
min_val = -2 * np.pi
max_val = 2 * np.pi

# Generate audio
t = np.linspace(min_val, max_val, duration * sampling_freq)
audio = np.sin(2 * np.pi * tone_freq * t)

# Add some noise
noise = 0.4 * np.random.rand(duration * sampling_freq)
audio += noise

# Scale it to 16-bit integer values
scaling_factor = pow(2,15) - 1
audio_normalized = audio / np.max(np.abs(audio))
audio_scaled = np.int16(audio_normalized * scaling_factor)

# Write to output file
write(output_file, sampling_freq, audio_scaled)

# Extract first 100 values for plotting
audio = audio[:100]

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
