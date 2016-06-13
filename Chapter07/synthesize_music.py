import json
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# Synthesize tone
def synthesizer(freq, duration, amp=1.0, sampling_freq=44100):
    # Build the time axis
    t = np.linspace(0, duration, duration * sampling_freq)

    # Construct the audio signal
    audio = amp * np.sin(2 * np.pi * freq * t)

    return audio.astype(np.int16) 

if __name__=='__main__':
    # Input file containing note to frequency mapping
    tone_map_file = 'tone_freq_map.json'
    
    # Read the frequency map
    with open(tone_map_file, 'r') as f:
        tone_freq_map = json.loads(f.read())
        
    # Set input parameters to generate 'G' tone
    input_tone = 'G'
    duration = 2     # seconds
    amplitude = 10000
    sampling_freq = 44100    # Hz

    # Generate the tone
    synthesized_tone = synthesizer(tone_freq_map[input_tone], duration, amplitude, sampling_freq)

    # Write to the output file
    write('output_tone.wav', sampling_freq, synthesized_tone)

    # Tone-duration sequence
    tone_seq = [('D', 0.3), ('G', 0.6), ('C', 0.5), ('A', 0.3), ('Asharp', 0.7)]

    # Construct the audio signal based on the chord sequence
    output = np.array([])
    for item in tone_seq:
        input_tone = item[0]
        duration = item[1]
        synthesized_tone = synthesizer(tone_freq_map[input_tone], duration, amplitude, sampling_freq)
        output = np.append(output, synthesized_tone, axis=0)

    # Write to the output file
    write('output_tone_seq.wav', sampling_freq, output)

