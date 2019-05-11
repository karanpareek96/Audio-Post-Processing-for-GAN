# Audio Post Processing Function

import numpy as np
import librosa
from scipy import signal

# Filter Params

# Bandpass to filter the noise
nyq = 0.5 * 16000
low_value = 1000
high_value = 7000
low_cutoff = low_value / nyq
high_cutoff = high_value / nyq

# Import IRs

'''
***Note:*** Reset path to IRs

List of Mic IR
1. IR_Crystal.wav
2. IR_Lomo52A5M.wav
3. IR_OktavaMD57.wav
4. IR_AKGD12.wav
5. IR_GaumontKalee.wav
6. IR_STC4035.wav
7. IR_MelodiumRM6.wav

List of Speaker IR
1. IR_ClestionBD300.wav
2. IR_CelestionV30E606.wav
3. IR_JensenCab.wav
4. IR_Unknown.wav

List of Room IR
1. BRIR.wav
'''

# Speaker IR
ir_speaker_dir = '/beegfs/kp2218/test_runs/conv_test/data/audio/ir_speaker/IR_ClestionBD300.wav'
ir_speaker, fs_speaker = librosa.load(ir_speaker_dir, sr=16000, mono=True)

# Microphone IR
ir_mic_dir = '/beegfs/kp2218/test_runs/conv_test/data/audio/ir_mic/IR_GaumontKalee.wav'
ir_mic, fs_mic = librosa.load(ir_mic_dir, sr=16000, mono=True)

# Room IR
ir_room_dir = '/beegfs/kp2218/test_runs/conv_test/data/audio/ir_room/BRIR.wav'
ir_room, fs_room = librosa.load(ir_room_dir, sr=16000, mono=True)

'''
Post Processing function (Numpy)
    Input: Audio tensor (input)
    Output: Processed audio tensor (audio_out)
'''


def audio_post_processing(input_file):

    # Convolve with Speaker IR
    speaker_out = np.convolve(input_file, ir_speaker, mode='same')

    # Add noise
    noise_param = np.random.normal(0, 5e-3, size=speaker_out.shape)

    b, a = signal.cheby1(15, 4, [low_cutoff, high_cutoff], btype='bandpass')
    noise_param = signal.filtfilt(b, a, noise_param)

    speaker_out += noise_param

    # Convolve with Room IR
    room_out = np.convolve(speaker_out, ir_room, mode='same')

    # Convolve with Mic IR
    audio_out = np.convolve(room_out, ir_mic, mode='same')

    return audio_out
