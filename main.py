# %%
"""
# SPEECH RECOGNITION AND TEXT TRANSFORMATION
"""

# %%
"""
## IMPORTS
"""

# %%
import pandas as pd
import numpy as np
import speech_recognition as sr
import time
from os import path
from os import listdir
from os.path import isfile, join
from os import walk
from scipy import signal
import wave
import librosa 
from pydub import AudioSegment
from pydub.effects import normalize
import soundfile as sf
import noisereduce as nr 
from scipy.io import wavfile
from pocketsphinx import AudioFile, get_model_path

# %%
"""
## CONVERSION OF MP3 INTO WAV
"""

# %%
f = []
for (dirpath, dirnames, filenames) in walk('data'):
    f.extend(filenames)
    break
f.remove('.DS_Store')   #FOR mac's

# %%
i=0;
for x in f:
    input_file = 'data'+'/'+f[i]
    i=i+1;
    output_file = "new_data/file"+i.__str__()+".wav"
    sound = AudioSegment.from_mp3(input_file)
    sound.export(output_file, format="wav")

# %%
datax = []
for (dirpath, dirnames, filenames) in walk('new_data'):
    datax.extend(filenames)
    break
datax.remove('.DS_Store')   #FOR mac's

# %%
"""
### Setting Up the Google Speech Recognizer
"""

# %%
recognizer = sr.Recognizer()
recognizer.energy_threshold = 3000

# %%
"""
### Importing Audio files from the new_data
"""

# %%
data = []
for x in datax:
    audio_ex = sr.AudioFile('new_data/'+x)
    with audio_ex as source:
        audiodata = recognizer.record(audio_ex)
    data.append(audiodata)

# %%
# text = recognizer.recognize_google(
#   audio_data = , 
#   language='en-US')

# %%
"""
## PRE_PROCESSING OF AUDIO FILES
"""

# %%
"""
### Utility Functions for Denoising
"""

# %%
fs = 200 # sample rate in Hz
tau = 10 # time window in seconds for mean computation

# Boxcar/SMA filter params
N = int(tau*fs) # Number of samples corresponding to tau
h_boxcar = np.ones(N)/N  # impulse response of boxcar filter

# EMA filter params
alpha = np.exp(-1/(fs*tau))
a_ema = [1, -alpha] # Denominator
b_ema =  1 - alpha # Numerator

w, h = signal.freqz(h_boxcar)

w, h = signal.freqz(b_ema, a_ema)

# Using scipy's stft/istft function; see scipy's stft source code for details
def audio_to_frames(y: np.array, m, hop_size, fs) -> np.array:
    """Convert y[n] into a matrix of frames Y_m(w) where each row is a time slice"""   
    _, _, Zxx = signal.stft(y, fs=fs, nperseg=m, noverlap=hop_size, nfft=m*8)
    return Zxx.T

def frames_to_audio(Y: np.array, m, hop_size, fs) -> np.array:
    """Convert Y_m(w) matrix of frames into a 1D signal y[n] using Overlap-Add"""
    _, xrec = signal.istft(Y.T, fs=fs, nperseg=m, noverlap=hop_size, nfft=m*8)
    return xrec

def spec_oversubtract(Y, est_Pn):
    # Compute the alpha values for each frame
    snr = 10*np.log10(sum(abs(Y)**2)/sum(est_Pn))
    alpha = []
    for gamma in snr:  # Implement the purple curve above
        if gamma >= -5 and gamma <= 20:
            a = -6.25*gamma/25 + 6
            alpha.append(a)
        elif gamma > 20:
            alpha.append(1)
        else:
            alpha.append(7.25)
    beta = 0.002
    est_powX = np.maximum(abs(Y)**2 - alpha * est_Pn, beta * est_Pn) # Oversubtraction & spectral flooring
    est_phaseX = np.angle(Y)
    est_Sx = np.sqrt(est_powX) * np.exp(1j*est_phaseX)
    return est_Sx
def noise_estimation_snr(Y: np.array) -> (np.array, np.array):
    """Estimates the magnitude and power spectrum of the noise for each frame"""
    
    # Prepare the output variables
    est_Mn = np.zeros(Y.shape)
    est_Pn = np.zeros(Y.shape)
    
    N = 10 # Number of frames to use for estimating a-posteriori SNR
    
    # Iterate through each frame and estimate noise
    for m in range(Y.shape[0]):
        if m < N:
            # Use noisy spectra for first 10 iterations
            est_Mn[m] = abs(Y[m])
            est_Pn[m] = est_Mn[m] ** 2
        else:
            a = 25
            # A-posteriori SNR            
            gammak = (abs(Y[m])**2)/np.mean(abs(Y[m-N:m])**2, axis=0) 
            alpha = 1/(1+np.exp(-a*(gammak-1.5)))
            est_Mn[m] = alpha * abs(est_Mn[m-1]) + (1-alpha) * abs(Y[m])
            est_Pn[m] = alpha * (abs(est_Mn[m-1])**2) + (1-alpha) * (abs(Y[m])**2)
            
    return est_Mn, est_Pn
        


# %%
"""
### Cleaning of Audio Files
"""

# %%
def denoise(speech_file,noise_file):
    x, fs = librosa.load(speech_file, sr=16000)
    n, fs_noise = librosa.load(noise_file, sr=16000)

    noise_gain = 0.7
    n = noise_gain * n[:len(x)]

    # Padding 
    for i in range(len(x)):
        if(i>=len(n)):
            n = np.append(n, [0])

    y = x + n

    # Compute the SNR for signal x to noise n
    y_snr = 10*np.log10(np.sum(x**2)/np.sum(n**2))

    win_t = 30e-3 # window size in seconds
    win_s = round(fs*win_t) # window size in samples
    hop_size = win_s//2

    Y = audio_to_frames(y, win_s, hop_size, fs)
    est_Mn, est_Pn = noise_estimation_snr(Y)
    est_Sx_oversub = spec_oversubtract(Y, est_Pn)
    x_hat_oversub = frames_to_audio(est_Sx_oversub, win_s, hop_size, fs)[:len(x)]
    print("The File is processed",x_hat_oversub,fs)
    return x_hat_oversub,fs

# %%
### Denoising all the files from the new_data
noise_file = 'noise.wav'
i=0
for file_name in datax:
    denoised,fs=denoise("new_data/"+file_name,noise_file)
    i=i+1;
    sf.write('processed_data/file'+i.__str__()+'.wav', denoised, fs)
    rate, data = wavfile.read('processed_data/file'+i.__str__()+'.wav') 
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    sf.write('processed_data/file'+i.__str__()+'.wav', denoised, fs)

# %%
"""
## APPLYING CLASS LABELS
"""

# %%
 df = pd.read_csv('data.csv',            
            names=['audio_files', 'no_of_channels', 'frame_rate', 'sample_width','max_amplitude','no_of_ms'])    

# %%
f = []
for (dirpath, dirnames, filenames) in walk('processed_data'):
    f.extend(filenames)
    break

# %%
channels =[]
framerates =[]
samplewidths = []
framewidths =[]
Ampmax = []
framewidths = []
size =[]
f.remove('.DS_Store')
for i in f:
    x = AudioSegment.from_file('processed_data/'+i,format='wav')
    channels.append(x.channels)
    framerates.append(x.frame_rate)
    samplewidths.append(x.sample_width)
    framewidths.append(x.frame_width)
    Ampmax.append(x.max)
    size.append(len(x))

# %%
df['audio_files'] = f
df['no_of_channels'] = channels
df['frame_rate'] = framerates
df['sample_width'] = samplewidths
df['max_amplitude']= Ampmax
df['no_of_ms'] = size
df['frame_width'] = framewidths

# %%
for i in f:
    x = AudioSegment.from_file('processed_data/'+i,format='wav')
    x+160
    normalize(x)
    x.export('pre_processed_data/'+i,format='wav')

# %%
"""
## APPLYING THE CMU TRANSCRIPTION
"""

# %%
transcriptions = []
for i in f:
    for phrase in AudioFile('processed_data/'+i): 
        transcriptions.append(str(phrase))
    

# %%
transcriptions

# %%
df['cmu_transcriptions'] = transcriptions

# %%
df.to_csv('data.csv', encoding='utf-8')

# %%
"""
### APPLYING THE [PAUSE] IN THE TRANSCRIPTION
"""

# %%
def freq(file, start_time, end_time):
    sample_rate, data = wavfile.read(file)
    start_point = int(sample_rate * start_time / 1000)
    end_point = int(sample_rate * end_time / 1000)
    length = (end_time - start_time) / 1000
    counter = 0
    for i in range(start_point, end_point):
        if data[i] < 0 and data[i+1] > 0:
            counter += 1
    return counter/length    

# %%
freq("stereo_file1.wav", 0 ,41400)

# %%
