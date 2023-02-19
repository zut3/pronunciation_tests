import librosa
import soundfile as sf
import numpy as np
from hashlib import sha1

class AudioSpeech:
  def __init__(self, signal, sample_rate: int):
    self.signal = signal
    self.sample_rate = sample_rate

  @staticmethod
  def load_audio(filename: str):
    audio, sr = librosa.load(filename)
    audio = audio[~np.isnan(audio)]

    return AudioSpeech(audio, sr)  

  def write(self, filename, format_=None):
      sf.write(filename, self.signal, self.sample_rate, format=format_, subtype='PCM_24')

  def hash(self):
      return sha1(bytearray(self.signal)).hexdigest()
