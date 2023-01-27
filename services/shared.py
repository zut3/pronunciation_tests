import librosa
import soundfile as sf
import audioread.ffdec


class AudioSpeech:
  def __init__(self, signal, sample_rate: int):
    self.signal = signal
    self.sample_rate = sample_rate

  @staticmethod
  def load_audio(filename: str):
    filename = audioread.ffdec.FFmpegAudioFile(filename)
    audio, sr = librosa.load(filename)
  
    if sr == 16000:
      return AudioSpeech(audio, sr)

    audio = librosa.resample(audio, sr, 16000)
    return AudioSpeech(audio, 16000)

  def write(self, filename, format_=None):
      sf.write(filename, self.signal, self.sample_rate, format=format_, subtype='PCM_24')
