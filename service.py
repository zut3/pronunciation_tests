import numpy as np
import nemo
import nemo.collections.asr as nemo_asr
import regex
import re
import librosa
import soundfile as sf
import utils
import ctc_segmentation as cs


MODEL = "QuartzNet15x5Base-En"

class AudioSpeech:
  def __init__(self, signal: np.ndarray, sample_rate: int):
    self.signal = signal
    self.sample_rate = sample_rate

  @staticmethod
  def load_audio(filename: str):
    audio, sr = librosa.load(filename)
    if sr == 16000:
      return AudioSpeech(audio, sr)

    audio = librosa.resample(audio, sr, 16000)
    return AudioSpeech(audio, 16000)
  
  def write(self, filename):
      sf.write(filename, self.signal, self.sample_rate, subtype='PCM_24')

def clip(start: float, end: float, audio: AudioSpeech):
  start = int(start * audio.sample_rate)
  end = int(end * audio.sample_rate)
  return AudioSpeech(audio.signal[start:end], audio.sample_rate)

def split_text(transcript, vocabulary):
  voc_symbols = list(vocabulary)
  voc_symbols += [x.upper() for x in voc_symbols]
  voc_symbols = set(voc_symbols)
  if " " in voc_symbols:
      voc_symbols.remove(" ")

  transcript = re.sub(r"([\.\?\!])([\"\'])", r"\g<2>\g<1> ", transcript)
  transcript = re.sub(r" +", " ", transcript)

  split_pattern = f"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!|\.”|\?”\!”)\s"

  matches = re.findall(r'[a-z]\.\s[a-z]\.', transcript)
  for m in matches:
      transcript = transcript.replace(m, m.replace('. ', '.'))

  with_quotes = re.finditer(r'“[A-Za-z ?]+.*?”', transcript)
  sentences = []
  last_idx = 0
  for m in with_quotes:
      match = m.group()
      match_idx = m.start()
      if last_idx < match_idx:
          sentences.append(transcript[last_idx:match_idx])
      sentences.append(match)
      last_idx = m.end()
  sentences.append(transcript[last_idx:])
  sentences = [s.strip() for s in sentences if s.strip()]

  new_sentences = []
  for sent in sentences:
    new_sentences.extend(regex.split(split_pattern, sent))
  
  sentences = [s.strip() for s in new_sentences if s.strip()]
  sentences = [s.strip() for s in sentences if len(voc_symbols.intersection(set(s.lower()))) > 0 and s.strip()]

  
  return sentences

def _get_segments(log_probs: np.ndarray, vocabulary, index_duration: float, text, window_size: int = 8000):
  config = cs.CtcSegmentationParameters()
  config.char_list = vocabulary
  config.min_window_size = window_size
  config.index_duration = index_duration
  config.excluded_characters = ".,-?!:»«;'›‹()"
  config.blank = vocabulary.index(" ")
  ground_truth_mat, utt_begin_indices = cs.prepare_text(config, text)
  config.blank = 0

  timings, char_probs, char_list = cs.ctc_segmentation(config, log_probs, ground_truth_mat)
  segments = utils.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, text, char_list)

  return segments


def segment(audio_path, text: str, window_size: int = 8000):
  asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(MODEL, strict=False)
  vocabulary = ["ε"] + list(asr_model.cfg.decoder.vocabulary)
  sentences = split_text(text, vocabulary)

  audio = AudioSpeech.load_audio(audio_path)
    
  log_probs = asr_model.transcribe(paths2audio_files=[audio_path,], batch_size=1, logprobs=True)[0]
  blank_col = log_probs[:, -1].reshape((log_probs.shape[0], 1))
  log_probs = np.concatenate((blank_col, log_probs[:, :-1]), axis=1)
    
  index_duration = len(audio.signal) / log_probs.shape[0] / audio.sample_rate

  segments = _get_segments(log_probs=log_probs, vocabulary=vocabulary, index_duration=index_duration, text=sentences, window_size=window_size)
  return segments

