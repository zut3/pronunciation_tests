import tensorflow as tf
from tensorflow_addons.losses import contrastive_loss
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from services.shared import AudioSpeech
import numpy as np

def get_mel(audio):
    spec = tf.signal.stft(audio, frame_length=255, frame_step=128)
    spec = tf.abs(spec)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = spec.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, 16000, lower_edge_hertz,
      upper_edge_hertz)

    mel_spectrograms = tf.tensordot(
      spec, linear_to_mel_weight_matrix, 1)

    mel_spectrograms.set_shape(spec.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
      log_mel_spectrograms)

    return mfccs

def load_audio(audio: AudioSpeech, target_shape):
  audio = tf.constant(audio.signal)
  audio = tf.cast(audio, tf.float32)
  print(audio.shape)

  spec = get_mel(audio)
  spec = tf.expand_dims(spec, axis=-1)
  spec = tf.image.grayscale_to_rgb(spec)
  spec = tf.image.resize(spec, target_shape)

  return spec

def scale(X, min, max):
      std = (X - min) / (max - min)
      std *= max - min
      return (std + min).astype(np.int32)
