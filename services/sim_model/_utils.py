import numpy as np
from ._models import model, processor
from phonemizer import phonemize
from phonemizer.separator import Separator
import torch

def audio2phonem(audio: np.array):
  # tokenize
  input_values = processor(audio, return_tensors="pt").input_values

  # retrieve logits
  with torch.no_grad():
    logits = model(input_values).logits

  # take argmax and decode
  predicted_ids = torch.argmax(logits, dim=-1)
  transcription = processor.batch_decode(predicted_ids)
  return transcription[0]

def text2phonem(text: str):
    return phonemize(
    text,
    language='en-us',
    backend='espeak',
    separator=Separator(phone=" ", word="<s>", syllable=None),
    strip=True,
    njobs=1).replace("<s>", '')


def scale(X, min, max):
    std = (X - min) / (max - min)
    std *= max - min
    return (std + min).astype(np.int32)
