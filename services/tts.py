from .shared import AudioSpeech
from gtts import gTTS
from io import BytesIO
import numpy as np
import storage

def text_to_speech(text: str):
    uid, fp = storage.write_to_fp()

    tts = gTTS(text.strip())
    tts.write_to_fp(fp)

    return uid   
