from services.shared import AudioSpeech
from uuid import uuid4
import os

def write(audio: AudioSpeech):
    filename = str(uuid4()) + '.wav'
    path = './storage/' + filename
    with open(path, 'x'):
        audio.write(path)
    return filename

def get(filename: str):
    path = './storage/' + filename
    if not os.path.exists(path):
        raise ValueError('file not exist')
    return AudioSpeech.load_audio(path)
