from service import AudioSpeech
from uuid import uuid4
import os

def write(audio: AudioSpeech):
    filename = str(uuid4()) + '.wav'
    path = 'storage/' + filename
    with open(path, 'x'):
        audio.write(path)
    return filename

def get(filename: str):
    if not os.path.exists('storage/' + filename):
        raise ValueError('file not exist')
    return AudioSpeech.load_audio(filename)
