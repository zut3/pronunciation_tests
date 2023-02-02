from services.shared import AudioSpeech
from uuid import uuid4
import os

def write_to_fp():
    filename = str(uuid4()) + '.wav'    
    path = './storage/' + filename
    f = open(path, 'wb')
    return filename, f
 

def write(audio: AudioSpeech):
    uid, fp = write_to_fp()
    audio.write('./storage/' + uid)
    fp.close()

    return uid

   
def get(filename: str):
    path = './storage/' + filename
    if not os.path.exists(path):
        raise ValueError('file not exist')
    return AudioSpeech.load_audio(path)
