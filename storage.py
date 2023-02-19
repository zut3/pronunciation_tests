from services.shared import AudioSpeech
from uuid import uuid4
import os

def write_to_fp(uid=None):
    if not uid:
        uid = str(uuid4())

    filename = uid + '.wav'     
    path = './storage/' + filename
    f = open(path, 'wb')
    return uid, f
 

def write(audio: AudioSpeech):
    h = audio.hash()
    if os.path.exists('./storage' + h + '.wav'):
        return uid

    uid, fp = write_to_fp(h)
    audio.write('./storage/' + uid + '.wav')
    fp.close()

    return uid

   
def get(filename: str):
    path = './storage/' + filename + '.wav'
    if not os.path.exists(path):
        raise ValueError('file not exist')
    return AudioSpeech.load_audio(path)
