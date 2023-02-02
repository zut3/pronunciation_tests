from ._utils import audio2phonem, text2phonem
from fuzzywuzzy import fuzz
from services.shared import AudioSpeech

def sim(audio: AudioSpeech, text: str):
    p1 = audio2phonem(audio.signal)
    p2 = text2phonem(text)

    score = fuzz.ratio(p1, p2)  
    print(score)

    return 5 * (score / 100)

