from fastapi import FastAPI, UploadFile, HTTPException, Body
from fastapi.responses import FileResponse
from services.shared import AudioSpeech
from services.segmentation import segment, clip
from services.sim_model.service import sim
import storage 
import os
from models import SimRequest
import numpy as np
import re

app = FastAPI()

@app.post('/seg')
async def seg(file: UploadFile, text: str):
    path = 'files/' + file.filename
    with open(path, 'wb') as f:
        data = await file.read()
        f.write(data)
    file.close()

    audio = AudioSpeech.load_audio(path)
    segments = segment(path, text.strip())

    clipped = []
    for start,end, _ in segments:
        clip_ = clip(start, end, audio)
        clipped += [storage.write(clip_)]

    return {'res': clipped}

@app.get('/storage/{uid}')
async def get_file(uid: str):
    path = 'storage/' + uid + '.wav'
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)

@app.post('/storage/upload')
async def upload_file(file: UploadFile):
    uid, fp = storage.write_to_fp()
    fp.write(await file.read())
    fp.close()
    file.close()

    return uid

@app.post('/simi')
async def test_simi(uid: str = Body(), text: str = Body()):
    audio = storage.get(uid)
    simi = sim(audio, text)

    return {'res': round(simi)}

@app.post('/score')
async def get_score(uid: str = Body(), text: str = Body()):
    path = './storage/' + uid + '.wav'
    speech = storage.get(uid)
    segments = segment(path, text.strip())

    clipped = []
    for start,end, _ in segments:
        clip_ = clip(start, end, speech)
        clipped += [storage.write(clip_)]
    
    sentences = re.findall(r'[\w+ ,:;]+[.?!]', text.strip())
    print(clipped)
    print(sentences)

    res = []
    for i, cl in enumerate(clipped):
        audio = storage.get(cl)
        res.append(sim(audio, sentences[i]))
    
    return {'score': round(np.array(res).mean())}
