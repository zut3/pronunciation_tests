from fastapi import FastAPI, UploadFile, HTTPException, Body
from fastapi.responses import FileResponse
from services.shared import AudioSpeech
from services.segmentation import segment, clip
from services.sim_model.service import sim
import storage 
import os
from models import SimRequest

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
    path = 'storage/' + uid
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse('storage/' + uid)

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

