from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
from service import segment, clip, AudioSpeech
import storage
import os

app = FastAPI()

@app.post('/upload')
async def main(file: UploadFile, text: str):
    path = 'files/' + file.filename
    with open(path, 'wb') as f:
        data = await file.read()
        f.write(data)

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
