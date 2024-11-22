from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import sqlite3
import os
import uuid
import torch
import h5py
from typing import List
import torch.nn as nn


class ECAPA_TDNN(nn.Module):
    def __init__(self, embd_dim=512):
        super().__init__()
        self.embd_dim = embd_dim
        
    def forward(self, x):
        pass

app = FastAPI()

AUDIO_UPLOAD_DIR = "uploads"
DATABASE_PATH = "voice_auth.db"
SIMILARITY_THRESHOLD = 0.70
WEIGHTS_PATH = "audio_feature_extractor_epoch_4.pth"
EMBEDDINGS_PATH = "user_embeddings.h5"

os.makedirs(AUDIO_UPLOAD_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECAPA_TDNN(embd_dim=512)
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.to(device)
model.eval()

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            embeddings_stored BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def preprocess_audio(file_path: str) -> List[np.ndarray]:
    """Preprocess audio file and return processed segments"""
    audio, sample_rate = librosa.load(file_path, sr=None)
    
    frame_length_ms = 25
    frame_shift_ms = 10
    frame_length = int(frame_length_ms * sample_rate / 1000)
    hop_length = int(frame_shift_ms * sample_rate / 1000)
    
    mfccs = librosa.feature.mfcc(
        y=audio, 
        sr=sample_rate, 
        n_mfcc=80, 
        hop_length=hop_length,
        n_fft=frame_length
    )
    mfccs = mfccs.T
    

    processed_segments = []
    for j in range(0, mfccs.shape[0], 200):
        crop = mfccs[j:j + 200]
        if len(crop) < 200:
            break
        means = np.mean(crop, axis=0)
        norm_crop = crop - means
        processed_segments.append(norm_crop)
    
    return processed_segments

def extract_embeddings(processed_segments: List[np.ndarray]) -> List[torch.Tensor]:
    """Extract embeddings using the model"""
    embeddings = []
    with torch.no_grad():
        for segment in processed_segments[:5]: 
            x = torch.tensor(segment[None, :]).to(device)
            embedding = model(x)
            embeddings.append(embedding)
    return embeddings

def save_embeddings(username: str, embeddings: List[torch.Tensor]):
    """Save embeddings to H5 file"""
    with h5py.File(EMBEDDINGS_PATH, "a") as f:
        if username in f:
            del f[username]  
        f.create_dataset(username, data=np.array([emb.cpu().numpy() for emb in embeddings]))

def get_embeddings(username: str) -> List[np.ndarray]:
    """Retrieve embeddings from H5 file"""
    with h5py.File(EMBEDDINGS_PATH, "r") as f:
        if username not in f:
            raise HTTPException(status_code=404, content="User embeddings not found")
        return list(f[username][:])

def compute_similarity(emb1: List[np.ndarray], emb2: List[np.ndarray]) -> float:
    """Compute similarity between two sets of embeddings"""  
    return np.mean([np.dot(e1.flatten(), e2.flatten()) / 
                   (np.linalg.norm(e1) * np.linalg.norm(e2)) 
                   for e1, e2 in zip(emb1, emb2)])

@app.post("/signup")
async def signup(username: str, audio_file: UploadFile = File(...)):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    
    try:
        c.execute("SELECT username FROM users WHERE username = ?", (username,))
        if c.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists")
        
        file_path = os.path.join(AUDIO_UPLOAD_DIR, f"{uuid.uuid4()}.wav")
        with open(file_path, "wb") as buffer:
            buffer.write(await audio_file.read())
        
        processed_segments = preprocess_audio(file_path)
        if not processed_segments:
            raise HTTPException(status_code=400, detail="Audio file too short")
        
        embeddings = extract_embeddings(processed_segments)
        save_embeddings(username, embeddings)
        
        c.execute("INSERT INTO users (username, embeddings_stored) VALUES (?, ?)",
                 (username, True))
        conn.commit()
        
        return {"message": "User registered successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        conn.close()
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/login")
async def login(username: str, audio_file: UploadFile = File(...)):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    
    try:

        c.execute("SELECT embeddings_stored FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        if not result or not result[0]:
            raise HTTPException(status_code=404, detail="User not found or no embeddings stored")
        
        file_path = os.path.join(AUDIO_UPLOAD_DIR, f"{uuid.uuid4()}.wav")
        with open(file_path, "wb") as buffer:
            buffer.write(await audio_file.read())
        
        processed_segments = preprocess_audio(file_path)
        if not processed_segments:
            raise HTTPException(status_code=400, detail="Audio file too short")
        
        new_embeddings = extract_embeddings(processed_segments)
        stored_embeddings = get_embeddings(username)
        
        similarity = compute_similarity(
            [emb.cpu().numpy() for emb in new_embeddings],
            stored_embeddings
        )
        
        if similarity >= SIMILARITY_THRESHOLD:
            return {"message": "Login successful"}
        else:
            raise HTTPException(status_code=401, detail="Voice authentication failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        conn.close()
        if os.path.exists(file_path):
            os.remove(file_path)