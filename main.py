from fastapi import FastAPI, UploadFile, File
from typing import List
from pydantic import BaseModel
import os
import whisper
from transformers import pipeline

app = FastAPI()

class TranscriptionResponse(BaseModel):
    text: str
    timestamps: List[dict]

class SummaryResponse(BaseModel):
    summary: str

# Load the whisper-large-v3 model
model = whisper.load_model("large")

# Choose a suitable summarization model
summarizer = pipeline("summarization")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Read the audio file
        audio_data = await file.read()
        
        # Transcribe the audio file using the whisper model
        result = whisper.transcribe(model, audio_data)
        
        # Extract the transcription text and timestamps
        transcription_text = result["text"]
        timestamps = result["segments"]
        
        # Save the transcription, summary, and timestamps to the local machine
        transcription_file = "transcription.txt"
        summary_file = "summary.txt"
        timestamps_file = "timestamps.json"
        
        with open(transcription_file, "w") as f:
            f.write(transcription_text)
        
        with open(timestamps_file, "w") as f:
            import json
            json.dump(timestamps, f)
        
        # Generate a summary using the summarization model
        summary = summarizer(transcription_text, max_length=100, min_length=30)[0]["summary_text"]
        
        with open(summary_file, "w") as f:
            f.write(summary)
        
        return TranscriptionResponse(text=transcription_text, timestamps=timestamps)
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/summarize")
async def summarize_text(text: str):
    try:
        # Generate a summary using the summarization model
        summary = summarizer(text, max_length=100, min_length=30)[0]["summary_text"]
        
        # Save the summary to the local machine
        summary_file = "summary.txt"
        with open(summary_file, "w") as f:
            f.write(summary)
        
        return SummaryResponse(summary=summary)
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)