from typing import Union

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from data import DataFetcher
import os
import torch
import faster_whisper
from driver import GraphRAG, Chat
import uvicorn
import io
import base64
import asyncio
from fastapi.responses import JSONResponse
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend's domain in production
    allow_credentials=True,
    allow_methods=["*"],  # Update with your frontend's domain in production
    allow_headers=["*"],  # Update with your frontend's domain in production
)

rag = GraphRAG()
rag.build_graph()
chat = Chat(rag)

class PromptRequest(BaseModel):
    prompt: str


@app.get("/")  # Update with your frontend's domain in production
async def read_root():
    return {"Hello": "World"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint to accept audio data and return transcription.
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        audio_data = await file.read()
        audio_stream = io.BytesIO(audio_data)

        # Transcribe audio using Whisper model
        model = faster_whisper.WhisperModel(
            model_size_or_path="tiny.en",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        segments, _ = model.transcribe(
            audio_stream, language="en", vad_filter=True, beam_size=1
        )
        transcription = " ".join([segment.text for segment in segments])

        print(f"Transcription: {transcription}")
        return JSONResponse(content={"transcription": transcription})
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error during transcription: {str(e)}"
        )

@app.post("/rag_qa")
async def rag_qa(request: Request, prompt: PromptRequest):
    response =  rag.answer_question(prompt.prompt)
    return {"response": response}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive text data from the client
            data = await websocket.receive_text()
            print(f"Received text: {data}")

            # Generate response audio asynchronously
            async for response in chat.generate_audio_stream(data):
                if response["type"] == "node_ids":
                    # Send node IDs as JSON
                    await websocket.send_json({"type": "node_ids", "data": response["data"]})
                elif response["type"] == "audio":
                    # Send audio chunks as bytes
                    await websocket.send_bytes(response["data"])
                elif response["type"] == "chart":
                    # Send response as JSON
                    await websocket.send_json({"type": "chart", "data": response["data"]})
                elif response["type"] == "reply":
                    # Send response as JSON
                    await websocket.send_json({"type": "reply", "data": {"text":response["data"]}})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8555, reload=True)