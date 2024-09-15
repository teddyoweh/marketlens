from typing import Union

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import torch
import faster_whisper
import uvicorn
import io
import base64
import yfinance as yf
import matplotlib.pyplot as plt
from PIL import Image
import openai
from .driver import GraphRAG, Chat


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
 
graph_rag = GraphRAG()
chat_system = Chat(graph_rag)



@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint to accept audio data and return transcription.
    """
    if not file:
        return {"error": "No audio file provided"}, 400
    
    if file.filename == '':
        return {"error": "No selected file"}, 400
    
    try:
        # Read the audio file into memory
        audio_data = await file.read()
        audio_stream = io.BytesIO(audio_data)
        
        # Transcribe audio using Whisper model
        model = faster_whisper.WhisperModel(
            model_size_or_path="tiny.en",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        segments, _ = model.transcribe(audio_stream, language="en", vad_filter=True, beam_size=1)
        transcription = " ".join([segment.text for segment in segments])
        
        print(transcription, "transcription")
        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/generate_script")
async def generate_script(request: PromptRequest):
    """
    Endpoint to generate a script based on the RAG system using the provided prompt.
    """
    user_prompt = request.prompt
    response_text = generate_gpt_response(user_prompt)
    return {"script": response_text}

def generate_gpt_response(prompt: str) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=250
    )
    return response.choices[0].text.strip()



@app.post("/generate-response/")
async def generate_response(request: PromptRequest):
    try:
        prompt = request.prompt
        
        # Generate a response using GPT-3
        response_text = generate_gpt_response(prompt)
        
        # Optionally, generate a corresponding graph using GraphRAG
        graph = graph_rag.create_graph(prompt)
        
        # Encode the graph as a base64 string if it's generated as an image
        graph_bytes = io.BytesIO()
        graph.save(graph_bytes, format='PNG')
        graph_bytes.seek(0)
        graph_base64 = base64.b64encode(graph_bytes.read()).decode()
        
        return {
            "response": response_text,
            "graph": graph_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def fetch_stock_data(ticker: str):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    return hist

@app.get("/stock/{ticker}")
async def get_stock_data(ticker: str):
    try:
        hist = fetch_stock_data(ticker)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
        
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        
        # Generate a graph
        plt.figure()
        plt.plot(dates, prices, marker='o')
        plt.title(f"{ticker} Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.xticks(rotation=45)
        
        # Save graph to a bytes object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        graph_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return {
            "ticker": ticker,
            "dates": dates,
            "prices": prices,
            "graph": graph_base64,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)