# Text to Speech

from abc import ABC, abstractmethod
import asyncio
import threading
import tkinter as tk
from tkinter import simpledialog
from typing import Iterable
import pyaudio
import websockets
import json

class TextToSpeech(ABC):

    # 16-bit mono
    @abstractmethod
    def say(self, text, **kwargs) -> Iterable[bytes]:
        pass 

    @abstractmethod
    def sample_rate(self) -> int:
        pass


class TTSWebSocketServer:
    def __init__(self, tts_engine):
        self.tts_engine = tts_engine

    async def handler(self, websocket, path):
        async for message in websocket:
            data = json.loads(message)
            text = data.get('text')
            if text:
                audio_generator = self.tts_engine.say(text)
                await self.send_audio_stream(websocket, audio_generator)

    async def send_audio_stream(self, websocket, audio_generator):
        for audio_chunk in audio_generator:
            await websocket.send(audio_chunk)

    def run(self, host, port):
        start_server = websockets.serve(self.handler, host, port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

    

# class AudioSynthesizerGUI:
#     def __init__(self, tts: TextToSpeech):
#         self.tts = tts
#         self.root = tk.Tk()
#         self.root.title("Text to Speech")

#         self.start_button = tk.Button(self.root, text="Synthesize and Play", command=self.start_synthesis)
#         self.start_button.pack(pady=10)

#         self.audio_stream = None

#     def start_synthesis(self):
#         text = simpledialog.askstring("Input", "Enter text to synthesize")
#         if text:
#             threading.Thread(target=self.synthesize_and_play, args=(text,), daemon=True).start()

#     def synthesize_and_play(self, text):
#         p = pyaudio.PyAudio()
#         self.audio_stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.tts.sample_rate(), output=True) 
    
#         for audio_chunk in self.tts.say(text):
#             self.audio_stream.write(audio_chunk) 

#         self.audio_stream.stop_stream()
#         self.audio_stream.close()
#         p.terminate()

#     def run(self):
#         self.root.mainloop()
