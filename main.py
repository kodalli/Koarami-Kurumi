from llm.deciLM_7b import DeciLM7b
from llm.llm import LanguageModel
from stt.stt import SpeechToText
from stt.whisper_stt import WhisperSTT
from tts.piper_tts import PiperTTS


# def main():
#     thought = think()
#     print(thought)

# def listen():
#     pass

# def think():
#     deciLM = DeciLM7b(quantize=True) 
#     response = deciLM.run(prompt="Yeehaw!")
#     return response
    
import asyncio
import websockets
import json

from tts.tts import TextToSpeech

class WebSocketServer:
    def __init__(self, tts_engine: TextToSpeech, stt_engine: SpeechToText, llm_engine: LanguageModel):
        self.tts_engine = tts_engine
        self.stt_engine = stt_engine
        self.llm_engine = llm_engine

    async def handler(self, websocket, path):
        audio_data = b''
        async for message in websocket:
            if isinstance(message, bytes):
                audio_data += message
            elif isinstance(message, str) and message == "end_of_audio":
                break
        
        transcribed_text = self.stt_engine.hear(audio_data)
        llm_response = self.llm_engine.think(transcribed_text)
        audio_generator = self.tts_engine.say(llm_response)

        await self.send_audio_stream(websocket, audio_generator)

    async def send_audio_stream(self, websocket, audio_generator):
        for audio_chunk in audio_generator:
            await websocket.send(audio_chunk)

    def run(self, host, port):
        start_server = websockets.serve(self.handler, host, port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

def main():
    tts_engine = PiperTTS(model_path="tts/models/en_US-libritts-high.onnx", config_path="tts/models/en_US-libritts-high.onnx.json")
    stt_engine = WhisperSTT()
    llm_engine = DeciLM7b(quantize=True)
    server = WebSocketServer(tts_engine, stt_engine, llm_engine)
    server.run('localhost', 8765)


if __name__ == "__main__":
    main()
