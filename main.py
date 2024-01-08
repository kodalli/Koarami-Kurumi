from llm.deciLM_7b import DeciLM7b
from llm.llm import LanguageModel
from stt.stt import SpeechToText
from stt.whisper_stt import WhisperSTT
from tts.piper_tts import PiperTTS
from tts.tts import TextToSpeech
import asyncio
import websockets

class WebSocketServer:
    def __init__(self, tts_engine: TextToSpeech, stt_engine: SpeechToText, llm_engine: LanguageModel):
        self.tts_engine = tts_engine
        self.stt_engine = stt_engine
        self.llm_engine = llm_engine

    async def handler(self, websocket, path):
        audio_data = b''
        counter = 0
        async for message in websocket:
            counter += 1
            if isinstance(message, bytes):
                audio_data += message
            elif isinstance(message, str) and message == "end_of_audio":
                print("End of audio")
                break
            print(len(audio_data))

        print("Received {} audio chunks".format(counter))
        
        transcribed_text = self.stt_engine.hear(audio_data)
        print("Transcribed text: " + transcribed_text)

        llm_response = self.llm_engine.think(transcribed_text)
        print("Response text: " + llm_response)

        audio_generator = self.tts_engine.say(llm_response, sentence_silence=0.1, length_scale=1.4, noise_scale=0.3)

        await self.send_audio_stream(websocket, audio_generator)

    async def send_audio_stream(self, websocket, audio_generator):
        if audio_generator is None:
            print("Audio generator is None")
            return
        for audio_chunk in audio_generator:
            await websocket.send(audio_chunk)

    def run(self, host, port):
        start_server = websockets.serve(self.handler, host, port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

def main():
    tts_engine = PiperTTS(model_path="tts/models/en_US-libritts_r-medium.onnx", config_path="tts/models/en_US-libritts_r-medium.onnx.json")
    stt_engine = WhisperSTT()
    llm_engine = DeciLM7b(quantize=True)
    server = WebSocketServer(tts_engine, stt_engine, llm_engine)
    server.run('localhost', 8765)

if __name__ == "__main__":
    main()
