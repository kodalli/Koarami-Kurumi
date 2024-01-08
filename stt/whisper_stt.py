import numpy as np
import whisper
import torch
from stt import SpeechToText


class WhisperSTT(SpeechToText):
    def __init__(self, model_name="medium.en"):
        self.model = whisper.load_model(model_name)

    def hear(self, audio: bytes) -> str:
        # Convert in-ram buffer to something the model can use directly without needing a temp file.
        # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
        # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Read the transcription.
        result = self.model.transcribe(audio_np, fp16=torch.cuda.is_available())
        text = result['text'].strip()
        return text




