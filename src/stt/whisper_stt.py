import numpy as np
import whisper
import torch
from src.stt import SpeechToText
import librosa
from typing import Union


class WhisperSTT(SpeechToText):
    def __init__(self, model_name="medium.en"):
        self.model = whisper.load_model(model_name)

    def hear(self, audio: Union[bytes, np.array]) -> str:
        # Convert in-ram buffer to something the model can use directly without needing a temp file.
        # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
        # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
        if isinstance(audio, bytes):
            audio_np = np.frombuffer(audio, dtype=np.int16)
            # Resample to 16kHz if necessary.
            if audio_np.shape[0] != 16000:
                print("Resampling audio from {} to 16000".format(audio_np.shape[0]))
                audio_np = librosa.resample(audio_np.astype(float), orig_sr=8000, target_sr=16000)
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif isinstance(audio, np.ndarray):
            audio_np = audio.astype(np.float32) / 32768.0

        # Read the transcription.
        result = self.model.transcribe(audio_np, fp16=torch.cuda.is_available())
        text = result['text'].strip()
        return text

