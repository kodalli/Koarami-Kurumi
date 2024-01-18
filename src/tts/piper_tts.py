import os
import wave
from piper import PiperVoice
from src.tts import TextToSpeech


class PiperTTS(TextToSpeech):
    def __init__(self, model_path, config_path, use_cuda=False):
        self.voice = PiperVoice.load(model_path, config_path, use_cuda)

    def say(self, text, sentence_silence=0.1, length_scale=1.2, noise_scale=0.3, **kwargs):
        return self.voice.synthesize_stream_raw(text, **kwargs)

    def sample_rate(self):
        return self.voice.config.sample_rate
    
    def save(self, text, path, **kwargs):
        with wave.open(path, "wb") as wav_file:
            return self.voice.synthesize(text, wav_file, **kwargs)

def play_voice(text):
    # name = "en_US-libritts_r-medium.onnx"
    # name = "en_US-libritts-high.onnx"
    # name = "en_US-amy-medium.onnx"
    # name = "en_US-lessac-medium.onnx"
    name = "en_US-kathleen-low.onnx"
    # name = "en_GB-jenny_dioco-medium.onnx"

    tts_engine = PiperTTS(model_path=f"models/{name}", config_path=f"models/{name}.json")
    # tts_engine.save(text, "test.wav", length_scale=1.2, sentence_silence=0.3, noise_scale=0.1, noise_w=0.2)
    # tts_engine.save(text, "test.wav", length_scale=1.2, sentence_silence=0.3, noise_scale=0.3, noise_w=0.5)
    tts_engine.save(text, "test.wav", length_scale=0.75, sentence_silence=0.3, noise_scale=0.3, noise_w=0.5)


if __name__ == "__main__":
    text = "Hi what's up?"
    play_voice(text)
