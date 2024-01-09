import wave
from piper import PiperVoice
from tts.tts import TextToSpeech


class PiperTTS(TextToSpeech):
    def __init__(self, model_path, config_path, use_cuda=False):
        self.voice = PiperVoice.load(model_path, config_path, use_cuda)

    def say(self, text, sentence_silence=0.1, length_scale=1.2, noise_scale=0.3, **kwargs):
        return self.voice.synthesize_stream_raw(text, **kwargs)

    def sample_rate(self):
        return self.voice.config.sample_rate



if __name__ == "__main__":
    tts_engine = PiperTTS(model_path="models/en_US-libritts_r-medium.onnx", config_path="models/en_US-libritts_r-medium.onnx.json")
    # file = wave.open("test.wav", "wb")
    text = "Some times like to haw before I yee, but some might consider it sacrilege. "
    # tts_engine.voice.synthesize(text=text, wav_file=file, sentence_silence=0.1, length_scale=1.4, noise_scale=0.3)
    generator = tts_engine.voice.synthesize_stream_raw(text=text, sentence_silence=0.1, length_scale=1.2, noise_scale=0.3)
    for g in generator:
        print(g)