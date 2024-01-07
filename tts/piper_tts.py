from piper import PiperVoice
from tts.tts import TextToSpeech


class PiperTTS(TextToSpeech):
    def __init__(self, model_path, config_path, use_cuda=False):
        self.voice = PiperVoice.load(model_path, config_path, use_cuda)

    def say(self, text, **kwargs):
        self.voice.synthesize_stream_raw(text, **kwargs)

    def sample_rate(self):
        return self.voice.config.sample_rate



if __name__ == "__main__":
    pass