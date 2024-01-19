from vtv import VoiceToVoice

class RetrievalVoiceConversion(VoiceToVoice):
    def __init__(self, retrieval_model, voice_conversion_model):
        self.retrieval_model = retrieval_model
        self.voice_conversion_model = voice_conversion_model

    def convert(self, audio, **kwargs):
        pass

    def convert_file(self, audio_file, **kwargs):
        pass