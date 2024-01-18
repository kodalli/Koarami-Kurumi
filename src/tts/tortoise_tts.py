from tortoise import api, utils

clips_paths = []
reference_clips = [utils.audio.load_audio(p, 22050) for p in clips_paths]
tts = api.TextToSpeech(use_deepspeed=True)
pcm_audio = tts.tts_with_preset("Hello world", voice_samples=reference_clips, preset="fast")