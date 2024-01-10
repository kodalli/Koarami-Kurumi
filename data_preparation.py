from moviepy.editor import VideoFileClip
import os
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
from pydub import AudioSegment, silence
import pandas as pd
from stt.whisper_stt import WhisperSTT 


def convert_mp4_to_wav(mp4_file_folder):
    wav_file_folder = os.path.join(os.path.dirname(mp4_file_folder), "audio/")
    os.makedirs(wav_file_folder, exist_ok=True)
    for file in os.listdir(mp4_file_folder):
        if file.endswith(".mp4"):
            mp4_file = os.path.join(mp4_file_folder, file)
            wav_file = os.path.join(wav_file_folder, file.replace(".mp4", ".wav"))
            clip = VideoFileClip(mp4_file)
            clip.audio.write_audiofile(wav_file)


def resample_wav(streamer_folder, name):
    streamer_folder = os.path.dirname(streamer_folder)
    wav_source_folder = os.path.join(streamer_folder, "audio/")
    resampled_audio_folder = os.path.join(streamer_folder, "resampled_audio/")
    os.makedirs(resampled_audio_folder, exist_ok=True)
    count = 0
    for file in os.listdir(wav_source_folder):
        if file.endswith(".wav"):
            sound = AudioSegment.from_wav(os.path.join(wav_source_folder, file))
            sound = sound.set_channels(1)
            sound = sound.set_frame_rate(8000)
            file_name = os.path.basename(file).split("/")[-1]
            sound.export(
                os.path.join(resampled_audio_folder, f"{name}_{count}.wav"),
                format="wav",
            )
            count += 1

def clip_audio(audio_file, output_file, start_time, end_time):
    sound = AudioSegment.from_wav(audio_file)
    clip = sound[start_time:end_time]
    clip.export(output_file, format="wav")

def create_clip_metadata(resampled_audio_folder, clip_folder, batch_length_sec=100):
    files = [file for file in os.listdir(resampled_audio_folder) if file.endswith(".wav")]
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    start_time = 0
    counter = 0
    batch_length_ms = batch_length_sec * 1000
    data = []  
    os.makedirs(clip_folder, exist_ok=True)

    total_length = 0

    for file in files:
        sound = AudioSegment.from_wav(os.path.join(resampled_audio_folder, file))
        length_ms = len(sound)
        total_length += length_ms
        for i in range(0, length_ms, batch_length_ms):
            start = i
            end = i + batch_length_ms
            batch = sound[start:end]
            batch_file_name = os.path.join(clip_folder, f"{counter}.wav")
            batch.export(batch_file_name, format="wav")
            batch_data = {"file_name": batch_file_name, "start": start + start_time, "end": len(batch) + start_time}
            data.append(batch_data) 
            counter += 1
            start_time += len(batch)

    print(f"Total length: {total_length / 1000} seconds")
    print(f"start_time: {start_time / 1000} seconds")
    df = pd.DataFrame(data, columns=["file_name", "start", "end"])
    df.to_csv(os.path.join(clip_folder, "metadata.csv"), index=False)

def isolate_voice(streamer_folder):
    model = separator.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir="data/pretrained_models/sepformer-wsj02mix",
        run_opts={"device": "cuda"},
    )
    streamer_folder = os.path.dirname(streamer_folder)
    wav_source_folder = os.path.join(streamer_folder, "resampled_audio/")
    separated_audio_folder = os.path.join(streamer_folder, "separated_audio/")
    os.makedirs(separated_audio_folder, exist_ok=True)
    for file in os.listdir(wav_source_folder):
        if file.endswith(".wav"):
            file_name = os.path.basename(file).split("/")[-1]
            est_sources = model.separate_file(os.path.join(wav_source_folder, file), "data/audio_cache")
            torchaudio.save(
                os.path.join(separated_audio_folder, "1-" + file_name),
                est_sources[:, :, 0].detach().cpu(),
                8000,
            )
            torchaudio.save(
                os.path.join(separated_audio_folder, "2-" + file_name),
                est_sources[:, :, 1].detach().cpu(),
                8000,
            )


def detect_silence_and_split(filename, min_silence_len=1000, silence_thresh=-40):
    audio_segment = AudioSegment.from_file(filename)
    non_silent_chunks = silence.detect_nonsilent(
        audio_segment, min_silence_len, silence_thresh
    )
    return non_silent_chunks, audio_segment


def process_chunks(non_silent_chunks, audio_segment, model):
    results = []
    for start_ms, end_ms in non_silent_chunks:
        chunk = audio_segment[start_ms:end_ms]
        chunk_bytes = chunk.raw_data
        text = model.hear(chunk_bytes)
        results.append({"start": start_ms / 1000, "end": end_ms / 1000, "text": text})
    return results


def save_to_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)


def transcribe_voice(streamer_folder):
    # Collect parts of audio that are not silent
    # Timestamps of each part
    # Transcribe each part
    # Output csv start_time, end_time, text
    model = WhisperSTT()
    streamer_folder = os.path.dirname(streamer_folder)
    wav_source_folder = os.path.join(streamer_folder, "separated_audio/")
    transcription_folder = os.path.join(streamer_folder, "transcription/")
    os.makedirs(transcription_folder, exist_ok=True)
    for file in os.listdir(wav_source_folder):
        if file.endswith(".wav") and file.startswith("2-"):
            file_name = os.path.basename(file).split("/")[-1]
            source_audio_file = os.path.join(wav_source_folder, file)
            non_silent_chunks, audio_segment = detect_silence_and_split(
                source_audio_file
            )
            results = process_chunks(non_silent_chunks, audio_segment, model)
            save_to_csv(
                results,
                os.path.join(transcription_folder, file_name.replace(".wav", ".csv")),
            )


def prepare_audio(folder):
    convert_mp4_to_wav(folder)
    resample_wav(folder)
    isolate_voice(folder)
    transcribe_voice(folder)

if __name__ == "__main__":
    # prepare_audio("data/XL/")
    # isolate_voice("data/Toma/")
    # clip_audio("data/Toma/resampled_audio/resampled-3.0 TOMA DEBUT 💛 BIRTHDAY SUBATHON 💛 (4).wav", "data/XL/resampled_audio/toma_debut_4_clip_0s_1000s.wav", 0, 100 * 1000)
    # isolate_voice("data/XL/")
    # transcribe_voice("data/XL/")
    create_clip_metadata("data/Toma/resampled_audio/", "data/Toma/clips/")
    pass
