# Adapted from https://github.com/JarodMica/audiosplitter_whisper
import os
import re
import unicodedata
import wave
from pydub import AudioSegment
import pysrt
import dotenv
import subprocess
import shutil

def split_audio_file(file_path, segment_duration=10):
    with wave.open(file_path) as f:
        sample_rate, audio_data = f.getframerate(), f.readframes(f.getnframes())

    num_segments = int(len(audio_data) / sample_rate / segment_duration)
    remainder = len(audio_data) % (sample_rate * segment_duration)
    if remainder > 0:
        num_segments += 1

    ouput_dir = os.path.dirname(file_path)
    base_filename = os.path.basename(file_path).split(".")[0]

    for i in range(num_segments):
        start = i * sample_rate * segment_duration
        end = min((i + 1) * sample_rate * segment_duration, len(audio_data))
        segment = audio_data[start:end]
        segment_path = os.path.join(ouput_dir, f"{base_filename}_{i}.wav")
        with wave.open(segment_path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(segment)
        
        print(f"Segment {i+1}/{num_segments} saved: {segment_path}")

def split_all_audio(input_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_directory, filename)
            split_audio_file(file_path)
    
def sanitize_filename(filename):
    # Remove diacritics and normalize Unicode characters
    normalized_filename = unicodedata.normalize("NFKD", filename).encode("ASCII", "ignore").decode("ASCII")
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, "_", normalized_filename)

def process_subtitle(audio, sub, output_dir, count, padding=0.0):
    """
    Args:
        - padding(int): how much additional sound to include before and after audio, useful for audio that is getting clipped.
    """
    start_time  = max(0, sub.start.ordinal - padding * 1000)
    end_time = min(len(audio), sub.end.ordinal + padding * 1000)
    segment = audio[start_time:end_time]
    output_filename = os.path.join(output_dir, f"segment_{count}.wav")
    output_path = os.path.join(output_dir, output_filename)
    segment.export(output_path, format="wav")
    print(f"Segment saved: {output_path}")

def diarize_audio_with_srt(audio_file, srt_file, output_dir):
    """
    Use whisperX generated SRT files to split the audio files w/ speaker numbering and diarization.

    Args:
        - audio_file(str): path to audio file
        - srt_file(str): path to srt file use for splicing
        - output_dir(str): path to output directory
    """
    audio = AudioSegment.from_file(audio_file)
    subs = pysrt.open(srt_file)
    for i, sub in enumerate(subs):
        speaker = sub.text.split("]")[0][1:]
        sanitize_speaker = sanitize_filename(speaker)
        speaker_dir = os.path.join(output_dir, sanitize_speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        process_subtitle(audio, sub, speaker_dir, i)

def extract_audio_with_srt(audio_file, srt_file, output_dir):
    """
    Use whisperX generated SRT files to split the audio files

    Args:
        - audio_file(str): path to audio file
        - srt_file(str): path to srt file use for splicing
        - output_dir(str): path to output directory
    """
    audio = AudioSegment.from_file(audio_file)
    subs = pysrt.open(srt_file)
    os.makedirs(output_dir, exist_ok=True)
    for i, sub in enumerate(subs):
        process_subtitle(audio, sub, output_dir, i)

def run_whisperX(audio_files, output_dir, model, diarize):
    compute_type = "float16"
    device = "cuda"
    language = "en"
    output_format = "srt"
    
    base_cmd = [
        "whisperx", audio_files,
        "--device", device,
        "--model", model,
        "--output_dir", output_dir,
        "--language", language,
        "--output_format", output_format,
        "--compute_type", compute_type,
    ]

    if diarize:
        dotenv.load_dotenv()
        base_cmd.extend(["--diarize", "--hf_token", os.getenv("HF_TOKEN")])

    subprocess.run(base_cmd)

def process_audio_files(input_folder, model, diarize):
    output_dir = os.path.join(input_folder, "output")
    wav_dir = os.path.join(input_folder, "wav_files")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    for audio_file in os.listdir(input_folder):
        audio_file_path = os.path.join(input_folder, audio_file)
        if not os.path.isfile(audio_file_path):
            continue

        if not audio_file.endswith(".wav"):
            wav_file_path = os.path.join(wav_dir, f"{os.path.splitext(audio_file)[0]}.wav")
            try:
                subprocess.run(["ffmpeg", "-i", audio_file_path, wav_file_path], check=True)
                audio_file_path = wav_file_path
            except subprocess.CalledProcessError as e:
                print(f"Error converting {audio_file_path} to wav: {e}")
                continue

        run_whisperX(audio_file_path, output_dir, model, diarize)
        srt_file = os.path.join(output_dir, f"{os.path.splitext(audio_file)[0]}.srt")

        speaker_segments_dir = os.path.join(output_dir, os.path.splitext(audio_file)[0])
        os.makedirs(speaker_segments_dir, exist_ok=True)

        if diarize:
            diarize_audio_with_srt(audio_file_path, srt_file, speaker_segments_dir)
        else:
            extract_audio_with_srt(audio_file_path, srt_file, speaker_segments_dir)
    if not diarize:
        merge_segments(output_dir)

def merge_segments(output_dir):
    combined_dir = os.path.join(output_dir, "combined_folder")
    os.makedirs(combined_dir, exist_ok=True)

    for folder_name in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder_name)
        if not os.path.isdir(folder_path) or folder_name == "combined_folder":
            continue

        for segment_name in os.listdir(folder_path):
            segment_path = os.path.join(folder_path, segment_name)
            new_segment_name = f"{folder_name}_{segment_name.split('_')[-1]}"
            new_segment_path = os.path.join(combined_dir, new_segment_name)
            shutil.move(segment_path, new_segment_path)

        os.rmdir(folder_path)

if __name__ == "__main__":
    # Choices: ["tiny","base","small","medium","large-v2"]
    process_audio_files(input_folder="input", model="large-v2", diarize=True)