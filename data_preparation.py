import json
from moviepy.editor import VideoFileClip
import os
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
from pydub import AudioSegment, silence
import pandas as pd
from llm.inference.deciLM_7b import DeciLM7b
from stt.whisper_stt import WhisperSTT 
from tqdm import tqdm


def convert_mp4_to_wav(mp4_file_folder):
    """
    Convert MP4 files to WAV format.

    Args:
        mp4_file_folder (str): The path to the folder containing the MP4 files.

    Returns:
        None
    """
    wav_file_folder = os.path.join(os.path.dirname(mp4_file_folder), "audio/")
    os.makedirs(wav_file_folder, exist_ok=True)
    for file in os.listdir(mp4_file_folder):
        if file.endswith(".mp4"):
            mp4_file = os.path.join(mp4_file_folder, file)
            wav_file = os.path.join(wav_file_folder, file.replace(".mp4", ".wav"))
            clip = VideoFileClip(mp4_file)
            clip.audio.write_audiofile(wav_file)


def resample_wav(streamer_folder, name):
    """
    Resamples the WAV files in the specified streamer folder to a single channel and a sample rate of 8000 Hz.
    
    Args:
        streamer_folder (str): The path to the streamer folder.
        name (str): The name to be appended to the resampled audio files.
    """
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
            sound.export(
                os.path.join(resampled_audio_folder, f"{name}_{count}.wav"),
                format="wav",
            )
            count += 1

def clip_audio(audio_file, output_file, start_time, end_time):
    """
    Clips a portion of an audio file and exports it to a new file.

    Parameters:
    audio_file (str): The path to the input audio file.
    output_file (str): The path to the output audio file.
    start_time (int): The start time of the clip in milliseconds.
    end_time (int): The end time of the clip in milliseconds.
    """
    sound = AudioSegment.from_wav(audio_file)
    clip = sound[start_time:end_time]
    clip.export(output_file, format="wav")

def batch_clips(resampled_audio_folder, clip_folder, batch_length_sec=100):
    """
    Batch clips from resampled audio files and export them as individual WAV files.
    Filenames must have a "_#.wav" suffix where # is the clip number.
    
    Args:
        resampled_audio_folder (str): Path to the folder containing the resampled audio files.
        clip_folder (str): Path to the folder where the batch clips will be saved.
        batch_length_sec (int, optional): Length of each batch clip in seconds. Defaults to 100.
    
    Returns:
        None
    
    Raises:
        None
    """
    files = [file for file in os.listdir(resampled_audio_folder) if file.endswith(".wav")]
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    print(files)

    start_time = 0
    counter = 0
    batch_length_ms = batch_length_sec * 1000
    data = []  
    os.makedirs(clip_folder, exist_ok=True)

    total_length = 0

    for file in files:
        sound = AudioSegment.from_wav(os.path.join(resampled_audio_folder, file))
        length_ms = len(sound)
        print(f"length: {length_ms}")
        total_length += length_ms
        for i in range(0, length_ms, batch_length_ms):
            start = i
            end = i + batch_length_ms
            # print(f"start: {start}, end: {end}")
            batch = sound[start:end]
            batch_file_name = os.path.join(clip_folder, f"{counter}.wav")
            batch.export(batch_file_name, format="wav")
            batch_data = {"file_name": batch_file_name, "start": start_time, "end": start_time + len(batch)}
            data.append(batch_data) 
            counter += 1
            start_time += len(batch)
            print(f"batch length: {len(batch)}")

    print(f"Total length: {total_length / 1000} seconds")
    print(f"start_time: {start_time / 1000} seconds")
    df = pd.DataFrame(data, columns=["file_name", "start", "end"])
    df.to_csv(os.path.join(clip_folder, "clip_timestamps.csv"), index=False)

def isolate_voice(streamer_folder):
    """
    Function to isolate the voice from an audio stream using a pre-trained model.

    Args:
        streamer_folder (str): The path to the folder containing the streamers audio data.

    Returns:
        None
    """
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
    """
    Detects silence in an audio file and splits it into non-silent chunks.

    Args:
        filename (str): The path to the audio file.
        min_silence_len (int, optional): The minimum duration of silence in milliseconds to be considered as non-silent. Defaults to 1000.
        silence_thresh (float, optional): The threshold in dBFS below which the audio is considered as silence. Defaults to -40.

    Returns:
        tuple: A tuple containing the non-silent chunks and the original audio segment.
    """
    audio_segment = AudioSegment.from_file(filename)
    non_silent_chunks = silence.detect_nonsilent(
        audio_segment, min_silence_len, silence_thresh
    )
    return non_silent_chunks, audio_segment


def process_chunks(non_silent_chunks, audio_segment, model):
    """
    Process non-silent audio chunks and transcribe them using a given model.

    Args:
        non_silent_chunks (list): List of tuples representing the start and end times of non-silent audio chunks.
        audio_segment (AudioSegment): Audio segment containing the entire audio.
        model (Model): Model used for transcription.

    Returns:
        list: List of dictionaries containing the start time, end time, and transcribed text for each chunk.
    """
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


def transcribe_voice(audio_clips_folder, transcription_folder):
    model = WhisperSTT()
    os.makedirs(transcription_folder, exist_ok=True)
    for file in tqdm(os.listdir(audio_clips_folder), desc="Transcribing"):
        if file.endswith(".wav"): # and file.startswith("2-"):
            file_name = os.path.basename(file).split("/")[-1]
            source_audio_file = os.path.join(audio_clips_folder, file)
            non_silent_chunks, audio_segment = detect_silence_and_split(
                source_audio_file
            )
            results = process_chunks(non_silent_chunks, audio_segment, model)
            save_to_csv(
                results,
                os.path.join(transcription_folder, file_name.replace(".wav", ".csv")),
            )

def combine_transcriptions_csv(transcription_folder, clips_folder):
    files = [file for file in os.listdir(transcription_folder) if file.endswith(".csv") and file != "full_transcription.csv"]
    files.sort(key=lambda x: int(x.split(".")[0]))
    df_combined = pd.DataFrame(columns=["start", "end", "text"])
    start_time = 0
    for file in tqdm(files, desc="Combining"):
        if file.endswith(".csv"):
            file_path = os.path.join(transcription_folder, file)
            try:
                df_local = pd.read_csv(file_path)
            except:
                audio_length = AudioSegment.from_file(os.path.join(clips_folder, file.replace(".csv", ".wav"))).duration_seconds
                start_time += audio_length
                print(f"{file_path} is empty. Skipping...")
                continue
            print(file_path)
            df_local.sort_values(by=["start"], inplace=True)
            end = df_local["end"].iloc[-1]
            df_local["start"] += start_time
            df_local["end"] += start_time
            start_time += end
            df_combined = pd.concat([df_combined, df_local], ignore_index=True)
    # Sort by timestamp
    df_combined.sort_values(by=["start"], inplace=True)
    # Convert to int
    df_combined["start"] = df_combined["start"].astype(int)
    df_combined["end"] = df_combined["end"].astype(int)
    # Remove empty rows
    df_combined = df_combined[pd.notnull(df_combined["text"])].reset_index(drop=True)
    # Remove leading and trailing double quotes, csv will still have them b/c of special characters
    df_combined["text"] = df_combined["text"].str.strip()
    df_combined["text"] = df_combined["text"].str.replace('"', "")
    transcription_folder = os.path.join(transcription_folder, "../")
    df_combined.to_csv(os.path.join(transcription_folder, "full_transcription.csv"), index=False)

def combine_twitch_chat_with_streamer_transcription(streamer_full_transcription_file, twitch_chat_file, output_file, streamer_name, time_offset=0):
    df_streamer = pd.read_csv(streamer_full_transcription_file)
    df_streamer["user_name"] = streamer_name
    df_streamer["start"] += time_offset
    df_streamer.drop(columns=["end"], inplace=True)

    df_twitch = pd.read_csv(twitch_chat_file, on_bad_lines="skip")
    df_twitch.rename(columns={"time": "start", "message": "text"}, inplace=True)
    df_twitch.drop(columns=["user_color"], inplace=True)

    df_combined = pd.concat([df_streamer, df_twitch], ignore_index=True)
    df_combined.sort_values(by=["start"], inplace=True)
    df_combined["text"] = "[" + df_combined["user_name"] + "]: " + df_combined["text"]
    tokenizer = DeciLM7b.get_tokenizer()
    df_combined["tokens"] = df_combined["text"].apply(lambda x: tokenizer.encode(str(x), return_tensors="pt").shape[1])
    print(f"Total tokens: {df_combined['tokens'].sum()}")
    df_combined.to_csv(output_file, index=False)

def batch_rows_by_token_count(combined_transcription_file, max_tokens=4096):
    df = pd.read_csv(combined_transcription_file)
    batches = []
    current_batch = []
    current_sum = 0
    for index, row in df.iterrows():
        if current_sum + row["tokens"] > max_tokens:
            cur_df = pd.DataFrame(current_batch)
            print(f"Batch {len(batches)}: {cur_df['tokens'].sum()}")
            batches.append(cur_df)
            current_batch = [row]
            current_sum = row["tokens"]
        else:
            current_batch.append(row)
            current_sum += row["tokens"]
    if current_batch:
        batches.append(pd.DataFrame(current_batch))
    print(f"Total batches: {len(batches)}")
    return batches

def chat_template_for_batches(combined_transcription_file, streamer_name, chat_dataset_file, max_tokens):
    batches = batch_rows_by_token_count(combined_transcription_file, max_tokens)
    response_template = lambda x: f"\n### Response:\n{x.strip()}" 
    input_template = lambda x: f"\n### Input:\n{x.strip()}" 
    chat_dataset = []
    for batch in batches:
        current_chat = []
        in_input = False
        for index, row in batch.iterrows():
            text = str(row["text"])
            if row["user_name"] == streamer_name:
                current_chat.append(response_template(text))
                in_input = False
            elif in_input:
                current_chat.append("\n" + text)
            else:
                current_chat.append(input_template(text))
                in_input = True
        chat_dataset.append("".join(current_chat))
    df = pd.DataFrame(chat_dataset, columns=["chat"])
    df.to_csv(chat_dataset_file, index=False)
        
if __name__ == "__main__":
    # prepare_audio("data/XL/")
    # isolate_voice("data/Toma/")
    # isolate_voice("data/XL/")
    # transcribe_voice("data/XL/")
    # batch_clips("data/Toma/resampled_audio/", "data/Toma/clips/")
    # transcribe_voice("data/Toma/clips/", "data/Toma/transcriptions/")
    # combine_transcriptions_csv("data/Toma/transcriptions/", "data/Toma/clips/")
    # combine_twitch_chat_with_streamer_transcription("data/Toma/full_transcription.csv", "data/Toma/twitch-chat-2025248149.csv", "data/Toma/combined_transcription.csv", "toma", 60)
    # chat_template_for_batches("data/Toma/combined_transcription.csv", "toma", "data/Toma/chat_dataset.csv", max_tokens=3000)
    pass
