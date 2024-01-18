from datetime import datetime, timedelta
import io
import queue
import threading
import time
from typing import Any, Optional
import discord
from discord.ext import listening
import asyncio
import os
import librosa
import pyaudio
import resampy
from pydub import AudioSegment
import opuslib
import numpy as np
import logging
import dotenv
from src.tts.piper_tts import PiperTTS
from src.stt.whisper_stt import WhisperSTT
from src.stt.stt import SpeechToText
from src.llm.inference.deciLM_7b import DeciLM7b

logger = logging.getLogger(__name__)
dotenv.load_dotenv()

# pool that will be used for processing audio
# 1 signifies having 1 process in the pool
process_pool = listening.AudioProcessPool(1)

# Maps a file format to a sink object
# FILE_FORMATS = {"mp3": listening.MP3AudioFile, "wav": listening.WaveAudioFile}

class AudioBuffer:
    def __init__(self, client):
        self._clean_lock = threading.Lock()
        self._last_timestamp: Optional[int] = None
        self._last_sequence: Optional[int] = None
        self._packet_count = 0
        self.buffer = io.BytesIO()
        self.done: bool = False
        self.phrase_time = datetime.utcnow()
        self.client = client
        self.audio_threshold = 300

    def is_silence(self, audio_data) -> bool:
        rms_value = np.sqrt(np.mean(np.square(audio_data)))
        print(f"RMS: {rms_value}")
        return rms_value < self.audio_threshold

    def on_audio(self, frame: listening.AudioFrame) -> None:
        """
        Parameters
        ----------
        frame: :class:`AudioFrame`
            The frame which will be added to the buffer.
        """
        self._clean_lock.acquire()
        if self.done:
            return
        if self._packet_count < 7:
            self._packet_count += 1
        self._write_frame(frame)
        self._clean_lock.release()

    def _write_frame(self, frame: listening.AudioFrame) -> None:
        # When the bot joins a vc and starts listening and a user speaks for the first time,
        # the timestamp encompasses all that silence, including silence before the bot even
        # joined the vc. It goes in a pattern that the 6th packet has a 11 sequence skip, so
        # this last part of the if statement gets rid of that silence.
        if self._last_timestamp is not None and not (self._packet_count == 6 and frame.sequence - self._last_sequence == 11):  # type: ignore
            silence = frame.timestamp - self._last_timestamp - listening.sink.OpusDecoder.SAMPLES_PER_FRAME
            if silence > 0:
                self.buffer.write(b"\x00" * silence * listening.sink.OpusDecoder.SAMPLE_SIZE)

        if frame.audio != listening.sink.SILENT_FRAME:
            self.buffer.write(frame.audio)

        now = datetime.utcnow()
        if  now - self.phrase_time > timedelta(seconds=3):
            raw_bytes = self.buffer.getvalue()
            print(len(raw_bytes))
            audio_np = np.frombuffer(raw_bytes, dtype=np.int16)
            audio_mono_np = audio_np.reshape((-1, listening.sink.OpusDecoder.CHANNELS)).mean(axis=1)
            audio_mono_resampled_np = resampy.resample(audio_mono_np, listening.sink.OpusDecoder.SAMPLING_RATE, 16000)
            if not self.is_silence(audio_mono_resampled_np):
                question = self.client.stt_engine.hear(audio_mono_resampled_np)
                if question.strip() != "":
                    print(f"Q: {question}")
                    response = self.client.llm_engine.think(question, max_new_tokens=4096)
                    if response.strip() != "":
                        print(f"A: {response}")
                        self.client.text = response
                        asyncio.run(self.client.voice_send())

            self.buffer.seek(0)
            self.buffer.truncate(0)
            self.phrase_time = now
            
        self._last_timestamp = frame.timestamp
        self._last_sequence = frame.sequence
        # self._cache_user(frame.user)

    def cleanup(self) -> None:
        """Writes remaining frames in buffer to file and then closes it."""
        self._clean_lock.acquire()
        if self.done:
            return
        self.file.close()
        self.done = True
        self._clean_lock.release()


class AudioHandler(listening.AudioHandlingSink):

    VALIDATION_WAIT_TIMEOUT = 1

    def __init__(self, message: discord.Message, client):
        super().__init__()
        self.message = message
        self._clean_lock: threading.Lock = threading.Lock()
        self.buffer = AudioBuffer(client)
        self.done: bool = False
    
    def on_valid_audio(self, frame: listening.AudioFrame):
        self._clean_lock.acquire()

        # frame.ssrc is the speaker, multiple speakers can be in a channel
        # print(f"Member: {frame.user}")
        self.buffer.on_audio(frame)
        self._clean_lock.release()

    def on_rtcp(self, packet: listening.RTCPPacket) -> None:
        """This function receives RTCP Packets, but does nothing with them since
        there is no use for them in this sink.

        Parameters
        ----------
        packet: :class:`RTCPPacket`
            A RTCP Packet received from discord. Can be any of the following:
            :class:`RTCPSenderReportPacket`, :class:`RTCPReceiverReportPacket`,
            :class:`RTCPSourceDescriptionPacket`, :class:`RTCPGoodbyePacket`,
            :class:`RTCPApplicationDefinedPacket`
        """
        return

    def cleanup(self) -> None:
        """Waits a maximum of `VALIDATION_WAIT_TIMEOUT` for packet validation to finish and
        then calls `cleanup` on all :class:`AudioFile` objects.

        Sets `done` to True after calling all the cleanup functions.
        """
        self._done_validating.wait(self.VALIDATION_WAIT_TIMEOUT)
        self._clean_lock.acquire()
        if self.done:
            return
        self.buffer.cleanup()
        self.done = True
        self._clean_lock.release()


class Client(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.token = os.getenv("DISCORD_TOKEN")
        self.guild = os.getenv("DISCORD_GUILD")
        self.channel = os.getenv("DISCORD_CHANNEL")
        self.base_dir = os.getenv("BASE_DIRECTORY")

        name = "en_US-kathleen-low.onnx"
        self.tts_engine = PiperTTS(model_path=f"{self.base_dir}/src/tts/models/piper/{name}", config_path=f"{self.base_dir}/src/tts/models/piper/{name}.json")
        self.llm_engine = DeciLM7b()
        self.stt_engine = WhisperSTT()
        self.text = None
        self.vc = None

        if not discord.opus.is_loaded():
            discord.opus.load_opus("libopus.so.0")

        self.stt_engine = WhisperSTT()

    async def send_audio_file(self, channel: discord.TextChannel, file: listening.AudioFile):
        # Get the user id of this audio file's user if possible
        # If it's not None, then it's either a `Member` or `Object` object, both of which have an `id` attribute.
        user = file.user if file.user is None else file.user.id

        # Send the file and if the file is too big (ValueError is raised) then send a message
        # saying the audio file was too big to send.
        try:
            await channel.send(
                f"Audio file for <@{user}>" if user is not None else "Could not resolve this file to a user...",
                file=discord.File(file.path),
            )
        except ValueError:
            await channel.send(
                f"Audio file for <@{user}> is too big to send"
                if user is not None
                else "Audio file for unknown user is too big to send"
            )

    # The key word arguments passed in the listen function MUST have the same name.
    # You could alternatively do on_listen_finish(sink, exc, channel, ...) because exc is always passed
    # regardless of if it's None or not.
    async def on_listen_finish(self, sink: listening.AudioFileSink, exc=None, channel=None, message=None):
        # Convert the raw recorded audio to its chosen file type
        # and wait for it to finish.
        sink.convert_files()
        await sink.wait_for_convert()
        if channel is not None:
            for file in sink.output_files.values():
                await self.send_audio_file(channel, file)

        audio_data = None
        for file in os.listdir("./"):
            if file.endswith(".wav"):
                audio_data = librosa.load(file, sr=16000, mono=True)[0].tobytes() 
                break
        text = self.stt_engine.hear(audio_data)
        await message.channel.send(text)

        # Raise any exceptions that may have occurred
        if exc is not None:
            raise exc

    async def on_ready(self):
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
    
    async def join(self, message: discord.Message):
        voice = message.author.voice
        if not voice:
            await message.channel.send("You are not connected to a voice channel!")
            return
        channel = voice.channel
        if not channel:
            await message.channel.send("You are not connected to a voice channel!")
            return

        self.vc = await channel.connect(cls=listening.VoiceClient)

        if not isinstance(self.vc, listening.VoiceClient):
            await message.channel.send("Failed to initialize the listening voice client!") 
            return

        await self.start_listening(message)

        
    async def start_listening(self, message: discord.Message):
        sink = AudioHandler(message, self)
        self.vc.listen(sink, process_pool, after=self.on_listen_finish, message=message)
        await message.channel.send(f"Connected to {message.author.voice.channel} and started listenging!")

    async def stop_listening(self, message: discord.Message):
        self.vc.stop_listening() 

    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.content.startswith("!leave"):
            if self.vc:
                await self.vc.disconnect()
                self.vc = None
        
        if message.content.startswith("!stop"):
            if self.vc:
                await self.stop_listening(message)

        if message.content.startswith("!start"):
            if self.vc:
                await self.start_listening(message)   

        if message.content.startswith("!join"):
            await self.join(message)

        if message.content.startswith("!tts"):
            print(message.content)
            self.text = message.content[4:]
            await self.voice_send()

    async def voice_send(self):
        if self.text and self.vc.is_playing() == False:
            mono_generator = self.tts_engine.say(self.text, length_scale=0.8, sentence_silence=0.3, noise_scale=0.3, noise_w=0.5)
            buffer = io.BytesIO()
            for mono_chunk in mono_generator:
                buffer.write(mono_chunk)
            buffer.seek(0)
            mono_audio = AudioSegment.from_file(buffer, format="raw", frame_rate=self.tts_engine.sample_rate(), channels=1, sample_width=2)
            stereo_audio = mono_audio.set_frame_rate(48000).set_channels(2)
            stereo_buffer = io.BytesIO()
            stereo_audio.export(stereo_buffer, format="raw")
            stereo_buffer.seek(0)

            pcm_audio = discord.PCMAudio(stereo_buffer)
            res = discord.PCMVolumeTransformer(pcm_audio, volume=0.2)

            self.vc.play(res)
            self.text = None


    def cleanup(self):
        pass

if __name__ == "__main__":
    client = Client()
    try:
        client.run(client.token)
    except KeyboardInterrupt:
        client.cleanup()

