import traceback
import speech_recognition as sr
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading
import websocket
import pyaudio
import discord
import asyncio
import logging

logger = logging.getLogger(__name__)


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI")

        self.ws = websocket.WebSocket()
        self.ws.connect("ws://localhost:8765")

        self.recording_thread = threading.Thread(target=self.record_and_send, daemon=True)
        self.recording_thread.start()

        self.playback_thread = threading.Thread(target=self.receive_and_play, daemon=True)
        self.playback_thread.start()

    async def poll_voice_ws(self, reconnect: bool) -> None:
        backoff = discord.backoff.ExponentialBackoff()
        while True:
            try:
                await self.ws.poll_event()
            except (discord.errors.ConnectionClosed, asyncio.TimeoutError) as exc:
                if isinstance(exc, discord.errors.ConnectionClosed):
                    # The following close codes are undocumented so I will document them here.
                    # 1000 - normal closure (obviously)
                    # 4014 - voice channel has been deleted.
                    # 4015 - voice server has crashed
                    if exc.code in (1000, 4015):
                        logger.info('Disconnecting from voice normally, close code %d.', exc.code)
                        await self.disconnect()
                        break
                    if exc.code == 4014:
                        logger.info('Disconnected from voice by force... potentially reconnecting.')
                        successful = await self.potential_reconnect()
                        if not successful:
                            logger.info('Reconnect was unsuccessful, disconnecting from voice normally...')
                            await self.disconnect()
                            break
                        else:
                            continue

                if not reconnect:
                    await self.disconnect()
                    raise

                retry = backoff.delay()
                logger.exception('Disconnected from voice... Reconnecting in %.2fs.', retry)
                self._connected.clear()
                await asyncio.sleep(retry)
                await self.voice_disconnect()
                try:
                    await self.connect(reconnect=True, timeout=self.timeout)
                except asyncio.TimeoutError:
                    # at this point we've retried 5 times... let's continue the loop.
                    logger.warning('Could not connect to voice... Retrying...')
                    continue 

    def record_and_send(self):
        recorder = sr.Recognizer()
        recorder.energy_threshold = 1000
        recorder.dynamic_energy_threshold = False
        record_timeout = 2
        phrase_timeout = 3
        source = sr.Microphone(sample_rate=16000)
        data_queue = Queue()

        with source:
            recorder.adjust_for_ambient_noise(source)

        def record_callback(_, audio: sr.AudioData) -> None:
            # Grab the raw bytes and push it into the thread safe queue.
            data = audio.get_raw_data()
            data_queue.put(data)

        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

        phrase_time = None

        while True:
            try:
                now = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                if not data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    phrase_time = now

                    # Combine audio data from queue
                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()

                    if phrase_complete:
                        # Send the audio data to the SpeechRecognizer to be transcribed.
                        self.ws.send_binary(audio_data)
                        self.ws.send("end_of_audio")
                        print(f"Sent audio data {audio_data}")

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
            except KeyboardInterrupt:
                self.ws.close()
                print(f"Interrupted: {e}")
                break
            except Exception as e:
                self.ws.close()
                print(f"Stopped recording: {e}")
                break

    def receive_and_play(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)

        count = 0
        try:
            while True:
                audio_chunk = self.ws.recv()
                print(f"Received {len(audio_chunk)} bytes")
                count += 1
                if audio_chunk:
                    stream.write(audio_chunk)
        except Exception as e:
            print(f"Stopped playback: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
        print(f"Played {count} chunks")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = GUI()
    app.run()
