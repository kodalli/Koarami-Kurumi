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
import json


def listener():
    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = "pulse"
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(device_index=1, sample_rate=16000)

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    record_timeout = 2
    phrase_timeout = 3

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Listening...\n")

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

                if phrase_complete:
                    # Combine audio data from queue
                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()
                    print(audio_data)
                    # TODO: Send the audio data to the SpeechRecognizer to be transcribed.

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI")

        self.listen_button = tk.Button(self.root, text="Start Recording", command=self.toggle_recording)
        self.listen_button.pack(pady=10)

        self.ws = websocket.WebSocket()
        self.ws.connect("ws://localhost:8765")

        self.recording = False
        self.recording_thread = None

        # Start the receive and play thread immediately
        self.playback_thread = threading.Thread(target=self.receive_and_play, daemon=True)
        self.playback_thread.start()

    def toggle_recording(self):
        if self.recording:
            # Stop recording
            self.recording = False
            self.listen_button.config(text="Start Recording")
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join()
        else:
            # Start recording
            self.recording = True
            self.listen_button.config(text="Stop Recording")
            self.recording_thread = threading.Thread(target=self.record_and_send, daemon=True)
            self.recording_thread.start()

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

        while self.recording:
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
                        break

                    # Infinite loops are bad for processors, must sleep.
                    sleep(0.25)
            except Exception as e:
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
