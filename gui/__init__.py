import tkinter as tk
from tkinter import simpledialog
import threading
import websocket
import json
import pyaudio

class AudioSynthesizerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Text to Speech")

        self.start_button = tk.Button(self.root, text="Synthesize and Play", command=self.start_synthesis)
        self.start_button.pack(pady=10)

        self.audio_stream = None
        self.ws = websocket.create_connection("ws://localhost:8765")

    def start_synthesis(self):
        text = simpledialog.askstring("Input", "Enter text to synthesize")
        if text:
            self.ws.send(json.dumps({'text': text}))
            threading.Thread(target=self.receive_and_play, daemon=True).start()

    def receive_and_play(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)

        while True:
            audio_chunk = self.ws.recv()
            if audio_chunk:
                stream.write(audio_chunk)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AudioSynthesizerGUI()
    app.run()
