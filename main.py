from tts.piper_tts import PiperTTS
from tts.tts import AudioSynthesizerGUI

def main():
    gui = AudioSynthesizerGUI(PiperTTS(model_path="tts/models/en_US-libritts-high.onnx", config_path="tts/models/en_US-libritts-high.onnx.json"))
    gui.run()

def listen():
    pass

def think():
    pass
    
def respond():
    pass

if __name__ == "__main__":
    main()