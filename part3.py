import wave
import time
import threading
import tkinter as tk
import pyaudio
from PIL import ImageTk
import pickle
from part1 import extract_feature

audio = pyaudio.PyAudio()


class VoiceRecorder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Recorder")
        self.root.resizable(False, False)
        self.root.iconbitmap('app_icon.ico')
        microphone_photo = ImageTk.PhotoImage(file="mic.ico")
        self.button = tk.Button(image=microphone_photo, command=self.get_audio)
        self.button.pack(pady=10)
        self.label = tk.Label(text='00:00:00')
        self.label.pack()
        self.recording = False
        self.root.mainloop()

    def get_audio(self):
        if self.recording:
            self.recording = False
            self.button.config(fg="black")
        else:
            self.recording = True
            self.button.config(fg="red")
            threading.Thread(target=self.record).start()

    def record(self):
        pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

        frames = []

        start = time.time()

        while self.recording:
            data = stream.read(1024)
            frames.append(data)
            passed = time.time() - start
            secs = passed % 60
            minis = passed // 60
            hours = minis // 60
            self.label.config(text=f"{int(hours):02d}:{int(minis):02d}:{int(secs):02d}")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        def path_file(path):
            sound_file = wave.open(path, "wb")
            sound_file.setnchannels(1)
            sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            sound_file.setframerate(44100)
            sound_file.writeframes(b"".join(frames))
            sound_file.close()

        if __name__ == "__main__":
            model = pickle.load(open("mlp_classifier.model", "rb"))
            filename = "Last_voice.wav"
            path_file(filename)
            features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)

            result = model.predict(features)[0]
            print("Emotion:", result)


VoiceRecorder()

