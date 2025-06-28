import time
import wave
import queue
import struct
import threading
import subprocess

import pyaudio
import speech_recognition as sr
import pyttsx3

from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager

LANG = "EN" # CN for Chinese, EN for English
DEBUG = True

# Model Configuration
WHISP_PATH = "models/whisper-large-v3"
MODEL_PATH = "models/yi-34b-chat.Q8_0.gguf" # Or models/yi-chat-6b.Q8_0.gguf

# Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 500
SILENT_CHUNKS = 2 * RATE / CHUNK  # two seconds of silence marks the end of user voice input
MIC_IDX = 0 # Set microphone id. Use tools/list_microphones.py to see a device list.

def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i ** 2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=MIC_IDX, frames_per_buffer=CHUNK)

    silent_chunks = 0
    audio_started = False
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = compute_rms(data)
        if audio_started:
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            audio_started = True

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save audio to a WAV file
    with wave.open('recordings/output.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

class VoiceOutputCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.generated_text = ""
        self.lock = threading.Lock()
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.tts_busy = False
        self.engine = pyttsx3.init()

    def on_llm_new_token(self, token, **kwargs):
        with self.lock:
            self.generated_text += token
        if token in ['.', '。', '!', '！', '?', '？']:
            with self.lock:
                self.speech_queue.put(self.generated_text)
                self.generated_text = ""

    def process_queue(self):
        while True:
            text = self.speech_queue.get()
            if text is None:
                self.tts_busy = False
                continue
            self.tts_busy = True
            self.text_to_speech(text)
            self.speech_queue.task_done()
            if self.speech_queue.empty():
                self.tts_busy = False

    def text_to_speech(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")


if __name__ == '__main__':
    if LANG == "CN":
        prompt_path = "prompts/example-cn.txt"
    else:
        prompt_path = "prompts/example-en.txt"
    with open(prompt_path, 'r', encoding='utf-8') as file:
        template = file.read().strip() # {dialogue}
    prompt_template = PromptTemplate(template=template, input_variables=["dialogue"])

    # Create an instance of the VoiceOutputCallbackHandler
    voice_output_handler = VoiceOutputCallbackHandler()

    # Create a callback manager with the voice output handler
    callback_manager = BaseCallbackManager(handlers=[voice_output_handler])

    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=1, # Metal set to 1 is enough.
        n_batch=512,    # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        n_ctx=4096,     # Update the context window size to 4096
        f16_kv=True,    # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        stop=["<|im_end|>"],
        verbose=False,
    )
    dialogue = ""
    try:
        while True:
            if voice_output_handler.tts_busy:
                continue
            print("Listening...")
            record_audio()
            print("Transcribing...")
            time_ckpt = time.time()
            recognizer = sr.Recognizer()
            with sr.AudioFile("recordings/output.wav") as source:
                audio_data = recognizer.record(source)
            try:
                user_input = recognizer.recognize_google(audio_data, language='en-US' if LANG == "EN" else 'zh-CN')
            except sr.UnknownValueError:
                print("Could not understand audio, please try again")
                continue
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                continue
            print("%s: %s (Time %d ms)" % ("Guest", user_input, (time.time() - time_ckpt) * 1000))
            time_ckpt = time.time()
            print("Generating...")
            dialogue += "*Q* {}\n".format(user_input)
            prompt = prompt_template.format(dialogue=dialogue)
            reply = llm(prompt, max_tokens=4096)
            if reply is not None:
                voice_output_handler.speech_queue.put(None)
                dialogue += "*A* {}\n".format(reply)
                print("%s: %s (Time %d ms)" % ("Server", reply.strip(), (time.time() - time_ckpt) * 1000))
    except KeyboardInterrupt:
        pass
