# animorph PoC for smooth audio handling

# for gcloud access you need to run the following two commands in the terminal first
# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
# gcloud init
# gcloud auth application-default login
# this will open a browser to chose your account and give the terminal access to your gcloud account

# run in terminal to install the library
# pip install -q google-cloud-speech
# pip install --upgrade google-cloud-texttospeech
# pip install -q -U google-generativeai
# pip install IPython
# pip install pyaudio
# pip install pydub
# pip install soundfile librosa
# pip install audioread scipy
# 
import pathlib
from google.cloud import speech
import google.generativeai as genai
import os


# identify if a person speaks and cut exactly this part of the audio
import pyaudio
import numpy as np
import wave
import sys
import io
import time

# time metrics in milliseconds
timeToGenerateSpeechText = 0
timeToGenerateSpeechAudio = 0
timeForSpeeching = 0
# first latency results: 
# response with audio input about 2 secs
# one sentence audio generation 0.8 sec
# response with text only 1.6 sec?

# Parameter für den Audiostream
CHUNK = 1024        # Anzahl der Samples pro Frame
RATE = 16000        # Abtastrate in Hz

# PyAudio-Stream starten
p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Speech Analyse läuft. Drücke STRG+C zum Beenden.")

# the speech detection logic will work as follows
# 1. set speech_status to false, speech_counter to 0, silent_counter to 0, recording to false
# 2. if volume > volume_threshold: increment speech_counter, silent_counter = 0
#    else: increment silent_counter, speech_counter = 0 
# 4. if speech_counter == min_chunks_for_start_speech: speech_status = true
# 5. if silent_counter == min_chunks_for_end_speech and speech_status = true: speech_status = false, store recorded stream, recording = false
# 6. if volume > volume_threshold and recording == false: start recording stream, recording = true
# 7. if volume < volume_threshold and speech_status == false: stop recording stream, recording = false

volume_threshold = 15
min_chunks_for_start_speech = 15
min_chunks_for_end_speech = 15

speech_status = False
speech_counter = 0
silent_counter = 0
recording = False

try:
    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)  # Daten lesen und in numpy Array umwandeln
        volume = np.sqrt(np.mean(data**2))  # Root Mean Square (RMS) für Lautstärke
        #print(f"Volume: {volume}")
        if volume > volume_threshold:
            speech_counter += 1
            silent_counter = 0
        else:
            silent_counter += 1
            speech_counter = 0
        
        if speech_counter == min_chunks_for_start_speech:   # start of speech detected
            speech_status = True
            print("Start of speech detected.")

        if silent_counter == min_chunks_for_end_speech and speech_status:   # end of speech detected
            speech_status = False
            recording = False
            print("End of speech detected. Recording is stored in file")
            # store recorded stream in file  
            wf = wave.open("audiance.wav", 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            break

        if volume > volume_threshold and not recording: # start recording
            recording = True
            # initializing the recorded frames
            frames = []
            print("Start fresh recording.")

        if volume < volume_threshold and not speech_status: # silent phase
            recording = False

        if recording:
            frames.append(data.tobytes())

except KeyboardInterrupt:
    print("\nAnalyse beendet.")

# Stream stoppen und schließen
stream.stop_stream()
stream.close()
p.terminate()




# use gemeini flash to transcribe/send the audio - this would enable us to directly transmit 
# the audio together with the system prompt and the history of the conversation

genai.configure(api_key="<your API key for google gemini>")

model = genai.GenerativeModel("gemini-1.5-flash")

# Create the prompt.
prompt = "You are Plato and react to the user input accordingly"

# start time measurement for timeToGenerateSpeechText
start = time.time()

# Load the audiance.wav file into a Python Blob object containing the audio
# file's bytes and then pass the prompt and the audio to Gemini.
# response = model.generate_content([prompt])
response = model.generate_content([
    prompt,
    {
        "mime_type": "audio/wav",
        "data": pathlib.Path('audiance.wav').read_bytes()
    }
])

# end time measurement for timeToGenerateSpeechText
end = time.time()
timeToGenerateSpeechText = end - start
print(f"Time to generate speech text: {timeToGenerateSpeechText}")

# Print the transcript.
print(response.text)

"""Synthesizes speech from the input string of text or ssml.
Make sure to be working in a virtual environment.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""
from google.cloud import texttospeech

# Instantiates a client
client = texttospeech.TextToSpeechClient()

# Set the text input to be synthesized
synthesis_input = texttospeech.SynthesisInput(text=response.text)

# Build the voice request, select the Journey voice as the voice type
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US", name="en-US-Journey-D"
)
voice = texttospeech.VoiceSelectionParams(
    language_code="de-DE", name="de-DE-Journey-D"
)

# Select the type of audio file you want returned
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16
)

# start time measurement for timeToGenerateSpeechAudio
start = time.time()

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(
    input=synthesis_input, voice=voice, audio_config=audio_config
)

# end time measurement for timeToGenerateSpeechAudio
end = time.time()
timeToGenerateSpeechAudio = end - start
print(f"Time to generate speech audio: {timeToGenerateSpeechAudio}")

audio_content = response.audio_content

# Create an in-memory bytes buffer
audio_buffer = io.BytesIO(audio_content)

# start time measurement for timeForSpeeching
start = time.time()

# Open the WAV audio from the buffer
with wave.open(audio_buffer, 'rb') as wf:
    # Set up the PyAudio stream
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    
    # Read and play the audio content in chunks
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

# end time measurement for timeForSpeeching
end = time.time()
timeForSpeeching = end - start
print(f"Time for speeching: {timeForSpeeching}")

# store audio content as wav file
with open("output.wav", "wb") as out:
    out.write(response.audio_content)

print("WAV file has been created successfully.")