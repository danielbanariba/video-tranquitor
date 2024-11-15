import speech_recognition as sr
import moviepy.editor as mp
from pydub import AudioSegment
import math
import os
import json

def extract_audio_from_video(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def convert_audio_format(input_audio_path, output_audio_path):
    audio = AudioSegment.from_file(input_audio_path)
    audio.export(output_audio_path, format="wav")

def transcribe_audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(audio_path)
    chunk_length_ms = 30000  # Duración de cada segmento en milisegundos (por ejemplo, 30 segundos)

    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    transcription = []
    for i, chunk in enumerate(chunks):
        chunk_filename = f"chunk{i}.wav"
        chunk.export(chunk_filename, format="wav")

        with sr.AudioFile(chunk_filename) as source:
            audio_listened = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_listened, language="es-ES")
        except sr.UnknownValueError:
            text = "No se pudo entender el audio"
        except sr.RequestError:
            text = "Error al solicitar resultados del servicio de reconocimiento de voz"

        start_time = i * chunk_length_ms / 1000
        end_time = start_time + len(chunk) / 1000

        segment = {
            "inicio": f"{int(start_time//3600):02}:{int((start_time%3600)//60):02}:{int(start_time%60):02}",
            "fin": f"{int(end_time//3600):02}:{int((end_time%3600)//60):02}:{int(end_time%60):02}",
            "texto": text
        }
        transcription.append(segment)

        os.remove(chunk_filename)

    return transcription

def save_text_to_file(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(text, f, ensure_ascii=False, indent=4)

def main(video_path):
    temp_audio_path = "temp_audio.mp3"
    temp_wav_path = "temp_audio.wav"
    output_text_path = "output/transcription.json"

    if not os.path.exists("output"):
        os.makedirs("output")

    extract_audio_from_video(video_path, temp_audio_path)
    convert_audio_format(temp_audio_path, temp_wav_path)
    text = transcribe_audio_to_text(temp_wav_path)
    save_text_to_file(text, output_text_path)

    os.remove(temp_audio_path)
    os.remove(temp_wav_path)

if __name__ == "__main__":
    video_path = "C:/Users/banar/Videos/2024-10-09 08-17-47.mkv"
    main(video_path)