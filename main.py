import os
import moviepy.editor as mp
from pydub import AudioSegment
import speech_recognition as sr

def extract_audio_from_video(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def convert_audio_format(input_audio_path, output_audio_path):
    audio = AudioSegment.from_file(input_audio_path)
    audio.export(output_audio_path, format="wav")

def transcribe_audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="es-ES")
        return text
    except sr.UnknownValueError:
        return "No se pudo entender el audio"
    except sr.RequestError:
        return "Error al solicitar resultados del servicio de reconocimiento de voz"

def save_text_to_file(text, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

def main(video_path):
    base_path = os.path.dirname(video_path)
    temp_audio_path = os.path.join(base_path, "temp_audio.wav")
    output_text_path = os.path.join(base_path, "transcripcion.txt")
    
    extract_audio_from_video(video_path, temp_audio_path)
    convert_audio_format(temp_audio_path, temp_audio_path)
    text = transcribe_audio_to_text(temp_audio_path)
    save_text_to_file(text, output_text_path)

if __name__ == "__main__":
    video_path = "C:/Users/banar/Videos/2024-10-09 08-17-47.mkv"
    main(video_path)