# video_tranquitor.py
import reflex as rx
import speech_recognition as sr
import moviepy.editor as mp
from pydub import AudioSegment
import os
import json
from datetime import datetime
from typing import List, Dict

class State(rx.State):
    """Estado de la aplicación."""
    processing: bool = False
    transcription: List[Dict] = []
    error: str = ""
    progress: int = 0
    uploading: bool = False
    has_transcription: bool = False
    json_url: str = ""
    text_url: str = ""
    
    def create_download_urls(self):
        """Crea las URLs de descarga."""
        if not self.has_transcription:
            self.json_url = ""
            self.text_url = ""
            return
            
        # Crear URL para JSON
        json_content = json.dumps(self.transcription, ensure_ascii=False, indent=2)
        self.json_url = f"data:application/json;charset=utf-8,{json_content}"
        
        # Crear URL para texto
        text_content = ""
        for segment in self.transcription:
            text_content += f"[{segment['inicio']} - {segment['fin']}]\n{segment['texto']}\n\n"
        self.text_url = f"data:text/plain;charset=utf-8,{text_content}"
    
    async def handle_upload(self, files: list[rx.UploadFile]):
        """Maneja la subida de archivos."""
        if not files:
            return

        try:
            file = files[0]
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, file.filename)
            upload_data = await file.read()
            
            # Guardar el archivo
            with open(file_path, "wb") as f:
                f.write(upload_data)
            
            # Procesar el archivo
            await self.process_file(file_path, file.filename.lower().endswith(('.mp4', '.avi', '.mov')))
            
        except Exception as e:
            self.error = f"Error al subir el archivo: {str(e)}"
    
    def handle_upload_progress(self, progress: dict):
        """Maneja el progreso de la carga."""
        self.uploading = True
        self.progress = round(progress["progress"] * 100)
        if self.progress >= 100:
            self.uploading = False
    
    async def process_file(self, file_path: str, is_video: bool):
        """Procesa el archivo subido."""
        self.processing = True
        self.error = ""
        self.transcription = []
        self.has_transcription = False
        self.json_url = ""
        self.text_url = ""
        
        temp_audio_path = None
        temp_wav_path = None
        
        try:
            temp_audio_path = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            temp_wav_path = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            if is_video:
                video = mp.VideoFileClip(file_path)
                video.audio.write_audiofile(temp_audio_path)
                convert_audio_format(temp_audio_path, temp_wav_path)
            else:
                convert_audio_format(file_path, temp_wav_path)
            
            result = transcribe_audio_to_text(temp_wav_path)
            self.transcription = result
            self.has_transcription = True
            self.create_download_urls()
            
        except Exception as e:
            self.error = f"Error durante el procesamiento: {str(e)}"
        finally:
            self.processing = False
            # Limpieza de archivos
            for path in [temp_audio_path, temp_wav_path, file_path]:
                if path and os.path.exists(path):
                    os.remove(path)

def convert_audio_format(input_audio_path: str, output_audio_path: str):
    """Convierte el formato de audio a WAV."""
    audio = AudioSegment.from_file(input_audio_path)
    audio.export(output_audio_path, format="wav")

def transcribe_audio_to_text(audio_path: str) -> List[Dict]:
    """Transcribe el audio a texto."""
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(audio_path)
    chunk_length_ms = 30000
    
    chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    transcription = []
    
    for i, chunk in enumerate(chunks):
        chunk_filename = f"chunk_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.wav"
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

def index():
    return rx.center(
        rx.vstack(
            rx.heading("Transcriptor de Audio/Video", size="lg"),
            rx.text("Sube un archivo de audio o video para transcribirlo a texto."),
            rx.upload(
                rx.vstack(
                    rx.button(
                        "Seleccionar archivo",
                        color="rgb(107,99,246)",
                        bg="white",
                        border="1px solid rgb(107,99,246)",
                    ),
                    rx.text(
                        "O arrastra y suelta archivos aquí",
                        color="rgb(107,99,246)",
                    ),
                    padding="4",
                ),
                id="upload_audio",
                multiple=False,
                accept={
                    "audio/*": [".mp3", ".wav", ".ogg"],
                    "video/*": [".mp4", ".avi", ".mov"]
                },
                on_drop=State.handle_upload(
                    rx.upload_files(
                        upload_id="upload_audio",
                        on_upload_progress=State.handle_upload_progress,
                    )
                ),
                border="1px dashed rgb(107,99,246)",
                border_radius="md",
                padding="4",
            ),
            rx.cond(
                State.uploading | State.processing,
                rx.vstack(
                    rx.progress(value=State.progress),
                    rx.text(f"{State.progress}%"),
                ),
            ),
            rx.cond(
                State.error != "",
                rx.box(
                    rx.text(State.error, color="red"),
                    padding="2",
                    margin_top="2",
                ),
            ),
            rx.cond(
                State.has_transcription,
                rx.vstack(
                    rx.hstack(
                        rx.link(
                            rx.button(
                                "Descargar JSON",
                                color_scheme="green",
                            ),
                            href=State.json_url,
                            download="transcripcion.json",
                        ),
                        rx.link(
                            rx.button(
                                "Descargar TXT",
                                color_scheme="blue",
                            ),
                            href=State.text_url,
                            download="transcripcion.txt",
                        ),
                        spacing="4",
                    ),
                    rx.box(
                        rx.heading("Transcripción:", size="md", margin_top="4"),
                        rx.vstack(
                            rx.foreach(
                                State.transcription,
                                lambda segment: rx.box(
                                    rx.hstack(
                                        rx.text(f"{segment['inicio']} - {segment['fin']}", font_weight="bold"),
                                        rx.text(segment["texto"]),
                                    ),
                                    padding="2",
                                    border="1px solid",
                                    border_color="gray.200",
                                    border_radius="md",
                                    margin_y="1",
                                    width="100%",
                                ),
                            ),
                            align_items="stretch",
                            width="100%",
                        ),
                    ),
                ),
            ),
            padding="4",
            max_width="800px",
            width="100%",
            spacing="4",
        ),
    )

# Configuración de la aplicación
app = rx.App()
app.add_page(index)