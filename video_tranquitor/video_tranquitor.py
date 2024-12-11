import reflex as rx
import speech_recognition as sr
import moviepy.editor as mp
from pydub import AudioSegment
import os
import json
from datetime import datetime
import time
import shutil
from typing import List, Dict
import whisper
import traceback 
import numpy as np
import torch
import asyncio
from typing import List
import logging

# Configurar el registro
logging.basicConfig(level=logging.INFO)

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
    processing_stage: str = ""
    total_duration: float = 0
    current_time: float = 0
    estimated_time: float = 0
    cancel_processing: bool = False
    start_time: float = 0
    
    def set_progress(self, current: float, total: float):
        """Actualiza el progreso y estima el tiempo restante."""
        if total > 0:
            self.progress = min(100, int((current / total) * 100))
            if current > 0:
                elapsed_time = time.time() - self.start_time
                time_per_unit = elapsed_time / current
                remaining_units = total - current
                self.estimated_time = (remaining_units * time_per_unit) / 60  # en minutos
    
    def handle_upload_progress(self, progress: dict):
        """Actualiza el progreso de la subida del archivo."""
        self.uploading = True
        self.progress = int(progress.get("progress", 0) * 100)
        if self.progress >= 100:
            self.uploading = False
    
    def cancel_process(self):
        """Cancela el procesamiento actual."""
        self.cancel_processing = True
        self.processing = False
        self.error = "Procesamiento cancelado por el usuario"
    
    def create_download_urls(self):
        """Crea las URLs para descargar los archivos de transcripción."""
        if not self.has_transcription:
            self.json_url = ""
            self.text_url = ""
            return
            
        json_content = json.dumps(self.transcription, ensure_ascii=False, indent=2)
        self.json_url = f"data:application/json;charset=utf-8,{json_content}"
        
        text_content = ""
        for segment in self.transcription:
            text_content += f"[{segment['inicio']} - {segment['fin']}]\n{segment['texto']}\n\n"
        self.text_url = f"data:text/plain;charset=utf-8,{text_content}"
    
    async def handle_upload(self, files: List[rx.UploadFile]):
        """Maneja la subida de archivos."""
        if not files:
            return

        try:
            file = files[0]
            # Verificar tamaño máximo (100MB)
            file_data = await file.read()
            if len(file_data) > 100 * 1024 * 1024:
                self.error = "El archivo es demasiado grande. Máximo 100MB."
                return
            
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, file.filename)
            
            with open(file_path, "wb") as f:
                f.write(file_data)
            
            self.start_time = time.time()
            self.cancel_processing = False
            await self.process_file(file_path, file.filename.lower().endswith(('.mp4', '.avi', '.mov')))
            
        except Exception as e:
            print(f"Error en handle_upload: {str(e)}")
            self.error = f"Error al subir el archivo: {str(e)}"
    
    async def process_file(self, file_path: str, is_video: bool):
        """Procesa el archivo subido con timeout."""
        video = None
        temp_dir = None
        try:
            # Establecer timeout de 15 minutos
            try:
                await asyncio.wait_for(self._process_file_work(file_path, is_video), timeout=900)
            except asyncio.TimeoutError:
                self.error = "El procesamiento excedió el tiempo límite (15 minutos)"
        except Exception as e:
            self.error = f"Error durante el procesamiento: {str(e)}"
        finally:
            self.processing = False
            self.processing_stage = ""
            self.progress = 0

    async def _process_file_work(self, file_path: str, is_video: bool):
        """Trabajo real de procesamiento del archivo."""
        video = None
        temp_dir = None
        try:
            if not os.path.exists(file_path):
                raise Exception(f"El archivo no existe: {file_path}")
            
            self.processing = True
            self.error = ""
            self.transcription = []
            self.has_transcription = False
            self.progress = 0
            
            temp_dir = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_audio_path = os.path.join(temp_dir, "temp_audio.mp3")
            temp_wav_path = os.path.join(temp_dir, "temp_audio.wav")
            
            try:
                if is_video:
                    self.processing_stage = "Extrayendo audio del video..."
                    video = mp.VideoFileClip(file_path)
                    self.total_duration = video.duration
                    
                    video.audio.write_audiofile(
                        temp_audio_path,
                        verbose=False,
                        logger=None
                    )
                    convert_audio_format(temp_audio_path, temp_wav_path, self)
                else:
                    self.processing_stage = "Preparando archivo de audio..."
                    convert_audio_format(file_path, temp_wav_path, self)
                
                if self.cancel_processing:
                    return
                
                self.processing_stage = "Transcribiendo audio..."
                result = await asyncio.get_event_loop().run_in_executor(None, transcribe_audio_with_whisper, temp_wav_path, self)
                
                if result:
                    self.transcription = result
                    self.has_transcription = True
                    self.create_download_urls()
            
            except Exception as e:
                raise Exception(f"Error en el procesamiento: {str(e)}")
            
        finally:
            if video is not None:
                video.close()
            
            # Limpiar archivos temporales
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Error al limpiar archivos temporales: {e}")
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error al eliminar archivo original: {e}")

def convert_audio_format(input_audio_path: str, output_audio_path: str, state: State = None):
    """Convierte y optimiza el formato de audio."""
    try:
        audio = AudioSegment.from_file(input_audio_path)
        
        # Optimizaciones
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio = audio.normalize()
        audio = audio.set_sample_width(2)  # 16-bit audio
        audio = audio.low_pass_filter(8000)
        
        if state:
            state.total_duration = len(audio) / 1000.0
            state.set_progress(0, state.total_duration)
        
        audio.export(
            output_audio_path,
            format="wav",
            parameters=["-ar", "16000", "-ac", "1", "-b:a", "64k"]
        )
    except Exception as e:
        raise Exception(f"Error en la conversión del audio: {str(e)}")

def format_timestamp(seconds: float) -> str:
    """Formatea los segundos en formato HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def transcribe_audio_with_whisper(audio_path: str, state: State) -> List[Dict]:
    """Transcribe el audio usando Whisper con optimizaciones."""
    try:
        if not os.path.exists(audio_path):
            raise Exception(f"El archivo de audio no existe: {audio_path}")

        state.processing_stage = "Cargando modelo de transcripción..."
        model = whisper.load_model("tiny")

        segments_dir = os.path.join(os.path.dirname(audio_path), "segments")
        os.makedirs(segments_dir, exist_ok=True)
        audio_segments = split_audio(audio_path, segment_length=60, output_dir=segments_dir)

        all_transcriptions = []
        total_segments = len(audio_segments)

        for i, segment_path in enumerate(audio_segments):
            if state.cancel_processing:
                break

            state.processing_stage = f"Transcribiendo parte {i+1} de {total_segments}..."
            state.set_progress(i+1, total_segments)

            try:
                result = model.transcribe(
                    segment_path,
                    language="es",
                    task="transcribe",
                    fp16=False,
                    temperature=0.0,
                    no_speech_threshold=0.3,
                )

                if result and isinstance(result, dict):
                    time_offset = i * 60
                    text = result.get("text", "").strip()
                    segments = result.get("segments", [])

                    if not segments and text:
                        segment_data = {
                            "inicio": format_timestamp(time_offset),
                            "fin": format_timestamp(time_offset + 60),
                            "texto": text
                        }
                        all_transcriptions.append(segment_data)
                    else:
                        for seg in segments:
                            segment_data = {
                                "inicio": format_timestamp(seg["start"] + time_offset),
                                "fin": format_timestamp(seg["end"] + time_offset),
                                "texto": seg["text"].strip()
                            }
                            if segment_data["texto"]:
                                all_transcriptions.append(segment_data)

            except Exception as e:
                print(f"Error en segmento {i+1}: {str(e)}")
                continue

        try:
            shutil.rmtree(segments_dir)
        except Exception as e:
            print(f"Error al limpiar segmentos: {e}")

        return all_transcriptions if all_transcriptions else None

    except Exception as e:
        raise Exception(f"Error en la transcripción con Whisper: {str(e)}")

def split_audio(audio_path: str, segment_length: int = 60, output_dir: str = None) -> List[str]:
    """Divide el audio en segmentos más pequeños."""
    if output_dir is None:
        output_dir = os.path.dirname(audio_path)
    
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio)
    segments = []
    
    audio = audio.normalize()
    
    for i, start in enumerate(range(0, duration, segment_length * 1000)):
        end = min(start + segment_length * 1000, duration)
        segment = audio[start:end]
        
        segment = segment.set_channels(1)
        segment = segment.set_frame_rate(16000)
        
        segment_path = os.path.join(output_dir, f"segment_{i}.wav")
        segment.export(
            segment_path,
            format="wav",
            parameters=["-ar", "16000", "-ac", "1", "-b:a", "64k"]
        )
        segments.append(segment_path)
    
    return segments

def loading_spinner():
    """Componente de spinner de carga con estimación de tiempo."""
    return rx.vstack(
        rx.spinner(
            color="rgb(107,99,246)",
            size="xl",
            thickness="4px",
            speed="0.8s",
        ),
        rx.heading(
            State.processing_stage,
            size="md",
            color="rgb(107,99,246)",
        ),
        rx.progress(
            value=State.progress,
            width="100%",
        ),
        rx.text(
            f"{State.progress}% completado",
            color="rgb(107,99,246)",
            font_size="sm",
        ),
        rx.text(
            f"Tiempo estimado restante: {State.estimated_time:.1f} minutos",
            color="gray.500",
            font_size="sm",
        ),
        rx.button(
            "Cancelar procesamiento",
            on_click=State.cancel_process,
            color_scheme="red",
            size="sm",
            margin_top="2",
        ),
        align_items="center",
        justify_content="center",
        background="white",
        padding="8",
        border_radius="lg",
        shadow="lg",
        width="100%",
        spacing="4",
    )

def index():
    """Página principal de la aplicación."""
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
                on_drop=State.handle_upload(rx.upload_files("upload_audio")),
                border="1px dashed rgb(107,99,246)",
                border_radius="md",
                padding="4",
            ),
            rx.cond(
                State.uploading,
                rx.vstack(
                    rx.progress(value=State.progress),
                    rx.text(f"Subiendo archivo: {State.progress}%"),
                ),
            ),
            rx.cond(
                State.processing,
                loading_spinner(),
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

app = rx.App()
app.add_page(index)