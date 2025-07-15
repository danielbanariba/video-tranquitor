import speech_recognition as sr
import moviepy.editor as mp
from pydub import AudioSegment
import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Inicializar el cliente de OpenAI con la API key desde variables de entorno
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_audio_from_video(video_path, audio_path):
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        return True
    except KeyError as e:
        print(f"Error al extraer audio del video: {e}")
        return False
    except Exception as e:
        print(f"Error inesperado al extraer audio del video: {e}")
        return False

def convert_audio_format(input_audio_path, output_audio_path):
    try:
        audio = AudioSegment.from_file(input_audio_path)
        audio.export(output_audio_path, format="wav")
        return True
    except Exception as e:
        print(f"Error al convertir formato de audio: {e}")
        return False

def get_audio_duration(audio_path):
    """Obtener la duración del audio en segundos"""
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000  # Convertir de ms a segundos

def get_file_size_mb(file_path):
    """Obtener el tamaño del archivo en MB"""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def transcribe_with_openai(audio_path, language="es"):
    """Transcribe audio usando la API de OpenAI Whisper (nueva sintaxis)"""
    try:
        # Verificar tamaño del archivo
        file_size = get_file_size_mb(audio_path)
        if file_size > 24:  # Si es mayor a 24 MB, es probable que falle
            print(f"Advertencia: El archivo es de {file_size:.2f} MB, lo cual puede exceder el límite de la API")
        
        with open(audio_path, "rb") as audio_file:
            # Nueva sintaxis para OpenAI >= 1.0.0
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="json"
            )
        
        # Procesamos la respuesta
        text = ""
        if hasattr(response, 'text'):
            text = response.text.strip()
        else:
            text = str(response).strip()
            
        # Devolvemos el texto plano en lugar de un segmento formateado
        return text
    except Exception as e:
        print(f"Error en la transcripción con OpenAI: {e}")
        return ""

def format_time(seconds):
    """Formatea segundos a formato HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def transcribe_long_audio(audio_path, language="es", chunk_length_sec=120):
    """Transcribe audios largos dividiéndolos en segmentos de 2 minutos para mejor procesamiento"""
    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    chunk_length_ms = chunk_length_sec * 1000
    
    transcriptions = []  # Lista de diccionarios con tiempo de inicio, fin y texto
    
    # Crear directorio temporal si no existe
    if not os.path.exists("temp_chunks"):
        os.makedirs("temp_chunks")
    
    total_chunks = (duration_ms // chunk_length_ms) + 1
    
    for i in tqdm(range(0, duration_ms, chunk_length_ms), desc="Procesando audio", total=total_chunks):
        # Obtener segmento de audio
        chunk = audio[i:i + chunk_length_ms]
        chunk_filename = f"temp_chunks/chunk_{i}.mp3"
        
        # Exportar con calidad media para equilibrar tamaño y calidad
        chunk.export(chunk_filename, format="mp3", parameters=["-q:a", "6"])
        
        # Verificar tamaño del archivo
        file_size = get_file_size_mb(chunk_filename)
        
        # Transcribir segmento
        print(f"Transcribiendo segmento {i//chunk_length_ms + 1}/{total_chunks} (Tamaño: {file_size:.2f} MB)")
        text = transcribe_with_openai(chunk_filename, language)
        
        # Almacenar la transcripción con su tiempo de inicio y fin
        if text:
            start_time = i / 1000  # Convertir ms a segundos
            end_time = min((i + len(chunk)) / 1000, duration_ms / 1000)
            
            transcriptions.append({
                "inicio": format_time(start_time),
                "fin": format_time(end_time),
                "texto": text,
            })
        
        # Eliminar archivo temporal
        os.remove(chunk_filename)
        
        # Dormir un poco para no sobrecargar la API
        time.sleep(1)
    
    # Limpiar directorio temporal
    if os.path.exists("temp_chunks"):
        try:
            os.rmdir("temp_chunks")
        except:
            pass
            
    return transcriptions

def divide_segments_in_smaller_chunks(transcriptions, small_chunk_seconds=10):
    """Divide las transcripciones en segmentos más pequeños (10 segundos) para mejor procesamiento por IA"""
    small_segments = []
    
    for segment in transcriptions:
        # Convertir inicio y fin a segundos
        start_parts = segment["inicio"].split(":")
        start_seconds = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])
        
        end_parts = segment["fin"].split(":")
        end_seconds = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + int(end_parts[2])
        
        # Calcular duración
        duration = end_seconds - start_seconds
        
        # Si la duración es menor a nuestro chunk pequeño, mantenerlo igual
        if duration <= small_chunk_seconds:
            small_segments.append(segment)
            continue
        
        # Dividir texto (aproximadamente) basado en la longitud y los espacios
        texto = segment["texto"]
        
        # Número de segmentos pequeños que necesitamos
        num_small_chunks = duration // small_chunk_seconds
        if duration % small_chunk_seconds > 0:
            num_small_chunks += 1
        
        # Intentamos dividir el texto basado en palabras y puntuación
        # Primero separamos por puntos (oraciones)
        sentences = []
        current_sentence = ""
        for char in texto:
            current_sentence += char
            if char in '.!?':
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        if current_sentence:  # Si queda alguna oración final sin punto
            sentences.append(current_sentence.strip())
        
        # Si no hay oraciones (raro), o hay muy pocas, dividimos por espacios
        if len(sentences) < num_small_chunks / 2:
            words = texto.split()
            approx_words_per_chunk = max(1, len(words) // num_small_chunks)
            
            sentences = []
            for i in range(0, len(words), approx_words_per_chunk):
                word_group = words[i:i + approx_words_per_chunk]
                sentences.append(" ".join(word_group))
        
        # Ahora distribuimos estas oraciones en los segmentos pequeños
        for i in range(int(num_small_chunks)):
            small_start = start_seconds + (i * small_chunk_seconds)
            small_end = min(small_start + small_chunk_seconds, end_seconds)
            
            # Calculamos qué oraciones corresponden a este segmento pequeño
            segment_ratio = (small_end - small_start) / duration
            sentence_start_idx = int((i / num_small_chunks) * len(sentences))
            sentence_end_idx = int(((i + 1) / num_small_chunks) * len(sentences))
            
            # Asegurar que obtenemos al menos una oración
            if sentence_start_idx == sentence_end_idx and sentence_start_idx < len(sentences):
                sentence_end_idx = sentence_start_idx + 1
            
            # Obtener las oraciones para este segmento pequeño
            segment_text = " ".join(sentences[sentence_start_idx:sentence_end_idx]).strip()
            
            # Si no hay texto (posible en silencios), ponemos una marca
            if not segment_text:
                segment_text = "[silencio o ruido de fondo]"
            
            small_segments.append({
                "inicio": format_time(small_start),
                "fin": format_time(small_end),
                "texto": segment_text,
            })
    
    return small_segments

def save_text_to_file(text, output_path):
    """Guarda la transcripción en un archivo JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(text, f, ensure_ascii=False, indent=4)

def main(file_path, is_video=True, language="es"):
    # Verificar que la API key está configurada
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: No se ha configurado la API key de OpenAI en el archivo .env")
        print("Por favor, crea un archivo .env con el siguiente contenido:")
        print("OPENAI_API_KEY=tu_api_key_aquí")
        return False
    
    # Obtenemos el nombre del archivo sin la extensión para usarlo en el archivo de salida
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    temp_audio_path = f"temp_{base_name}.mp3"
    temp_wav_path = f"temp_{base_name}.wav"
    
    # Usamos el nombre del archivo original para el archivo de salida
    output_text_path = f"output/{base_name}_transcription.json"
    output_detailed_path = f"output/{base_name}_detailed_transcription.json" # Para los segmentos pequeños
    
    if not os.path.exists("output"):
        os.makedirs("output")
    
    try:
        print(f"Procesando archivo: {base_name}")
        
        # Extraer audio si es un video
        if is_video:
            print("Extrayendo audio del video...")
            if not extract_audio_from_video(file_path, temp_audio_path):
                return False
            print("Convirtiendo formato de audio...")
            if not convert_audio_format(temp_audio_path, temp_wav_path):
                return False
        else:
            print("Convirtiendo formato de audio...")
            if not convert_audio_format(file_path, temp_wav_path):
                return False
        
        # Obtener duración del audio
        duration = get_audio_duration(temp_wav_path)
        print(f"Duración total del audio: {format_time(duration)}")
        
        # Transcribir en segmentos de 2 minutos para obtener mejor calidad
        print(f"Transcribiendo audio en segmentos de 2 minutos...")
        transcriptions = transcribe_long_audio(temp_wav_path, language, chunk_length_sec=120)
        
        # Guardar la transcripción original (segmentos de 2 minutos)
        save_text_to_file(transcriptions, output_text_path)
        print(f"Transcripción completada para: {base_name}")
        print(f"Archivo guardado en: {output_text_path}")
        
        # Dividir en segmentos más pequeños de 10 segundos para la IA
        print("Dividiendo transcripción en segmentos de 10 segundos para mejor procesamiento...")
        small_segments = divide_segments_in_smaller_chunks(transcriptions, small_chunk_seconds=10)
        
        # Guardar la transcripción detallada (segmentos de 10 segundos)
        save_text_to_file(small_segments, output_detailed_path)
        print(f"Transcripción detallada guardada en: {output_detailed_path}\n")
        
        return True
        
    except Exception as e:
        print(f"Error durante el procesamiento de {base_name}: {e}")
        return False
    finally:
        # Limpieza de archivos temporales
        if is_video and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

if __name__ == "__main__":
    folder_path = "./Audios"
    language = "es"  # Idioma español por defecto
    
    # Detectar automáticamente si los archivos son video o audio
    for filename in os.listdir(folder_path):
        if filename.endswith(('.mp4', '.mkv', '.avi', '.mov')):
            # Es un video
            file_path = os.path.join(folder_path, filename)
            main(file_path, is_video=True, language=language)
        elif filename.endswith(('.ogg', '.mp3', '.wav', '.m4a', '.flac')):
            # Es un audio
            file_path = os.path.join(folder_path, filename)
            main(file_path, is_video=False, language=language)