import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
import moviepy.editor as mp
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_audio_from_video(video_path, audio_path):
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def convert_audio_format(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        # Normalizar audio para mejor transcripción
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio.dBFS
        normalized = audio.apply_gain(change_in_dBFS)
        normalized.export(output_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def transcribe_smart(audio_path, language="es"):
    """Transcribe en segmentos de 10-20 segundos sin detección de hablantes"""
    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    
    # Detectar pausas para mejor segmentación
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=300, silence_thresh=-40)
    
    segments = []
    os.makedirs("temp_chunks", exist_ok=True)
    
    # Calcular chunks de 10-20 segundos
    chunks_to_process = []
    i = 0
    while i < len(nonsilent_ranges):
        start = nonsilent_ranges[i][0]
        end = nonsilent_ranges[i][1]
        
        # Agrupar hasta 10-20 segundos
        while i + 1 < len(nonsilent_ranges):
            next_end = nonsilent_ranges[i + 1][1]
            duration_sec = (next_end - start) / 1000
            
            if duration_sec <= 10:
                end = next_end
                i += 1
            elif duration_sec <= 20:
                silence_gap = nonsilent_ranges[i + 1][0] - nonsilent_ranges[i][1]
                if silence_gap > 500:
                    break
                end = next_end
                i += 1
            else:
                break
        
        chunks_to_process.append((start, end))
        i += 1
    
    # Procesar con barra de progreso
    for start, end in tqdm(chunks_to_process, desc="Procesando audio", unit="segmento"):
        chunk = audio[start:end]
        chunk_file = f"temp_chunks/chunk_{start}.wav"
        chunk.export(chunk_file, format="wav", parameters=["-ar", "16000", "-ac", "1"])
        
        file_size = os.path.getsize(chunk_file) / (1024 * 1024)
        print(f"Transcribiendo segmento {len(segments) + 1}/{len(chunks_to_process)} (Tamaño: {file_size:.2f} MB)")
        
        try:
            with open(chunk_file, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=language,
                    response_format="verbose_json"
                )
            
            text = response.text.strip() if hasattr(response, 'text') else str(response).strip()
            
            if text:  # Solo agregar si hay texto
                segments.append({
                    "inicio": format_time(start / 1000),
                    "fin": format_time(end / 1000),
                    "texto": text
                })
            
        except Exception as e:
            print(f"Error en segmento: {e}")
        
        os.remove(chunk_file)
        time.sleep(0.5)  # Rate limiting
    
    try:
        os.rmdir("temp_chunks")
    except:
        pass
    
    return segments

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

def save_outputs(segments, base_name, duration):
    os.makedirs("output", exist_ok=True)
    
    # JSON simple para Claude
    output = {
        "archivo": base_name,
        "duracion": format_time(duration),
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "total_segmentos": len(segments),
        "segmentos": segments
    }
    
    json_path = f"output/{base_name}_claude.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Texto simple con timestamps
    txt_path = f"output/{base_name}_transcripcion.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"TRANSCRIPCIÓN: {base_name}\n")
        f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Duración: {format_time(duration)}\n")
        f.write("=" * 80 + "\n\n")
        
        for seg in segments:
            f.write(f"[{seg['inicio']}] {seg['texto']}\n\n")
    
    print(f"\n✅ Archivos guardados:")
    print(f"   • {json_path}")
    print(f"   • {txt_path}")
    
    return json_path, txt_path

def generate_claude_prompt(json_path):
    """Genera el prompt perfecto para Claude"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Crear el prompt completo
    prompt = f"""Analiza esta transcripción de una reunión de trabajo:

{json.dumps(data, ensure_ascii=False, indent=2)}

Por favor proporciona un análisis COMPLETO y DETALLADO que incluya:

## 📊 RESUMEN EJECUTIVO
- Resume en 3-4 líneas lo más importante de la reunión

## 🎯 TEMAS PRINCIPALES DISCUTIDOS
- Lista y explica cada tema tratado con detalle

## ✅ TAREAS ASIGNADAS
IMPORTANTE: Identifica CLARAMENTE quién debe hacer qué. Busca nombres mencionados (Daniel, Carlos, etc.) y qué se les asignó específicamente.
- [Persona] → [Tarea específica] → [Deadline si se mencionó]

## 🔧 DECISIONES TÉCNICAS
- Qué se decidió implementar
- Cómo se va a estructurar (tablas, endpoints, formularios)
- Tecnologías o herramientas mencionadas

## ⚠️ PUNTOS CRÍTICOS
- Problemas o preocupaciones mencionadas
- Riesgos identificados
- Cosas que no quedaron claras

## 💡 RECOMENDACIONES PARA HACER EL TRABAJO CORRECTAMENTE

Basándome en lo discutido, estas son las mejores prácticas para completar el trabajo exitosamente y evitar problemas:

### Para el desarrollo:
1. ¿Qué debería hacer PRIMERO?
2. ¿Qué estructura de código/base de datos es la más apropiada?
3. ¿Qué validaciones son CRÍTICAS que no debo olvidar?
4. ¿Qué errores comunes debo evitar?

### Para la gestión:
1. ¿Cómo debería comunicar mi progreso?
2. ¿Qué preguntas debería hacer ANTES de empezar?
3. ¿Qué documentación debería preparar?

## 🚨 ALERTAS IMPORTANTES
Si detectas alguna señal de preocupación sobre el desempeño de alguien, o críticas hacia el trabajo, por favor indícalo claramente aquí.

## 📅 PRÓXIMOS PASOS
- Qué se debe entregar primero
- Cuándo es la próxima reunión o revisión
- Qué debe estar listo para entonces

---
NOTA: Si identificas que alguien fue criticado o hay presión sobre algún tema, por favor sé MUY ESPECÍFICO sobre qué se espera para evitar problemas."""
    
    return prompt

def main(file_path, is_video=True, language="es"):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    temp_audio = f"temp_{base_name}.mp3"
    temp_wav = f"temp_{base_name}.wav"
    
    try:
        print(f"\n🎬 Procesando: {base_name}")
        
        if is_video:
            print("📹 Extrayendo audio del video...")
            if not extract_audio_from_video(file_path, temp_audio):
                return False
            if not convert_audio_format(temp_audio, temp_wav):
                return False
        else:
            print("🎵 Convirtiendo formato de audio...")
            if not convert_audio_format(file_path, temp_wav):
                return False
        
        audio = AudioSegment.from_file(temp_wav)
        duration = len(audio) / 1000
        
        print(f"⏱️ Duración: {format_time(duration)}")
        print("🎯 Transcribiendo audio...")
        
        segments = transcribe_smart(temp_wav, language)
        
        print(f"\n📊 Resumen:")
        print(f"  • Total segmentos: {len(segments)}")
        print(f"  • Duración: {format_time(duration)}")
        
        json_path, txt_path = save_outputs(segments, base_name, duration)
        
        # Generar prompt para Claude
        print("\n" + "="*80)
        print("📝 PROMPT PARA CLAUDE (copia todo desde aquí):")
        print("="*80)
        
        prompt = generate_claude_prompt(json_path)
        print(prompt)
        
        print("\n" + "="*80)
        print("☝️ COPIA TODO EL PROMPT ANTERIOR Y PÉGALO EN CLAUDE")
        print("="*80)
        
        # Guardar prompt en archivo
        prompt_path = f"output/{base_name}_prompt_claude.txt"
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"\n💾 Prompt también guardado en: {prompt_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Limpieza de archivos temporales
        for temp in [temp_audio, temp_wav]:
            if os.path.exists(temp):
                os.remove(temp)

if __name__ == "__main__":
    folder_path = "./Audios"
    language = "es"  # Cambiar a "en" para inglés
    
    # Verificar API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Falta OPENAI_API_KEY en archivo .env")
        print("Crea un archivo .env con: OPENAI_API_KEY=tu_api_key")
        exit(1)
    
    # Procesar archivos
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.webm')
    audio_extensions = ('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac')
    
    files_found = False
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(video_extensions):
            files_found = True
            main(os.path.join(folder_path, filename), is_video=True, language=language)
        elif filename.lower().endswith(audio_extensions):
            files_found = True
            main(os.path.join(folder_path, filename), is_video=False, language=language)
    
    if not files_found:
        print(f"\n⚠️ No se encontraron archivos en {folder_path}")
        print(f"   Formatos soportados: {video_extensions + audio_extensions}")