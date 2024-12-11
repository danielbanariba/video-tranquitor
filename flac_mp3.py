import os
from pydub import AudioSegment

carpeta_entrada = r"E:\Musica\Musica\FLAC\Soundtrack\Mac Quayle - Mr. Robot, Vol. 8 (Original Television Series Soundtrack) (2023) [24Bit]"
carpeta_salida = r"C:\Users\banar\Music\Mac Quayle - Mr. Robot, Vol. 8"  # Reemplaza con la ruta de tu carpeta de salida

if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

for archivo in os.listdir(carpeta_entrada):
    if archivo.endswith('.flac'):
        ruta_flac = os.path.join(carpeta_entrada, archivo)
        nombre_archivo = os.path.splitext(archivo)[0] + '.mp3'
        ruta_mp3 = os.path.join(carpeta_salida, nombre_archivo)
        sonido = AudioSegment.from_file(ruta_flac, format='flac')
        sonido.export(ruta_mp3, format='mp3')
        print(f'Convertido: {archivo} a {os.path.basename(ruta_mp3)}')