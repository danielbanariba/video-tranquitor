import os
import sys

def generar_estructura_directorios(ruta_directorio, archivo_salida, nivel=0, prefijo='', directorios_ignorados=None):
    """
    Genera la estructura de directorios en un archivo de texto.
    
    Args:
        ruta_directorio (str): Ruta del directorio a procesar
        archivo_salida (file): Archivo donde se guardará la estructura
        nivel (int): Nivel de profundidad actual
        prefijo (str): Prefijo para la indentación
        directorios_ignorados (list): Lista de nombres de directorios a ignorar
    """
    if directorios_ignorados is None:
        directorios_ignorados = ['target', '.git', '.env', 'node_modules', '__pycache__', 'venv', '__init__.py', "env"]
    
    # Obtener la lista de elementos en el directorio
    elementos = os.listdir(ruta_directorio)
    elementos.sort()
    
    # Filtrar elementos a ignorar
    elementos_filtrados = [e for e in elementos if not e.startswith('.') and e not in directorios_ignorados]
    
    # Procesar cada elemento
    for i, elemento in enumerate(elementos_filtrados):
        # Ruta completa del elemento
        ruta_completa = os.path.join(ruta_directorio, elemento)
        
        # Determinar si es el último elemento del nivel actual
        es_ultimo = (i == len(elementos_filtrados) - 1)
        
        # Elegir el conector adecuado
        if es_ultimo:
            conector = '└── '
            nuevo_prefijo = prefijo + '    '
        else:
            conector = '├── '
            nuevo_prefijo = prefijo + '│   '
        
        # Verificar si es un directorio para agregar el emoji de carpeta
        if os.path.isdir(ruta_completa):
            elemento_con_emoji = f"📁{elemento}"
        else:
            elemento_con_emoji = elemento
        
        # Escribir el elemento en el archivo
        archivo_salida.write(f"{prefijo}{conector}{elemento_con_emoji}\n")
        
        # Si es un directorio, procesar recursivamente
        if os.path.isdir(ruta_completa):
            generar_estructura_directorios(ruta_completa, archivo_salida, nivel + 1, nuevo_prefijo, directorios_ignorados)

def main():
    # Verificar si se proporcionó un directorio como argumento
    if len(sys.argv) < 2:
        print("Uso: python estructura_directorios.py <ruta_directorio> [archivo_salida] [--exclude=dir1,dir2,...]")
        print("Si no se especifica archivo_salida, se usará 'estructura_proyecto.txt'")
        print("Use --exclude=dir1,dir2,... para especificar directorios adicionales a ignorar")
        return
    
    # Obtener la ruta del directorio
    ruta_directorio = sys.argv[1]
    
    # Verificar si el directorio existe
    if not os.path.isdir(ruta_directorio):
        print(f"Error: '{ruta_directorio}' no es un directorio válido.")
        return
    
    # Procesar argumentos para el archivo de salida y directorios a excluir
    archivo_salida_nombre = "estructura_proyecto.txt"
    directorios_ignorados = ['target', '.git', '.env', 'node_modules', '__pycache__', 'venv', '__init__.py', "env"]
    
    for arg in sys.argv[2:]:
        if arg.startswith('--exclude='):
            dirs_extra = arg[10:].split(',')
            directorios_ignorados.extend([d.strip() for d in dirs_extra if d.strip()])
        elif not arg.startswith('--'):
            archivo_salida_nombre = arg
    
    # Abrir el archivo de salida
    with open(archivo_salida_nombre, 'w', encoding='utf-8') as archivo_salida:
        # Escribir el nombre del directorio raíz
        nombre_directorio = os.path.basename(os.path.abspath(ruta_directorio))
        archivo_salida.write(f"{nombre_directorio}\n")
        
        # Generar la estructura
        generar_estructura_directorios(ruta_directorio, archivo_salida, directorios_ignorados=directorios_ignorados)
    
    print(f"Estructura de directorios guardada en '{archivo_salida_nombre}'")
    print(f"Se ignoraron los siguientes directorios: {', '.join(directorios_ignorados)}")

if __name__ == "__main__":
    main()