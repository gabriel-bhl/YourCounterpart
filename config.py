from pathlib import Path

# --- directory paths ---
PROJ_PATH_LIST = [r'C:\Users\gabri\DataspellProjects\Counterpart']  # basta adicionar o seu caminho até o diretório raiz do
# projeto

found_path = False
for path in PROJ_PATH_LIST:
    if Path(path).exists():
        PROJ_PATH = Path(path)
        found_path = True

if not found_path:
    PROJ_PATH = Path.cwd().parents[1]

DATA_PATH = PROJ_PATH / 'data'

# Multithread Colors
COR = ['BLACK', 'RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE']
