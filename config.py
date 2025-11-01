"""
Файл конфигурации проекта
"""


# === ПАРАМЕТРЫ МОДЕЛИ ===
# Доступные варианты моделей YOLO
AVAILABLE_MODELS = {
    'yolo11n': 'yolo11n.pt',  # Nano - самая быстрая
    'yolo11s': 'yolo11s.pt',  # Small - хороший баланс
    'yolo11m': 'yolo11m.pt',  # Medium - высокая точность
    'yolo11l': 'yolo11l.pt',  # Large - максимальная точность
    'yolo11x': 'yolo11x.pt',  # Extra Large - экстремальная точность
}

# Модель по умолчанию
DEFAULT_MODEL = 'yolo11n.pt'

# Порог уверенности по умолчанию (0.0 - 1.0)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Минимальный порог для фильтрации низкоуверенных детекций
MIN_CONFIDENCE_THRESHOLD = 0.1

# Максимальный порог уверенности
MAX_CONFIDENCE_THRESHOLD = 0.95


# === ПАРАМЕТРЫ ВИДЕО ===
# Поддерживаемые форматы видео
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

# Кодек для выходного видео
OUTPUT_VIDEO_CODEC = 'mp4v'

# Минимальная высота кадра для обработки
MIN_FRAME_HEIGHT = 480

# Минимальная ширина кадра для обработки
MIN_FRAME_WIDTH = 640

# Максимальная разрешение входного видео (для оптимизации)
MAX_FRAME_HEIGHT = 1080
MAX_FRAME_WIDTH = 1920


# === ПАРАМЕТРЫ ОБРАБОТКИ ===
# Порог IoU для фильтрации дублирующихся детекций
NMS_IOU_THRESHOLD = 0.5

# Интервал вывода прогресса (каждые N кадров)
PROGRESS_PRINT_INTERVAL = 30

# Устройство по умолчанию ('cpu' или 'cuda')
DEFAULT_DEVICE = 'cpu'


# === ПАРАМЕТРЫ ОТРИСОВКИ ===
# Цвет bounding box в формате BGR
BBOX_COLOR = (0, 255, 0)  # Зеленый

# Цвет текста в формате BGR
TEXT_COLOR = (0, 0, 0)  # Черный

# Цвет фона для текста в формате BGR
TEXT_BG_COLOR = (0, 255, 0)  # Зеленый

# Толщина линий bounding box (в пикселях)
BBOX_THICKNESS = 2

# Размер шрифта для текста (0-1 диапазон)
FONT_SCALE = 0.6

# Толщина текста (в пикселях)
FONT_THICKNESS = 2

# Отступ текста от края bounding box (в пикселях)
TEXT_OFFSET = 10


# === ПАРАМЕТРЫ ЛОГИРОВАНИЯ ===
# Путь к файлу логирования
LOG_FILE = 'pedestrian_detection.log'

# Уровень логирования ('DEBUG', 'INFO', 'WARNING', 'ERROR')
LOG_LEVEL = 'INFO'


# === ПУТИ ПРОЕКТА ===
# Директория для входных видео
INPUT_DIR = 'data'

# Директория для выходных видео
OUTPUT_DIR = 'output'

# Директория для весов моделей
MODELS_DIR = 'models'

# Директория для логов
LOGS_DIR = 'logs'


# === КЛАСС ЧЕЛОВЕКА В COCO ===
# ID класса "person" в наборе данных COCO (всегда 0)
PERSON_CLASS_ID = 0

# Имя класса для вывода
PERSON_CLASS_NAME = 'person'


# === ПАРАМЕТРЫ ПРОИЗВОДИТЕЛЬНОСТИ ===
# Максимальное количество рабочих потоков для загрузки данных
MAX_WORKERS = 4

# Размер буфера для кэширования кадров
BUFFER_SIZE = 10

# Интервал сохранения промежуточных результатов (в кадрах)
CHECKPOINT_INTERVAL = 100