"""
Вспомогательные утилиты для работы с видео и детекциями
Функции для преобразования форматов, валидации и логирования
"""
from pathlib import Path
import logging
from typing import Tuple, List


def setup_logging(log_file: str = "pedestrian_detection.log") -> logging.Logger:
    """
    Настройка логирования для приложения.
    
    Аргументы:
        log_file: путь к файлу логирования
        
    Возвращает:
        Сконфигурированный логгер
    """
    # Создание логгера
    logger = logging.getLogger('PedestrianDetection')
    logger.setLevel(logging.DEBUG)
    
    # Создание обработчика для записи в файл
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Создание обработчика для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Форматирование сообщений логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Добавление обработчиков к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def validate_video_file(video_path: str) -> bool:
    """
    Проверка корректности пути и существования видеофайла.
    
    Аргументы:
        video_path: путь к видеофайлу
        
    Возвращает:
        True если файл существует и имеет поддерживаемый формат, False иначе
    """
    # Проверка существования файла
    path = Path(video_path)
    if not path.exists():
        return False
    
    # Проверка расширения файла
    supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    if path.suffix.lower() not in supported_formats:
        return False
    
    # Проверка что это файл, а не директория
    if not path.is_file():
        return False
    
    return True


def normalize_coordinates(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int
) -> Tuple[float, float, float, float]:
    """
    Нормализация координат bounding box к диапазону [0, 1].
    
    Аргументы:
        bbox: координаты bounding box (x1, y1, x2, y2) в пикселях
        frame_width: ширина кадра в пикселях
        frame_height: высота кадра в пикселях
        
    Возвращает:
        Нормализованные координаты в диапазоне [0, 1]
    """
    x1, y1, x2, y2 = bbox
    
    # Нормализация координат путем деления на размеры кадра
    norm_x1 = x1 / frame_width
    norm_y1 = y1 / frame_height
    norm_x2 = x2 / frame_width
    norm_y2 = y2 / frame_height
    
    return (norm_x1, norm_y1, norm_x2, norm_y2)


def denormalize_coordinates(
    norm_bbox: Tuple[float, float, float, float],
    frame_width: int,
    frame_height: int
) -> Tuple[int, int, int, int]:
    """
    Преобразование нормализованных координат обратно в пиксели.
    
    Аргументы:
        norm_bbox: нормализованные координаты bounding box (0-1 диапазон)
        frame_width: ширина кадра в пикселях
        frame_height: высота кадра в пикселях
        
    Возвращает:
        Координаты bounding box в пикселях
    """
    norm_x1, norm_y1, norm_x2, norm_y2 = norm_bbox
    
    # Восстановление координат в пиксели путем умножения на размеры кадра
    x1 = int(norm_x1 * frame_width)
    y1 = int(norm_y1 * frame_height)
    x2 = int(norm_x2 * frame_width)
    y2 = int(norm_y2 * frame_height)
    
    return (x1, y1, x2, y2)


def calculate_iou(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int]
) -> float:
    """
    Расчет пересечения над объединением (IoU) для двух bounding boxes.
    
    Используется для оценки перекрытия между детекциями и для NMS (Non-Maximum Suppression).
    
    Аргументы:
        bbox1: первый bounding box (x1, y1, x2, y2)
        bbox2: второй bounding box (x1, y1, x2, y2)
        
    Возвращает:
        Значение IoU в диапазоне [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Вычисление координат пересечения
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Вычисление площади пересечения
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # Вычисление площадей bounding boxes
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Вычисление площади объединения
    union_area = bbox1_area + bbox2_area - inter_area
    
    # Вычисление IoU с защитой от деления на ноль
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    
    return iou


def filter_detections_by_iou(
    detections: List[Tuple],
    iou_threshold: float = 0.5
) -> List[Tuple]:
    """
    Фильтрация дублирующихся детекций на основе IoU.
    
    Удаляет детекции с высоким перекрытием, сохраняя ту, у которой выше уверенность.
    
    Аргументы:
        detections: список детекций (x1, y1, x2, y2, уверенность, имя_класса)
        iou_threshold: порог IoU для удаления дублей
        
    Возвращает:
        Отфильтрованный список детекций
    """
    if not detections:
        return []
    
    # Сортировка детекций по уверенности в убывающем порядке
    sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)
    
    filtered = []
    
    # Обработка каждой детекции
    for detection in sorted_detections:
        should_keep = True
        x1, y1, x2, y2, conf, class_name = detection
        current_bbox = (x1, y1, x2, y2)
        
        # Проверка пересечения с уже добавленными детекциями
        for kept_detection in filtered:
            kept_x1, kept_y1, kept_x2, kept_y2, _, _ = kept_detection
            kept_bbox = (kept_x1, kept_y1, kept_x2, kept_y2)
            
            # Расчет IoU между текущей и сохраненной детекцией
            iou = calculate_iou(current_bbox, kept_bbox)
            
            # Если IoU выше порога, пропускаем текущую детекцию
            if iou > iou_threshold:
                should_keep = False
                break
        
        # Добавление детекции если она не является дублем
        if should_keep:
            filtered.append(detection)
    
    return filtered


def get_detection_statistics(detections: List[Tuple]) -> dict:
    """
    Расчет статистики по детекциям.
    
    Аргументы:
        detections: список детекций
        
    Возвращает:
        Словарь со статистикой (количество, средняя уверенность, мин/макс)
    """
    if not detections:
        return {
            'count': 0,
            'avg_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0
        }
    
    # Извлечение уверенностей из всех детекций
    confidences = [det[4] for det in detections]
    
    # Расчет статистики
    return {
        'count': len(detections),
        'avg_confidence': sum(confidences) / len(confidences),
        'min_confidence': min(confidences),
        'max_confidence': max(confidences)
    }