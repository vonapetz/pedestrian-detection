"""
Инициализационный файл пакета src
Экспортирует основные классы и функции для использования
"""

from .detector import PedestrianDetector
from .video_processor import VideoProcessor

__all__ = [
    'PedestrianDetector',
    'VideoProcessor',
]