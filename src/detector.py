"""
Модуль детектора пешеходов
Обработка детекции людей с использованием YOLO
"""
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple


class PedestrianDetector:
    """
    Детектор пешеходов с использованием архитектуры YOLO.
    
    Атрибуты:
        model: экземпляр модели YOLO
        confidence_threshold: минимальная уверенность для валидных детекций
        device: вычислительное устройство (cpu/cuda)
    """
    
    def __init__(
        self,
        model_name: str = 'yolo11n.pt',
        confidence_threshold: float = 0.5,
        device: str = 'cpu',
        imgsz: int = 640,  # Новое поле — размер входного изображения
    ):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.imgsz = imgsz
        self.model.to(device)
        self.person_class_id = 0
    
    def detect(self, frame: np.ndarray) -> List[Tuple]:
        """
        Детекция пешеходов в одном кадре.
        
        Аргументы:
            frame: входной кадр в виде массива numpy (формат BGR)
            
        Возвращает:
            Список кортежей с координатами (x1, y1, x2, y2, уверенность, имя_класса)
        """
        # Выполнение инференса модели с фильтрацией только класса "person"
        results = self.model(
        frame,
        conf=self.confidence_threshold,
        classes=[self.person_class_id],
        imgsz=self.imgsz,     # Увеличиваем размер входного кадра для инференса
        verbose=False
        )
        
        detections = []
        
        # Обработка результатов для каждого обнаруженного объекта
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Извлечение координат bounding box в формате (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Извлечение уверенности предсказания
                confidence = float(box.conf[0].cpu().numpy())
                
                # Извлечение класса и его имени
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                # Добавление результата в список с преобразованием координат в int
                detections.append(
                    (int(x1), int(y1), int(x2), int(y2), 
                     confidence, class_name)
                )
        
        return detections
    
    def get_model_info(self) -> dict:
        """
        Получение информации о загруженной модели.
        
        Возвращает:
            Словарь с информацией о модели (имя, устройство, порог уверенности)
        """
        return {
            'model_name': self.model.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold
        }