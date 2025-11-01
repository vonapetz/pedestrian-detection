"""
Модуль обработчика видео
Обработка чтения видео, обработки кадров и записи результата
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import time


class VideoProcessor:
    """
    Обработчик видеофайлов с поддержкой детекции пешеходов.
    
    Атрибуты:
        detector: экземпляр PedestrianDetector для выполнения детекции
        stats: словарь для сохранения статистики обработки видео
    """
    
    def __init__(self, detector):
        """
        Инициализация обработчика видео.
        
        Аргументы:
            detector: инициализированный экземпляр PedestrianDetector
        """
        self.detector = detector
        
        # Инициализация словаря для сбора статистики
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_confidence': 0.0,
            'processing_time': 0.0
        }
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        display: bool = False
    ) -> None:
        """
        Обработка видеофайла и сохранение результатов с отрисованными детекциями.
        
        Аргументы:
            input_path: путь к входному видеофайлу
            output_path: путь для сохранения выходного видео с детекциями
            display: отображать ли процесс обработки в реальном времени (по умолчанию False)
        
        Исключения:
            ValueError: если входной видеофайл не может быть открыт
        """
        # Открытие видеофайла через OpenCV
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Не удается открыть видеофайл: {input_path}")
        
        # Получение параметров видео (размер и частота кадров)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Создание директории для выхода если необходимо
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Инициализация видеописателя с кодеком mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        
        # Инициализация счетчиков и таймера
        frame_count = 0
        start_time = time.time()
        
        # Вывод информации о видео
        print(f"Информация о видео: {frame_width}x{frame_height} @ {fps}fps")
        print(f"Всего кадров: {total_frames}")
        
        # Основной цикл обработки видео кадр за кадром
        while True:
            ret, frame = cap.read()
            
            # Выход из цикла при окончании видео
            if not ret:
                break
            
            # Выполнение детекции пешеходов в текущем кадре
            detections = self.detector.detect(frame)
            
            # Отрисовка bounding boxes и меток на кадр
            annotated_frame = self._draw_detections(frame, detections)
            
            # Запись аннотированного кадра в выходной видеофайл
            out.write(annotated_frame)
            
            # Обновление статистики обработки
            self._update_stats(detections)
            
            frame_count += 1
            
            # Периодический вывод прогресса обработки
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Прогресс: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Опциональное отображение в реальном времени
            if display:
                cv2.imshow('Детекция пешеходов', annotated_frame)
                # Выход по нажатию клавиши 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Вычисление времени обработки
        processing_time = time.time() - start_time
        self.stats['processing_time'] = processing_time
        self.stats['total_frames'] = frame_count
        
        # Закрытие ресурсов (видеопотоки и окна)
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()
    
    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list
    ) -> np.ndarray:
        """
        Отрисовка bounding boxes и меток на кадр.
        
        Каждая детекция отображается с зеленым прямоугольником и черным текстом на зеленом фоне.
        
        Аргументы:
            frame: входной кадр для отрисовки
            detections: список кортежей детекций (x1, y1, x2, y2, уверенность, имя_класса)
            
        Возвращает:
            Аннотированный кадр с отрисованными детекциями
        """
        # Копирование кадра для сохранения оригинала
        annotated = frame.copy()
        
        # Обработка каждой детекции
        for detection in detections:
            x1, y1, x2, y2, confidence, class_name = detection
            
            # Отрисовка зеленого bounding box (толщина 2 пикселя)
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
            
            # Подготовка текстовой метки с именем класса и уверенностью
            label = f"{class_name}: {confidence:.2f}"
            
            # Вычисление размера текстовой метки для корректного расположения фона
            (label_w, label_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            
            # Отрисовка зеленого прямоугольника - фона для текста метки
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                (0, 255, 0),
                -1
            )
            
            # Отрисовка черного текста метки поверх зеленого фона
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return annotated
    
    def _update_stats(self, detections: list) -> None:
        """
        Обновление статистики обработки на основе текущего кадра.
        
        Аргументы:
            detections: список детекций в текущем кадре
        """
        # Увеличение счетчика общего количества детекций
        self.stats['total_detections'] += len(detections)
        
        # Накопление суммы уверенностей для последующего расчета среднего значения
        for detection in detections:
            confidence = detection[4]
            self.stats['total_confidence'] += confidence
    
    def get_statistics(self) -> dict:
        """
        Получение статистики обработки видео.
        
        Возвращает:
            Словарь со следующими метриками:
            - total_frames: общее количество обработанных кадров
            - avg_detections: среднее количество детекций на один кадр
            - avg_confidence: средняя уверенность всех детекций
            - fps: количество кадров в секунду при обработке
        """
        total_frames = self.stats['total_frames']
        total_detections = self.stats['total_detections']
        
        # Защита от деления на ноль
        if total_frames == 0:
            return self.stats
        
        # Расчет итоговой статистики
        return {
            'total_frames': total_frames,
            'avg_detections': total_detections / total_frames,
            'avg_confidence': (
                self.stats['total_confidence'] / total_detections
                if total_detections > 0 else 0.0
            ),
            'fps': total_frames / self.stats['processing_time']
        }