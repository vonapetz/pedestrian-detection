import argparse
import sys
from pathlib import Path
from src.detector import PedestrianDetector
from src.video_processor import VideoProcessor


def parse_arguments():
    """
    Парсинг аргументов командной строки.
    
    Возвращает:
        argparse.Namespace: Распарсенные аргументы
    """
    parser = argparse.ArgumentParser(
        description='Детекция и отслеживание пешеходов в видеофайлах'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Путь к входному видеофайлу'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/detected_video.mp4',
        help='Путь для сохранения видеофайла'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11n.pt',
        choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
                 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolo8l.pt', 'yolo8x.pt'],
        help='Разные модельки по новизне и размеру'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Порог уверенности для детекции (0.0-1.0)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', '0', '1'],
        help='Устройство для запуска инференса'
    )
    
    return parser.parse_args()


def main():
    # print('Точка входа работает')

    """Основная функция выполнения."""
    args = parse_arguments()
    
    # Проверка существования входного файла
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Ошибка: входной файл '{args.input}' не найден!")
        sys.exit(1)
    
    # Инициализация детектора
    print(f"Загрузка модели: {args.model}")
    detector = PedestrianDetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    # Обработка видео
    print(f"Обработка видео: {args.input}")
    processor = VideoProcessor(detector)
    processor.process_video(
        input_path=str(input_path),
        output_path=args.output
    )
    
    print(f"Обработка завершена! Результат сохранен в: {args.output}")
    
    # Вывод статистики
    stats = processor.get_statistics()
    print("\n=== Статистика детекции ===")
    print(f"Всего обработано кадров: {stats['total_frames']}")
    print(f"Среднее количество детекций на кадр: {stats['avg_detections']:.2f}")
    print(f"Средняя уверенность: {stats['avg_confidence']:.2f}")
    print(f"Скорость обработки (FPS): {stats['fps']:.2f}")


if __name__ == "__main__":
    main()
    