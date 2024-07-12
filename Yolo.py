import cv2
from ultralytics import YOLO
import math
import time
import sys

class ObjectTracker:
    def __init__(self):
        # Inicialización de tu clase ObjectTracker
        self.object_timers = {}
        self.object_total_times = {}
        self.last_seen_time = {}
        self.disappeared_threshold = 5
        self.next_object_id = 0
        self.current_objects = {}
        self.printed_objects = set()  # Set para almacenar los IDs de objetos ya impresos

    def update_timer(self, object_id, current_time):
        # Método para actualizar el temporizador de un objeto
        if object_id not in self.object_timers:
            self.object_timers[object_id] = current_time
            if object_id not in self.object_total_times:
                self.object_total_times[object_id] = 0
        else:
            time_elapsed = current_time - self.object_timers[object_id]
            self.object_total_times[object_id] += time_elapsed
            self.object_timers[object_id] = current_time

        self.last_seen_time[object_id] = current_time

    def get_total_time(self, object_id):
        # Método para obtener el tiempo total de un objeto
        return self.object_total_times.get(object_id, 0)

    def remove_disappeared_objects(self, current_time):
        # Método para eliminar objetos que han desaparecido
        disappeared_objects = []
        for object_id, last_seen in self.last_seen_time.items():
            if current_time - last_seen > self.disappeared_threshold:
                disappeared_objects.append(object_id)
        for object_id in disappeared_objects:
            if object_id not in self.printed_objects:
                total_time = self.get_total_time(object_id)
                output_text = f"Objeto ID {object_id} desapareció. Tiempo total: {total_time:.2f} s\n"
                sys.stdout.write(output_text)  # Escribir en la salida estándar
                self.printed_objects.add(object_id)
            del self.object_timers[object_id]
            del self.last_seen_time[object_id]
            del self.current_objects[object_id]

    def find_closest_object(self, cx, cy):
        # Método para encontrar el objeto más cercano a una posición dada
        closest_id = None
        min_distance = float('inf')
        for object_id, (prev_cx, prev_cy) in self.current_objects.items():
            distance = math.sqrt((prev_cx - cx) ** 2 + (prev_cy - cy) ** 2)
            if distance < min_distance and distance < 50:
                min_distance = distance
                closest_id = object_id
        return closest_id

class Yolo:
    def __init__(self):
        # Inicialización de tu clase Yolo
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("No se puede abrir la cámara")
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.ObjectModel = YOLO('Modelos/yolov8n.onnx')
        self.object_tracker = ObjectTracker()
        self.clsObject = ['persona', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                          'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                          'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                          'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                          'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def draw_area(self, img, color, xi, yi, xf, yf):
        # Método para dibujar un área en la imagen
        img = cv2.rectangle(img, (xi, yi), (xf, yf), color, 1, 1)
        return img

    def draw_text(self, img, color, text, xi, yi, size, thickness, back=False):
        # Método para dibujar texto en la imagen
        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, size, thickness)
        dim = sizetext[0]
        baseline = sizetext[1]
        if back:
            img = cv2.rectangle(img, (xi, yi - dim[1] - baseline), (xi + dim[0], yi + baseline - 7), (0, 0, 0), cv2.FILLED)
        img = cv2.putText(img, text, (xi, yi - 5), cv2.FONT_HERSHEY_DUPLEX, size, color, thickness)
        return img

    def draw_line(self, img, color, xi, yi, xf, yf):
        # Método para dibujar una línea en la imagen
        img = cv2.line(img, (xi, yi), (xf, yf), color, 1, 1)
        return img

    def area(self, frame, xi, yi, xf, yf):
        # Método para calcular el área de interés en la imagen
        al, an, c = frame.shape
        xi, yi = int(xi * an), int(yi * al)
        xf, yf = int(xf * an), int(yf * al)
        return xi, yi, xf, yf

    def prediction_model(self, clean_frame, frame, model):
        # Método para realizar predicciones y actualizar objetos
        bbox = []
        cls = 0
        results = model(clean_frame, stream=True, verbose=False)
        current_time = time.time()
        text_obj2 = 0
        id_objeto = []
        total_time = 0
        for res in results:
            boxes = res.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 < 0: x2 = 0
                if y2 < 0: y2 = 0

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                dx, dy = cx - frame.shape[1] // 2, cy - frame.shape[0] // 2
                distance = math.sqrt(dx ** 2 + dy ** 2)

                object_id = self.object_tracker.find_closest_object(cx, cy)
                if object_id is None:
                    object_id = self.object_tracker.next_object_id
                    self.object_tracker.next_object_id += 1

                self.object_tracker.current_objects[object_id] = (cx, cy)
                self.object_tracker.update_timer(object_id, current_time)

                cls = int(box.cls[0])

                text_obj = f'{self.clsObject[cls]}'
                size_obj, thickness_obj = 0.75, 1
                frame = self.draw_text(frame, (0, 255, 0), text_obj, x1, y1, size_obj, thickness_obj, back=True)
                frame = self.draw_area(frame, (255, 0, 0), x1, y1, x2, y2)

                total_time = self.object_tracker.get_total_time(object_id)
                text_time = f"{object_id} - {total_time:.2f} s"
                frame = self.draw_text(frame, (0, 0, 255), text_time, x1, y1 + 20, size_obj, thickness_obj, back=True)
                text_obj2 = text_obj
                id_objeto = cls
                total_time2 = total_time

        return frame

    def yolo_run(self, cap):
        # Método principal para ejecutar YOLO y procesar el flujo de video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = cv2.waitKey(5)

            clean_frame = frame.copy()
            detect_area_xi, detect_area_yi, detect_area_xf, detect_area_yf = self.area(frame, 0.0351, 0.0486, 0.9649, 0.9444)
            color = (0, 255, 0)
            text_detect = 'Detection area'
            size_detect, thickness_detect = 0.75, 1
            frame = self.draw_area(frame, color, detect_area_xi, detect_area_yi, detect_area_xf, detect_area_yf)
            frame = self.draw_text(frame, color, text_detect, detect_area_xi, detect_area_yf + 30, size_detect, thickness_detect)

            frame = self.prediction_model(clean_frame, frame, self.ObjectModel)
            cv2.imshow("Yolo", frame)

            self.object_tracker.remove_disappeared_objects(time.time())

            if t == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Redirige la salida estándar hacia un archivo
ruta_archivo = r"C:\tu\direccion\de\ruta\donde\guarda.txt"
sys.stdout = open(ruta_archivo, 'w')

# Crea una instancia de la clase Yolo y ejecuta el método yolo_run
yolo = Yolo()
yolo.yolo_run(yolo.cap)

# Restaura la salida estándar al valor predeterminado
sys.stdout.close()
sys.stdout = sys.__stdout__
