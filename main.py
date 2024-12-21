import cv2
import numpy as np

# Загрузка предварительно обученной модели YOLO
def load_yolo_model():
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Функция для обнаружения объектов на видео
def detect_objects(frame, net, output_layers):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    height, width, channels = frame.shape
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
    return boxes, indices

# Инициализация фильтра Калмана
def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], dtype=np.float32)
    return kalman



# Основной цикл обработки видео
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    net, output_layers = load_yolo_model()
    kalmans = {}

    object_id_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, indices = detect_objects(frame, net, output_layers)

        if len(indices) > 0:
            num_objects = len(indices.flatten())
            print(f"Количество объектов на кадре: {num_objects}")

            for i in indices.flatten():
                x, y, w, h = boxes[i]

                cx, cy = x + w // 2, y + h // 2

                object_id = object_id_counter
                object_id_counter += 1

                if object_id not in kalmans:
                    kalmans[object_id] = initialize_kalman()

                kalmans[object_id].correct(np.array([cx, cy], dtype=np.float32))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "Видео №1.MP4"
    process_video(video_path)
