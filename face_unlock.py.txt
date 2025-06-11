import os
import time
import cv2
import numpy as np
import face_recognition
import requests
from requests.auth import HTTPDigestAuth
import urllib3
import RPi.GPIO as GPIO
import config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

AUTH = HTTPDigestAuth(config.USERNAME, config.PASSWORD)
VTO_IP = config.VTO_IP
SNAPSHOT_URL = f"https://{VTO_IP}/cgi-bin/snapshot.cgi?channel=1"
EVENT_URL = f"https://{VTO_IP}/cgi-bin/eventManager.cgi?action=attach&codes=%5BAll%5D"
SNAPSHOT_PATH = "./latest.jpg"
ENCODING_DIR = "./FaceDB"
TOLERANCE = 0.5
RELAY_PIN = config.DOOR_RELAY_PIN
RELAY_OPEN_TIME = 5  # секунд

os.makedirs(ENCODING_DIR, exist_ok=True)

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_PIN, GPIO.OUT)
    GPIO.output(RELAY_PIN, GPIO.HIGH)  # реле выключено (зависит от реле)

def open_door():
    print("🔓 Открываем дверь...")
    GPIO.output(RELAY_PIN, GPIO.LOW)  # включаем реле (зависит от реле)
    time.sleep(RELAY_OPEN_TIME)
    GPIO.output(RELAY_PIN, GPIO.HIGH)
    print("🔒 Дверь закрыта.")

def capture_snapshot():
    try:
        response = requests.get(SNAPSHOT_URL, auth=AUTH, verify=False)
        if response.status_code == 200 and b"JFIF" in response.content[:20]:
            with open(SNAPSHOT_PATH, "wb") as f:
                f.write(response.content)
            print("📸 Снимок получен и сохранён.")
            return True
        else:
            print(f"❌ Ошибка при получении снимка: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка получения снимка: {e}")
        return False

def load_known_faces():
    encodings = []
    names = []
    for file in os.listdir(ENCODING_DIR):
        if file.endswith(".npy"):
            name = file[:-4]
            encoding = np.load(os.path.join(ENCODING_DIR, file))
            encodings.append(encoding)
            names.append(name)
    print(f"📚 Загружено лиц: {len(names)}")
    return encodings, names

def main():
    setup_gpio()
    known_encodings, known_names = load_known_faces()
    print("📡 Ожидание нажатия кнопки на VTO...")

    try:
        with requests.get(EVENT_URL, auth=AUTH, verify=False, stream=True) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8", errors="ignore")
                if "CallNoAnswered" in decoded:
                    print("🔔 Кнопка вызова нажата!")
                    if capture_snapshot():
                        image = face_recognition.load_image_file(SNAPSHOT_PATH)
                        face_locations = face_recognition.face_locations(image)
                        if not face_locations:
                            print("❌ Лицо не найдено на снимке.")
                            continue
                        face_encodings = face_recognition.face_encodings(image, face_locations)
                        matched = False
                        for face_encoding in face_encodings:
                            distances = face_recognition.face_distance(known_encodings, face_encoding)
                            if len(distances) == 0:
                                continue
                            best_match_index = np.argmin(distances)
                            if distances[best_match_index] <= TOLERANCE:
                                name = known_names[best_match_index]
                                print(f"✅ Совпадение с {name}: расстояние {distances[best_match_index]:.4f}")
                                open_door()
                                matched = True
                                break
                        if not matched:
                            print("❌ Совпадений не найдено. Дверь не открыта.")
    except KeyboardInterrupt:
        print("\n⛔ Прервано пользователем.")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
