import os
import cv2
import numpy as np
import face_recognition
import requests
from requests.auth import HTTPDigestAuth
import urllib3
import config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

AUTH = HTTPDigestAuth(config.USERNAME, config.PASSWORD)
SNAPSHOT_URL = f"https://{config.VTO_IP}/cgi-bin/snapshot.cgi?channel=1"
SNAPSHOT_PATH = "./latest.jpg"
ENCODING_DIR = "./FaceDB"

os.makedirs(ENCODING_DIR, exist_ok=True)

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

def encode_and_save(name, image_path):
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image, model="hog")
    if not locations:
        print("⚠️ Лицо не найдено на снимке.")
        return False
    encoding = face_recognition.face_encodings(image, known_face_locations=locations)[0]
    np.save(os.path.join(ENCODING_DIR, f"{name}.npy"), encoding)
    cv2.imwrite(os.path.join(ENCODING_DIR, f"{name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"✅ Лицо '{name}' сохранено.")
    return True

def main():
    input("Нажмите Enter для захвата снимка...")
    if capture_snapshot():
        name = input("Введите имя для сохранения: ").strip()
        if name:
            if encode_and_save(name, SNAPSHOT_PATH):
                print("✅ Успешно добавлено.")
            else:
                print("❌ Не удалось добавить лицо.")
        else:
            print("❌ Имя не введено.")

if __name__ == "__main__":
    main()
