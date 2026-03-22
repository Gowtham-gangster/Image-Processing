import requests
import time

url = "http://127.0.0.1:8000/predict-image"
image_path = "dataset/test/person1/1.jpg"

print(f"Testing API with {image_path}...")

while True:
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            r = requests.post(url, files=files)
            if r.status_code == 200:
                print("API Response:", r.json())
                break
            else:
                print(f"API returned {r.status_code}: {r.text}")
                break
    except requests.exceptions.ConnectionError:
        print("API not yet available, waiting 2s...")
        time.sleep(2)
