import requests
import cv2
import numpy as np
import time

API_URL = "http://localhost:8000/recognize/image"
TEST_IMG_PATH = "test_img.jpg"

import urllib.request

def create_dummy_image():
    # Download a realistic face so the detector actually finds a bounding box
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    
    # Intentionally blur it heavily to trigger Laplacian variance < threshold
    img = cv2.GaussianBlur(img, (45, 45), 0)
    
    # Increase brightness to trigger overexposure spoof check
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=50)
    
    cv2.imwrite(TEST_IMG_PATH, img)
    return TEST_IMG_PATH

def test_api():
    print("Waiting for API to boot...")
    time.sleep(5)
    
    # 1. Check health
    health = requests.get("http://localhost:8000/health")
    print("Health:", health.json())
    
    # 2. Test recognition endpoint
    img_path = create_dummy_image()
    with open(img_path, "rb") as f:
        print("Sending image to recognition endpoint...")
        response = requests.post(API_URL, files={"file": f})
        
    print("Status:", response.status_code)
    try:
        print("Response:", response.json())
    except:
        print("Raw:", response.text)

if __name__ == "__main__":
    test_api()
