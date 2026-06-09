"""Quick test to check if camera can be opened"""
import cv2

print("Testing camera access...")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("✓ Camera opened successfully!")
    ret, frame = cap.read()
    if ret:
        print(f"✓ Frame captured: {frame.shape}")
    else:
        print("✗ Failed to read frame")
    cap.release()
else:
    print("✗ Failed to open camera")
    print("Possible issues:")
    print("  - Camera is being used by another application")
    print("  - No camera connected")
    print("  - Camera permissions not granted")
