"""
Test RTSP connection to Tapo camera
"""
import cv2
import sys

# Test different RTSP URLs for Tapo C320WS
rtsp_urls = [
    "rtsp://admin:admin@192.168.1.6:554/stream1",
    "rtsp://admin:admin@192.168.1.6:554/stream2",
    "rtsp://192.168.1.6:554/stream1",
    "rtsp://192.168.1.6:554/stream2",
]

print("Testing RTSP connections to Tapo C320WS camera...")
print("=" * 60)

for url in rtsp_urls:
    print(f"\nTesting: {url}")
    print("-" * 60)
    
    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to read a frame with timeout
        print("Attempting to connect (this may take 10-15 seconds)...")
        
        if cap.isOpened():
            print("✓ Connection opened successfully")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"✓ Frame read successfully: {frame.shape}")
                print("✓✓✓ THIS URL WORKS! ✓✓✓")
                cap.release()
                sys.exit(0)
            else:
                print("✗ Failed to read frame")
        else:
            print("✗ Failed to open connection")
        
        cap.release()
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("All RTSP URLs failed. Possible issues:")
print("1. RTSP is not enabled in Tapo app")
print("2. Wrong IP address (current: 192.168.1.6)")
print("3. Camera is offline or not reachable")
print("4. Firewall blocking RTSP port 554")
print("\nTo enable RTSP on Tapo camera:")
print("1. Open Tapo app")
print("2. Select your camera")
print("3. Go to Settings → Advanced Settings")
print("4. Enable 'Camera Account' or 'RTSP'")
print("5. Set username and password")
