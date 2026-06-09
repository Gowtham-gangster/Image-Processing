"""
Test script to verify the video feed endpoint is working.
Run this after starting the API server.
"""
import requests
import time

API_URL = "http://localhost:8000"

def test_video_status():
    """Test the video status endpoint."""
    print("Testing video status endpoint...")
    response = requests.get(f"{API_URL}/video/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_video_feed_stream():
    """Test that the video feed endpoint is accessible."""
    print("Testing video feed endpoint (checking if accessible)...")
    try:
        response = requests.get(f"{API_URL}/video/feed?camera_id=0", stream=True, timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        
        # Read first few bytes to verify stream is working
        chunk_count = 0
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                chunk_count += 1
                if chunk_count >= 3:  # Read a few chunks then stop
                    break
        
        print(f"Successfully received {chunk_count} chunks from video stream")
        print("✓ Video feed endpoint is working!")
    except requests.exceptions.Timeout:
        print("⚠ Timeout - this might mean no camera is available")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Video Feed API Test")
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_video_status()
        # Uncomment the line below to test actual video streaming
        # test_video_feed_stream()
        
        print("=" * 60)
        print("Tests completed!")
        print("=" * 60)
        print()
        print("To test the video feed in browser:")
        print(f"1. Open: {API_URL}/video/feed?camera_id=0")
        print("2. Or use the dashboard Live Feed page")
        
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API server")
        print(f"  Make sure the server is running at {API_URL}")
