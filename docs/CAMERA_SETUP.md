# Camera Setup Guide

## Overview

The system supports multiple camera sources:
1. **Local Cameras** (Webcams) - Temporary solution for testing
2. **RTSP Streams** (CCTV Cameras) - Production solution with unique identifiers
3. **HTTP Streams** - Alternative streaming protocol

## Current Configuration

The system is currently configured with:
- **CAM001**: System Webcam (Local Camera) - **ENABLED** ✅
- **CAM002**: CCTV Camera 1 (RTSP) - DISABLED (placeholder)
- **CAM003**: CCTV Camera 2 (RTSP) - DISABLED (placeholder)

## Using System Webcam (Temporary)

The system webcam (CAM001) is enabled by default for immediate testing:

1. Navigate to **Live Feed** in the dashboard
2. Select **CAM001 - System Webcam** from the dropdown
3. Click **Start Stream** to begin live detection
4. The integrated camera will be accessed via OpenCV (index 0)

### Features Available:
- ✅ Real-time person detection
- ✅ Face recognition with mask detection
- ✅ Live statistics overlay
- ✅ Snapshot capture
- ✅ Event logging

## Connecting CCTV Cameras (Production)

### Step 1: Gather Camera Information

For each CCTV camera, collect:
- **RTSP URL**: `rtsp://username:password@ip:port/stream`
- **Device ID**: Unique identifier from manufacturer
- **MAC Address**: Network MAC address
- **Serial Number**: Physical device serial number
- **Location**: Physical installation location

### Step 2: Configure Camera

1. Navigate to **Camera Settings** in the dashboard
2. Click **Add Camera** or **Edit** existing placeholder
3. Fill in the details:

```
Camera ID: CAM002
Name: Main Entrance Camera
Type: RTSP Stream
Source: rtsp://admin:password@192.168.1.100:554/stream
Location: Main Entrance
Description: CCTV camera monitoring main entrance
Device ID: [Your device ID]
MAC Address: 00:11:22:33:44:55
Serial Number: [Your serial number]
Enabled: ✓
```

4. Click **Update Camera** or **Add Camera**

### Step 3: Test Connection

1. Go to **Live Feed**
2. Select your newly configured camera
3. Click **Start Stream**
4. Verify video feed appears with detection overlays

## RTSP URL Formats

Different CCTV manufacturers use different RTSP URL formats:

### Hikvision
```
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
```

### Dahua
```
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

### Axis
```
rtsp://admin:password@192.168.1.100:554/axis-media/media.amp
```

### Generic
```
rtsp://username:password@ip:port/stream
```

## Camera Configuration File

Cameras are stored in `camera_config.json`:

```json
{
  "cameras": [
    {
      "id": "CAM001",
      "name": "System Webcam",
      "type": "local",
      "source": 0,
      "enabled": true,
      "location": "Local System",
      "description": "Integrated system camera (temporary)"
    },
    {
      "id": "CAM002",
      "name": "CCTV Camera 1",
      "type": "rtsp",
      "source": "rtsp://admin:password@192.168.1.100:554/stream",
      "enabled": false,
      "location": "Main Entrance",
      "description": "CCTV camera at main entrance",
      "device_id": "HIK-12345",
      "mac_address": "00:11:22:33:44:55",
      "serial_number": "SN123456789"
    }
  ]
}
```

## API Endpoints

### Camera Management

- `GET /cameras` - List all cameras
- `GET /cameras/enabled` - List enabled cameras only
- `GET /cameras/{camera_id}` - Get specific camera
- `POST /cameras` - Add new camera
- `PUT /cameras/{camera_id}` - Update camera
- `DELETE /cameras/{camera_id}` - Delete camera
- `GET /cameras/{camera_id}/status` - Check if camera is active

### Video Streaming

- `GET /video/feed?camera_id={id}` - MJPEG stream with detection
- `GET /video/status?camera_id={id}` - Check stream status

## Troubleshooting

### Webcam Not Working

1. Check if another application is using the camera
2. Try different camera indices (0, 1, 2) in source field
3. Verify camera permissions in Windows settings
4. Check backend logs for OpenCV errors

### RTSP Stream Not Working

1. Verify RTSP URL is correct
2. Test URL with VLC Media Player first
3. Check network connectivity to camera
4. Verify username/password credentials
5. Ensure firewall allows RTSP traffic (port 554)
6. Check camera supports RTSP protocol

### Detection Not Working

1. Verify camera is enabled in settings
2. Check backend logs for errors
3. Ensure AI models are loaded correctly
4. Verify sufficient lighting for face detection
5. Check camera resolution and frame rate

## Network Configuration

### Port Requirements

- **Backend API**: 8000
- **Frontend Dashboard**: 5173
- **RTSP**: 554 (default)
- **HTTP Stream**: 80/8080 (varies)

### Firewall Rules

Allow inbound/outbound traffic on:
- Port 554 (RTSP)
- Camera IP addresses
- Local network subnet

## Security Best Practices

1. **Change Default Passwords**: Never use default camera credentials
2. **Use Strong Passwords**: Minimum 12 characters with mixed case, numbers, symbols
3. **Network Segmentation**: Place cameras on separate VLAN
4. **Regular Updates**: Keep camera firmware updated
5. **Access Control**: Limit who can access camera settings
6. **Encryption**: Use RTSPS (RTSP over TLS) when available
7. **Monitor Access**: Review camera access logs regularly

## Migration Path

### Current: Webcam (Testing)
- ✅ Quick setup for development
- ✅ No network configuration needed
- ❌ Limited to single camera
- ❌ Not suitable for production

### Future: CCTV Network (Production)
- ✅ Multiple camera support
- ✅ Professional surveillance quality
- ✅ Unique device identification
- ✅ Remote monitoring capability
- ✅ Scalable architecture

## Next Steps

1. **Test with webcam** to verify system functionality
2. **Gather CCTV information** (URLs, credentials, IDs)
3. **Configure CCTV cameras** in Camera Settings
4. **Test each camera** individually
5. **Enable production cameras** and disable webcam
6. **Monitor performance** and adjust settings as needed

## Support

For issues or questions:
- Check backend logs: `venv310/Scripts/python.exe start_backend.py`
- Review API documentation: `http://localhost:8000/docs`
- Verify camera configuration: `camera_config.json`
- Test RTSP URLs with VLC Media Player
