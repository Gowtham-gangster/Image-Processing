#!/usr/bin/env python
"""
Run the backend API directly on port 8000.
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Backend API Server")
    print("Port: http://localhost:8000")
    print("=" * 60)
    print("\n✓ Dashboard available at: http://localhost:5173")
    print("✓ API Docs: http://localhost:8000/docs\n")
    
    uvicorn.run(
        "api.index:app",
        host="127.0.0.1",
        port=8000,
        reload=False,  # Disabled to avoid multiprocessing issues
        log_level="info"
    )
