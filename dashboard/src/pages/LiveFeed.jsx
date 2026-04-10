import { useState, useEffect, useRef } from 'react'
import { API } from '../config'

function eventType(ev) {
  if (ev.person_id === 'SPOOF DETECTED' || ev.person_id === 'SPOOF') return 'spoof'
  if (!ev.is_known) return 'unknown'
  return 'known'
}

function formatTime(iso) {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function formatRelativeTime(iso) {
  const now = new Date()
  const then = new Date(iso)
  const diffMs = now - then
  const diffSec = Math.floor(diffMs / 1000)
  const diffMin = Math.floor(diffSec / 60)
  const diffHr = Math.floor(diffMin / 60)
  
  if (diffSec < 10) return 'Just now'
  if (diffSec < 60) return `${diffSec}s ago`
  if (diffMin < 60) return `${diffMin}m ago`
  if (diffHr < 24) return `${diffHr}h ago`
  return `${Math.floor(diffHr / 24)}d ago`
}

export default function LiveFeed() {
  const [events, setEvents] = useState([])
  const [connected, setConnected] = useState(false)
  const [videoActive, setVideoActive] = useState(false)
  const [cameras, setCameras] = useState([])
  const [selectedCamera, setSelectedCamera] = useState('CAM001')
  const [stats, setStats] = useState({ known: 0, unknown: 0, masked: 0 })
  const [showStats, setShowStats] = useState(true)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [snapshotPreview, setSnapshotPreview] = useState(null)
  const feedRef = useRef(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  // Load cameras on mount
  useEffect(() => {
    loadCameras()
  }, [])

  const loadCameras = async () => {
    try {
      const res = await fetch(`${API}/cameras`)
      const data = await res.json()
      setCameras(data.cameras || [])
      
      // Select first enabled camera by default
      const enabledCam = data.cameras.find(c => c.enabled)
      if (enabledCam) {
        setSelectedCamera(enabledCam.id)
      }
    } catch (err) {
      console.error('Failed to load cameras:', err)
      setError('Failed to load camera list')
    } finally {
      setLoading(false)
    }
  }

  // SSE connection for real-time events
  useEffect(() => {
    const es = new EventSource(`${API}/events/stream`)
    es.onopen = () => {
      setConnected(true)
      setError(null)
    }
    es.onerror = () => {
      setConnected(false)
      setError('Event stream disconnected')
    }
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        setEvents(prev => [data, ...prev].slice(0, 100))
        
        // Update stats
        setStats(prev => {
          const newStats = { ...prev }
          const type = eventType(data)
          if (type === 'known') newStats.known++
          else if (type === 'unknown') newStats.unknown++
          // Spoof events are not counted in stats
          if (data.is_masked) newStats.masked++
          return newStats
        })
      } catch (_) {}
    }
    return () => es.close()
  }, [])

  const startVideo = async () => {
    try {
      setError(null)
      setVideoActive(true)
      // Set src after a small delay to ensure the img element is rendered
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.src = `${API}/video/feed?camera_id=${selectedCamera}`
        }
      }, 100)
    } catch (err) {
      setError(`Failed to start camera: ${err.message}`)
      setVideoActive(false)
    }
  }

  const stopVideo = () => {
    if (videoRef.current) {
      videoRef.current.src = ''
      setVideoActive(false)
    }
  }

  const takeSnapshot = async () => {
    try {
      setError(null)
      
      // Fetch snapshot from backend
      const response = await fetch(`${API}/video/snapshot?camera_id=${selectedCamera}`)
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to capture snapshot' }))
        throw new Error(errorData.detail || 'Failed to capture snapshot')
      }
      
      // Convert response to blob and create object URL
      const blob = await response.blob()
      const dataUrl = URL.createObjectURL(blob)
      
      setSnapshotPreview(dataUrl)
    } catch (err) {
      setError(`Snapshot failed: ${err.message}`)
      console.error('Snapshot error:', err)
    }
  }

  const downloadSnapshot = () => {
    if (!snapshotPreview) return
    
    const a = document.createElement('a')
    a.href = snapshotPreview
    a.download = `snapshot_${selectedCamera}_${new Date().getTime()}.jpg`
    a.click()
    
    // Cleanup blob URL
    URL.revokeObjectURL(snapshotPreview)
    setSnapshotPreview(null)
  }

  const closeSnapshotPreview = () => {
    // Cleanup blob URL
    if (snapshotPreview) {
      URL.revokeObjectURL(snapshotPreview)
    }
    setSnapshotPreview(null)
  }

  const clearEvents = () => {
    setEvents([])
    setStats({ known: 0, unknown: 0, masked: 0 })
  }

  const selectedCameraInfo = cameras.find(c => c.id === selectedCamera)
  const enabledCameras = cameras.filter(c => c.enabled)

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 className="page-title">Live Feed</h1>
          <p className="page-subtitle">Real-time video stream and detection events</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div 
            className={`pulse-dot`} 
            style={connected ? {} : { background: 'var(--red)', animation: 'none' }} 
          />
          <span style={{ fontSize: 12, color: connected ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {error && (
        <div style={{
          padding: '12px 16px',
          background: 'var(--red)',
          color: 'white',
          borderRadius: 8,
          marginBottom: 16,
          display: 'flex',
          alignItems: 'center',
          gap: 8
        }}>
          <span>⚠️</span>
          <span>{error}</span>
          <button 
            onClick={() => setError(null)}
            style={{ marginLeft: 'auto', background: 'transparent', border: 'none', color: 'white', cursor: 'pointer', fontSize: 18 }}
          >
            ×
          </button>
        </div>
      )}

      {/* Camera Info Banner */}
      {/* Removed - camera info is shown in dropdown and video overlay */}

      {/* Video Stream Card */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <span className="card-title">📹 Live Camera Stream</span>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 6 }}>
              Camera:
              <select
                value={selectedCamera}
                onChange={(e) => setSelectedCamera(e.target.value)}
                disabled={videoActive || loading}
                style={{
                  padding: '0.5rem 0.75rem',
                  borderRadius: '6px',
                  border: '1px solid var(--border)',
                  background: 'var(--bg-secondary)',
                  color: '#ffffff',
                  fontSize: '15px',
                  fontWeight: 600,
                  minWidth: 200,
                  cursor: 'pointer'
                }}
              >
                {cameras.map(cam => (
                  <option 
                    key={cam.id} 
                    value={cam.id} 
                    disabled={!cam.enabled} 
                    style={{ 
                      fontSize: '15px', 
                      fontWeight: 600,
                      background: '#ffffff',
                      color: '#000000',
                      padding: '8px'
                    }}
                  >
                    {cam.name} {!cam.enabled ? '(Disabled)' : ''}
                  </option>
                ))}
              </select>
            </label>
            
            {videoActive && (
              <>
                <button 
                  className="btn btn-ghost btn-sm" 
                  onClick={takeSnapshot}
                  title="Take snapshot"
                >
                  📸 Snapshot
                </button>
                <button 
                  className="btn btn-ghost btn-sm" 
                  onClick={() => setShowStats(!showStats)}
                  title="Toggle statistics"
                >
                  {showStats ? '👁️ Hide Stats' : '👁️ Show Stats'}
                </button>
              </>
            )}
            
            {!videoActive ? (
              <button 
                className="btn btn-primary btn-sm" 
                onClick={startVideo}
                disabled={!selectedCameraInfo?.enabled || loading}
              >
                ▶️ Start Stream
              </button>
            ) : (
              <button className="btn btn-danger btn-sm" onClick={stopVideo}>
                ⏹️ Stop Stream
              </button>
            )}
          </div>
        </div>

        <div style={{ 
          background: '#000', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          minHeight: '480px',
          position: 'relative',
          borderRadius: '0 0 8px 8px',
          overflow: 'hidden'
        }}>
          {videoActive ? (
            <>
              <img
                ref={videoRef}
                alt="Live camera feed"
                style={{
                  maxWidth: '100%',
                  maxHeight: '720px',
                  width: 'auto',
                  height: 'auto',
                  display: 'block'
                }}
                onLoad={() => console.log('Video stream loaded')}
                onError={(e) => {
                  console.error('Video stream error:', e)
                  setError('Failed to load video stream. Check camera connection.')
                  setVideoActive(false)
                }}
              />
              
              {/* Statistics Overlay */}
              {showStats && (
                <div style={{
                  position: 'absolute',
                  top: 16,
                  right: 16,
                  background: 'rgba(0, 0, 0, 0.85)',
                  backdropFilter: 'blur(10px)',
                  padding: '16px',
                  borderRadius: 12,
                  minWidth: 200,
                  border: '1px solid rgba(255, 255, 255, 0.1)'
                }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: 'white', marginBottom: 12, letterSpacing: 0.5 }}>
                    📊 LIVE STATISTICS
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.7)' }}>✅ Known</span>
                      <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--green)' }}>{stats.known}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.7)' }}>❓ Unknown</span>
                      <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--red)' }}>{stats.unknown}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.7)' }}>😷 Masked</span>
                      <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--blue)' }}>{stats.masked}</span>
                    </div>
                    <div style={{ 
                      marginTop: 8, 
                      paddingTop: 10, 
                      borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}>
                      <span style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.7)' }}>Total</span>
                      <span style={{ fontSize: 16, fontWeight: 700, color: 'white' }}>
                        {stats.known + stats.unknown}
                      </span>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Camera Info Badge */}
              <div style={{
                position: 'absolute',
                top: 16,
                left: 16,
                background: 'rgba(0, 0, 0, 0.85)',
                backdropFilter: 'blur(10px)',
                padding: '8px 14px',
                borderRadius: 8,
                border: '1px solid rgba(255, 255, 255, 0.1)',
                display: 'flex',
                alignItems: 'center',
                gap: 8
              }}>
                <div className="pulse-dot" style={{ width: 8, height: 8 }} />
                <span style={{ fontSize: 12, color: 'white', fontWeight: 600 }}>
                  {selectedCameraInfo?.name || selectedCamera}
                </span>
              </div>
            </>
          ) : (
            <div style={{ 
              textAlign: 'center', 
              color: 'var(--text-secondary)',
              padding: '3rem'
            }}>
              <svg 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="1.5"
                style={{ width: '80px', height: '80px', margin: '0 auto 1.5rem', opacity: 0.5 }}
              >
                <rect x="2" y="4" width="20" height="16" rx="2"/>
                <circle cx="12" cy="12" r="3"/>
                <path d="M7 4v-2M17 4v-2"/>
              </svg>
              <p style={{ fontSize: '1.1rem', marginBottom: '0.5rem', color: 'var(--text-primary)' }}>
                No Active Stream
              </p>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                {selectedCameraInfo?.enabled 
                  ? `Click "Start Stream" to begin live video feed from ${selectedCameraInfo.name}`
                  : 'Selected camera is disabled. Please enable it or select another camera.'}
              </p>
              {enabledCameras.length === 0 && (
                <p style={{ fontSize: '0.875rem', color: 'var(--red)', marginTop: 12 }}>
                  ⚠️ No cameras are currently enabled
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Hidden canvas for snapshots */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Snapshot Preview Modal */}
      {snapshotPreview && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.9)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: 20
        }}>
          <div style={{
            background: 'var(--bg-elevated)',
            borderRadius: 12,
            maxWidth: '90vw',
            maxHeight: '90vh',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)'
          }}>
            {/* Modal Header */}
            <div style={{
              padding: '16px 20px',
              borderBottom: '1px solid var(--border)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <div>
                <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>
                  📸 Snapshot Preview
                </h3>
                <p style={{ margin: '4px 0 0 0', fontSize: 13, color: 'var(--text-secondary)' }}>
                  {selectedCameraInfo?.name || selectedCamera} • {new Date().toLocaleString()}
                </p>
              </div>
              <button
                onClick={closeSnapshotPreview}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--text-secondary)',
                  fontSize: 24,
                  cursor: 'pointer',
                  padding: 4,
                  lineHeight: 1
                }}
                title="Close"
              >
                ×
              </button>
            </div>

            {/* Modal Body - Image */}
            <div style={{
              padding: 20,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              maxHeight: 'calc(90vh - 140px)',
              overflow: 'auto'
            }}>
              <img
                src={snapshotPreview}
                alt="Snapshot preview"
                style={{
                  maxWidth: '100%',
                  maxHeight: '100%',
                  borderRadius: 8,
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
                }}
              />
            </div>

            {/* Modal Footer */}
            <div style={{
              padding: '16px 20px',
              borderTop: '1px solid var(--border)',
              display: 'flex',
              gap: 12,
              justifyContent: 'flex-end'
            }}>
              <button
                className="btn btn-ghost"
                onClick={closeSnapshotPreview}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={downloadSnapshot}
              >
                💾 Download Snapshot
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Detection Events Grid */}
      <div className="grid-2" style={{ gap: '1.5rem' }}>
        {/* Recent Events */}
        <div className="card" style={{ gridColumn: 'span 2' }}>
          <div className="card-header">
            <span className="card-title">🔔 Recent Detection Events</span>
            <div style={{ display: 'flex', gap: 8 }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                {events.length} events
              </span>
              <button className="btn btn-ghost btn-sm" onClick={clearEvents}>
                Clear All
              </button>
            </div>
          </div>

          {events.length === 0 ? (
            <div className="empty-state" style={{ padding: '3rem' }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ width: 48, height: 48, margin: '0 auto 1rem' }}>
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 6v6l4 2"/>
              </svg>
              <p>Waiting for detection events…</p>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginTop: 8 }}>
                Events will appear here when persons are detected
              </p>
            </div>
          ) : (
            <div style={{ 
              display: 'grid', 
              gap: 12, 
              maxHeight: 600, 
              overflowY: 'auto',
              padding: '4px 0'
            }}>
              {events.map((ev, i) => {
                const type = eventType(ev)
                const bgColor = type === 'known' ? 'var(--green)' : type === 'spoof' ? 'var(--orange)' : 'var(--red)'
                
                return (
                  <div 
                    key={i}
                    style={{
                      background: 'var(--bg-elevated)',
                      border: `1px solid ${bgColor}20`,
                      borderLeft: `4px solid ${bgColor}`,
                      borderRadius: 8,
                      padding: '14px 16px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 16,
                      transition: 'all 0.2s ease',
                      animation: i === 0 ? 'slideIn 0.3s ease' : 'none'
                    }}
                  >
                    {/* Avatar */}
                    <div style={{
                      width: 48,
                      height: 48,
                      borderRadius: '50%',
                      background: `${bgColor}20`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 20,
                      flexShrink: 0
                    }}>
                      {type === 'known' ? '✅' : type === 'spoof' ? '🎭' : '❓'}
                    </div>
                    
                    {/* Details */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                        <span style={{ 
                          fontSize: 15, 
                          fontWeight: 600, 
                          color: 'var(--text-primary)',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}>
                          {ev.name || ev.person_id}
                        </span>
                        <span className={`badge ${type}`} style={{ fontSize: 10, padding: '2px 8px' }}>
                          {type.toUpperCase()}
                        </span>
                        {ev.is_masked && (
                          <span className="badge masked" style={{ fontSize: 10, padding: '2px 8px' }}>
                            MASKED
                          </span>
                        )}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 12, color: 'var(--text-secondary)' }}>
                        <span title={new Date(ev.timestamp).toLocaleString()}>
                          🕐 {formatRelativeTime(ev.timestamp)}
                        </span>
                        <span>📹 {ev.camera_id}</span>
                        <span style={{ color: bgColor, fontWeight: 600 }}>
                          {(ev.confidence * 100).toFixed(1)}% confidence
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
      `}</style>
    </div>
  )
}
