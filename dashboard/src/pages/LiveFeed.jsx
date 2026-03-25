import { useState, useEffect, useRef } from 'react'

import { API } from '../config'

function eventType(ev) {
  if (ev.person_id === 'SPOOF DETECTED') return 'spoof'
  if (!ev.is_known) return 'unknown'
  return 'known'
}

function formatTime(iso) {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export default function LiveFeed() {
  const [events, setEvents] = useState([])
  const [connected, setConnected] = useState(false)
  const [videoActive, setVideoActive] = useState(false)
  const [cameraId, setCameraId] = useState(0)
  const feedRef = useRef(null)
  const videoRef = useRef(null)

  useEffect(() => {
    const es = new EventSource(`${API}/events/stream`)
    es.onopen = () => setConnected(true)
    es.onerror = () => setConnected(false)
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        setEvents(prev => [data, ...prev].slice(0, 80))
      } catch (_) {}
    }
    return () => es.close()
  }, [])

  const startVideo = () => {
    if (videoRef.current) {
      videoRef.current.src = `${API}/video/feed?camera_id=${cameraId}`
      setVideoActive(true)
    }
  }

  const stopVideo = () => {
    if (videoRef.current) {
      videoRef.current.src = ''
      setVideoActive(false)
    }
  }

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 className="page-title">Live Feed</h1>
          <p className="page-subtitle">Real-time video stream and detection events</p>
        </div>
        <div className="flex-row">
          <div className={`pulse-dot`} style={connected ? {} : { background: 'var(--red)', animation: 'none' }} />
          <span style={{ fontSize: 12, color: connected ? 'var(--green)' : 'var(--red)' }}>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Video Stream Card */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-header">
          <span className="card-title">Live Camera Stream</span>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <label style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
              Camera ID:
              <input
                type="number"
                value={cameraId}
                onChange={(e) => setCameraId(parseInt(e.target.value) || 0)}
                min="0"
                max="10"
                disabled={videoActive}
                style={{
                  marginLeft: '0.5rem',
                  width: '60px',
                  padding: '0.25rem 0.5rem',
                  borderRadius: '4px',
                  border: '1px solid var(--border)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)'
                }}
              />
            </label>
            {!videoActive ? (
              <button className="btn btn-primary btn-sm" onClick={startVideo}>
                Start Stream
              </button>
            ) : (
              <button className="btn btn-danger btn-sm" onClick={stopVideo}>
                Stop Stream
              </button>
            )}
          </div>
        </div>

        <div style={{ 
          background: '#000', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          minHeight: '400px',
          position: 'relative'
        }}>
          {videoActive ? (
            <img
              ref={videoRef}
              alt="Live camera feed"
              style={{
                maxWidth: '100%',
                maxHeight: '600px',
                width: 'auto',
                height: 'auto',
                display: 'block'
              }}
            />
          ) : (
            <div style={{ 
              textAlign: 'center', 
              color: 'var(--text-secondary)',
              padding: '2rem'
            }}>
              <svg 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="1.5"
                style={{ width: '64px', height: '64px', margin: '0 auto 1rem' }}
              >
                <rect x="2" y="4" width="20" height="16" rx="2"/>
                <circle cx="12" cy="12" r="3"/>
              </svg>
              <p>Click "Start Stream" to begin live video feed</p>
              <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                Camera {cameraId} will be used
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Event Stream Card */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Detection Events</span>
          <button className="btn btn-ghost btn-sm" onClick={() => setEvents([])}>Clear</button>
        </div>

        {events.length === 0 ? (
          <div className="empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="12" cy="12" r="10"/>
              <path d="M12 6v6l4 2"/>
            </svg>
            Waiting for detection events…
          </div>
        ) : (
          <div className="live-feed" ref={feedRef}>
            {events.map((ev, i) => {
              const type = eventType(ev)
              return (
                <div className="event-card-wrapper" key={i}>
                  <div className={`event-row ${type}`}>
                    <span className="event-time">{formatTime(ev.timestamp)}</span>
                    <span className="event-cam">{ev.camera_id}</span>
                    <span className="event-name">{ev.name || ev.person_id}</span>
                    <span className={`badge ${type}`}>{type.toUpperCase()}</span>
                    {ev.is_masked && <span className="badge masked">MASKED</span>}
                    <span className="event-conf">{(ev.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
