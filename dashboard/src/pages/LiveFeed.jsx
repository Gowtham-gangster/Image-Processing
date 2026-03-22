import { useState, useEffect, useRef } from 'react'

const API = 'http://localhost:8000'

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
  const feedRef = useRef(null)

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

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 className="page-title">Live Feed</h1>
          <p className="page-subtitle">Real-time detection events from all camera sources</p>
        </div>
        <div className="flex-row">
          <div className={`pulse-dot`} style={connected ? {} : { background: 'var(--red)', animation: 'none' }} />
          <span style={{ fontSize: 12, color: connected ? 'var(--green)' : 'var(--red)' }}>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Event Stream</span>
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
