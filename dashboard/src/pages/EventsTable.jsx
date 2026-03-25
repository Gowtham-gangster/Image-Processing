import { useState, useEffect } from 'react'

import { API } from '../config'

function Badge({ is_known, is_masked, is_live }) {
  if (!is_live) return <span className="badge spoof">SPOOF</span>
  if (!is_known) return <span className="badge unknown">UNKNOWN</span>
  if (is_masked) return <span className="badge masked">MASKED</span>
  return <span className="badge known">KNOWN</span>
}

export default function EventsTable() {
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)
  const limit = 20

  const load = async (pg = 0) => {
    setLoading(true)
    try {
      const r = await fetch(`${API}/events?limit=${limit}&offset=${pg * limit}`)
      const data = await r.json()
      setEvents(data.events || [])
    } catch (_) {}
    setLoading(false)
  }

  useEffect(() => { load(page) }, [page])

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Event History</h1>
        <p className="page-subtitle">Searchable detection log from the SQLite database</p>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Recent Detections</span>
          <button className="btn btn-ghost btn-sm" onClick={() => load(page)}>↻ Refresh</button>
        </div>

        {loading ? (
          <div className="empty-state">Loading…</div>
        ) : events.length === 0 ? (
          <div className="empty-state">No events recorded yet.</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Camera</th>
                <th>Person ID</th>
                <th>Confidence</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {events.map(ev => (
                <tr key={ev.id}>
                  <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{ev.timestamp}</td>
                  <td style={{ color: 'var(--accent)' }}>{ev.camera_id}</td>
                  <td style={{ color: 'var(--text-primary)' }}>{ev.person_id}</td>
                  <td>{(ev.confidence * 100).toFixed(1)}%</td>
                  <td>
                    <Badge
                      is_known={!!ev.is_known}
                      is_masked={!!ev.is_masked}
                      is_live={ev.person_id !== 'SPOOF'}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <div className="flex-row" style={{ marginTop: 14, justifyContent: 'flex-end', gap: 8 }}>
          <button className="btn btn-ghost btn-sm" disabled={page === 0} onClick={() => setPage(p => Math.max(0, p - 1))}>
            ← Prev
          </button>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Page {page + 1}</span>
          <button className="btn btn-ghost btn-sm" disabled={events.length < limit} onClick={() => setPage(p => p + 1)}>
            Next →
          </button>
        </div>
      </div>
    </div>
  )
}
