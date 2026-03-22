import { useState, useEffect } from 'react'

const API = 'http://localhost:8000'

function AlertBadge({ type }) {
  if (type === 'unknown_person') return <span className="badge unknown" style={{ backgroundColor: 'rgba(239, 68, 68, 0.2)', color: '#ef4444' }}>UNKNOWN</span>
  if (type === 'unmasked_person') return <span className="badge spoof" style={{ backgroundColor: 'rgba(245, 158, 11, 0.2)', color: '#f59e0b' }}>NO MASK</span>
  if (type === 'spoof_attempt') return <span className="badge spoof" style={{ backgroundColor: 'rgba(239, 68, 68, 0.2)', color: '#ef4444' }}>SPOOF</span>
  return <span className="badge">{type}</span>
}

export default function AlertHistory() {
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(0)
  const limit = 20

  const load = async (pg = 0) => {
    setLoading(true)
    try {
      const r = await fetch(`${API}/alerts/history?limit=${limit}&offset=${pg * limit}`)
      const data = await r.json()
      setAlerts(Array.isArray(data) ? data : [])
    } catch (_) {}
    setLoading(false)
  }

  useEffect(() => { load(page) }, [page])

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Alert History</h1>
        <p className="page-subtitle">Chronological record of all dispatched security alerts</p>
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Dispatched Alerts</span>
          <button className="btn btn-ghost btn-sm" onClick={() => load(page)}>↻ Refresh</button>
        </div>

        {loading ? (
          <div className="empty-state">Loading…</div>
        ) : alerts.length === 0 ? (
          <div className="empty-state">No security alerts recorded.</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Time (UTC)</th>
                <th>Alert Type</th>
                <th>Camera Engine</th>
                <th>Person ID</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map(al => (
                <tr key={al.id}>
                  <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{al.timestamp}</td>
                  <td><AlertBadge type={al.alert_type} /></td>
                  <td style={{ color: 'var(--accent)' }}>{al.camera_id}</td>
                  <td style={{ color: 'var(--text-primary)' }}>{al.person_id}</td>
                  <td>{(al.confidence * 100).toFixed(1)}%</td>
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
          <button className="btn btn-ghost btn-sm" disabled={alerts.length < limit} onClick={() => setPage(p => p + 1)}>
            Next →
          </button>
        </div>
      </div>
    </div>
  )
}
