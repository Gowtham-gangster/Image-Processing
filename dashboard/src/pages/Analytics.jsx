import { useState, useEffect } from 'react'

import { API } from '../config'

function Bar({ value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <div style={{ flex: 1, height: 6, background: 'var(--bg-elevated)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.4s ease' }} />
      </div>
      <span style={{ fontSize: 12, color: 'var(--text-secondary)', minWidth: 32 }}>{pct}%</span>
    </div>
  )
}

export default function Analytics() {
  const [stats, setStats]   = useState(null)
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const [sRes, eRes] = await Promise.all([
          fetch(`${API}/events/stats`),
          fetch(`${API}/events?limit=200`),
        ])
        setStats(await sRes.json())
        const d = await eRes.json()
        setEvents(d.events || [])
      } catch (_) {}
      setLoading(false)
    }
    load()
    const t = setInterval(load, 10000)
    return () => clearInterval(t)
  }, [])

  // Build hourly buckets for today
  const hourBuckets = Array(24).fill(0)
  events.forEach(ev => {
    const h = new Date(ev.timestamp.replace(' ', 'T')).getHours()
    if (!isNaN(h)) hourBuckets[h]++
  })
  const hourMax = Math.max(...hourBuckets, 1)

  const known   = events.filter(e => e.is_known).length
  const unknown = events.filter(e => !e.is_known).length
  const total   = events.length || 1

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Analytics</h1>
        <p className="page-subtitle">Detection volume and identity breakdown</p>
      </div>

      {loading ? (
        <div className="empty-state">Loading…</div>
      ) : (
        <>
          {/* Stat cards */}
          <div className="grid-4" style={{ marginBottom: 24 }}>
            <div className="stat-card">
              <span className="stat-label">Total Events</span>
              <span className="stat-value accent">{stats?.total ?? 0}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Today</span>
              <span className="stat-value">{stats?.today ?? 0}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Unknown Persons</span>
              <span className="stat-value red">{stats?.unknown ?? 0}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Unmasked</span>
              <span className="stat-value orange">{stats?.unmasked ?? 0}</span>
            </div>
          </div>

          <div className="grid-2">
            {/* Detection breakdown */}
            <div className="card">
              <div className="card-header"><span className="card-title">Identity Breakdown</span></div>
              <div className="flex-col">
                <div>
                  <div style={{ display:'flex', justifyContent:'space-between', marginBottom:6 }}>
                    <span style={{ fontSize:13, color:'var(--text-secondary)' }}>Known Persons</span>
                    <span style={{ fontSize:13, color:'var(--green)', fontWeight:600 }}>{known}</span>
                  </div>
                  <Bar value={known} max={total} color="var(--green)" />
                </div>
                <div>
                  <div style={{ display:'flex', justifyContent:'space-between', marginBottom:6 }}>
                    <span style={{ fontSize:13, color:'var(--text-secondary)' }}>Unknown Persons</span>
                    <span style={{ fontSize:13, color:'var(--red)', fontWeight:600 }}>{unknown}</span>
                  </div>
                  <Bar value={unknown} max={total} color="var(--red)" />
                </div>
                <div>
                  <div style={{ display:'flex', justifyContent:'space-between', marginBottom:6 }}>
                    <span style={{ fontSize:13, color:'var(--text-secondary)' }}>Unmasked</span>
                    <span style={{ fontSize:13, color:'var(--orange)', fontWeight:600 }}>{stats?.unmasked ?? 0}</span>
                  </div>
                  <Bar value={stats?.unmasked ?? 0} max={total} color="var(--orange)" />
                </div>
              </div>
            </div>

            {/* Hourly activity */}
            <div className="card">
              <div className="card-header"><span className="card-title">Hourly Activity (Today)</span></div>
              <div style={{ display:'flex', alignItems:'flex-end', gap:3, height:120 }}>
                {hourBuckets.map((count, h) => (
                  <div key={h} style={{ flex:1, display:'flex', flexDirection:'column', alignItems:'center', gap:4 }}>
                    <div
                      style={{
                        flex:1,
                        width:'100%',
                        display:'flex',
                        alignItems:'flex-end',
                      }}
                    >
                      <div
                        style={{
                          width:'100%',
                          height: `${Math.round((count / hourMax) * 100)}%`,
                          minHeight: count > 0 ? 4 : 0,
                          background: 'var(--accent)',
                          borderRadius:'3px 3px 0 0',
                          opacity: count > 0 ? 1 : 0.15,
                          transition:'height 0.3s ease',
                        }}
                        title={`${h}:00 — ${count} events`}
                      />
                    </div>
                    {h % 4 === 0 && (
                      <span style={{ fontSize:9, color:'var(--text-muted)' }}>{String(h).padStart(2,'0')}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
