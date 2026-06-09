import { useState, useEffect } from 'react'

import { API } from '../config'

function Bar({ value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <div style={{ flex: 1, height: 8, background: 'var(--bg-elevated)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 4, transition: 'width 0.4s ease' }} />
      </div>
      <span style={{ fontSize: 13, color: 'var(--text-secondary)', minWidth: 40, fontWeight: 600 }}>{pct}%</span>
    </div>
  )
}

function StatCard({ label, value, color = 'var(--accent)', icon }) {
  return (
    <div className="stat-card" style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, padding: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <span style={{ fontSize: 13, color: 'var(--text-secondary)', fontWeight: 500 }}>{label}</span>
        {icon && <span style={{ fontSize: 20 }}>{icon}</span>}
      </div>
      <span style={{ fontSize: 32, fontWeight: 700, color }}>{value}</span>
    </div>
  )
}

function DonutChart({ data, total }) {
  const radius = 60
  const strokeWidth = 20
  const circumference = 2 * Math.PI * radius
  
  let currentOffset = 0
  
  return (
    <div className="donut-chart-wrap" style={{ display: 'flex', alignItems: 'center', gap: 30 }}>
      <svg width={radius * 2 + strokeWidth} height={radius * 2 + strokeWidth} style={{ transform: 'rotate(-90deg)' }}>
        <circle
          cx={radius + strokeWidth / 2}
          cy={radius + strokeWidth / 2}
          r={radius}
          fill="none"
          stroke="var(--bg-base)"
          strokeWidth={strokeWidth}
        />
        {data.map((item, i) => {
          const percentage = total > 0 ? item.value / total : 0
          const strokeDasharray = `${circumference * percentage} ${circumference}`
          const strokeDashoffset = -currentOffset
          currentOffset += circumference * percentage
          
          return (
            <circle
              key={i}
              cx={radius + strokeWidth / 2}
              cy={radius + strokeWidth / 2}
              r={radius}
              fill="none"
              stroke={item.color}
              strokeWidth={strokeWidth}
              strokeDasharray={strokeDasharray}
              strokeDashoffset={strokeDashoffset}
              style={{ transition: 'stroke-dasharray 0.5s ease' }}
            />
          )
        })}
      </svg>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {data.map((item, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 12, height: 12, borderRadius: 3, background: item.color }} />
            <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{item.label}</span>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginLeft: 'auto' }}>
              {item.value} ({total > 0 ? Math.round((item.value / total) * 100) : 0}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Analytics() {
  const [stats, setStats] = useState(null)
  const [events, setEvents] = useState([])
  const [alerts, setAlerts] = useState([])
  const [persons, setPersons] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const [sRes, eRes, aRes, pRes] = await Promise.all([
          fetch(`${API}/events/stats`),
          fetch(`${API}/events?limit=500`),
          fetch(`${API}/alerts/history?limit=500`),
          fetch(`${API}/persons`),
        ])
        setStats(await sRes.json())
        const evData = await eRes.json()
        setEvents(evData.events || [])
        const alertData = await aRes.json()
        setAlerts(Array.isArray(alertData) ? alertData : [])
        const personsData = await pRes.json()
        setPersons(personsData.persons || [])
      } catch (e) {
        console.error('Failed to load analytics:', e)
      }
      setLoading(false)
    }
    load()
    const t = setInterval(load, 15000)
    return () => clearInterval(t)
  }, [])

  // Calculate statistics
  const known = events.filter(e => e.is_known).length
  const unknown = events.filter(e => !e.is_known).length
  const masked = events.filter(e => e.is_masked).length
  const total = events.length || 1

  // Alert type breakdown
  const alertsByType = {
    unknown: alerts.filter(a => a.alert_type === 'unknown_person').length,
    masked: alerts.filter(a => a.alert_type === 'masked_person').length,
    unmasked: alerts.filter(a => a.alert_type === 'unmasked_person').length,
    spoof: alerts.filter(a => a.alert_type === 'spoof_attempt').length,
  }

  // Top detected persons
  const personCounts = {}
  events.forEach(e => {
    if (e.person_id && e.person_id !== 'Unknown' && e.person_id !== 'Unknown Person') {
      personCounts[e.person_id] = (personCounts[e.person_id] || 0) + 1
    }
  })
  const topPersons = Object.entries(personCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([id, count]) => {
      const person = persons.find(p => p.id === id)
      return { id, name: person?.name || id, count }
    })

  // Hourly activity for last 24 hours
  const hourBuckets = Array(24).fill(0)
  const now = new Date()
  events.forEach(ev => {
    try {
      const evDate = new Date(ev.timestamp)
      const hoursDiff = Math.floor((now - evDate) / (1000 * 60 * 60))
      if (hoursDiff >= 0 && hoursDiff < 24) {
        const hour = evDate.getHours()
        hourBuckets[hour]++
      }
    } catch (e) {}
  })
  const hourMax = Math.max(...hourBuckets, 1)

  // Daily activity for last 7 days
  const dayBuckets = Array(7).fill(0).map((_, i) => {
    const date = new Date()
    date.setDate(date.getDate() - (6 - i))
    return { date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), count: 0 }
  })
  
  events.forEach(ev => {
    try {
      const evDate = new Date(ev.timestamp)
      const daysDiff = Math.floor((now - evDate) / (1000 * 60 * 60 * 24))
      if (daysDiff >= 0 && daysDiff < 7) {
        dayBuckets[6 - daysDiff].count++
      }
    } catch (e) {}
  })
  const dayMax = Math.max(...dayBuckets.map(d => d.count), 1)

  // Average confidence
  const avgConfidence = events.length > 0
    ? events.reduce((sum, e) => sum + (e.confidence || 0), 0) / events.length
    : 0

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Analytics Dashboard</h1>
        <p className="page-subtitle">Comprehensive detection metrics and insights</p>
      </div>

      {loading ? (
        <div className="empty-state">Loading analytics...</div>
      ) : (
        <>
          {/* Key Metrics */}
          <div className="grid-4" style={{ marginBottom: 24 }}>
            <StatCard label="Total Detections" value={stats?.total ?? 0} color="var(--accent)" icon="📊" />
            <StatCard label="Today's Activity" value={stats?.today ?? 0} color="var(--blue)" icon="📅" />
            <StatCard label="Unknown Persons" value={stats?.unknown ?? 0} color="var(--red)" icon="🚨" />
            <StatCard label="Enrolled Persons" value={persons.length} color="var(--green)" icon="👥" />
          </div>

          <div className="grid-2" style={{ marginBottom: 24 }}>
            {/* Detection Breakdown Donut Chart */}
            <div className="card">
              <div className="card-header"><span className="card-title">Detection Breakdown</span></div>
              <div style={{ padding: '20px 0', display: 'flex', justifyContent: 'center' }}>
                <DonutChart
                  data={[
                    { label: 'Known Persons', value: known, color: 'var(--green)' },
                    { label: 'Unknown Persons', value: unknown, color: 'var(--red)' },
                    { label: 'Masked', value: masked, color: 'var(--blue)' },
                  ]}
                  total={total}
                />
              </div>
            </div>

            {/* Alert Statistics */}
            <div className="card">
              <div className="card-header"><span className="card-title">Alert Statistics</span></div>
              <div className="flex-col" style={{ gap: 16 }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>🚨 Unknown Person</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--red)' }}>{alertsByType.unknown}</span>
                  </div>
                  <Bar value={alertsByType.unknown} max={alerts.length || 1} color="var(--red)" />
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>😷 Masked Person</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--blue)' }}>{alertsByType.masked}</span>
                  </div>
                  <Bar value={alertsByType.masked} max={alerts.length || 1} color="var(--blue)" />
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>⚠️ Unmasked Person</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--orange)' }}>{alertsByType.unmasked}</span>
                  </div>
                  <Bar value={alertsByType.unmasked} max={alerts.length || 1} color="var(--orange)" />
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>🎭 Spoof Attempt</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--red)' }}>{alertsByType.spoof}</span>
                  </div>
                  <Bar value={alertsByType.spoof} max={alerts.length || 1} color="var(--red)" />
                </div>
              </div>
            </div>
          </div>

          {/* Activity Charts */}
          <div className="grid-2" style={{ marginBottom: 24 }}>
            {/* Hourly Activity */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">Hourly Activity (Last 24h)</span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Peak: {Math.max(...hourBuckets)} detections</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 140, padding: '10px 0' }}>
                {hourBuckets.map((count, h) => (
                  <div key={h} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
                    <div style={{ flex: 1, width: '100%', display: 'flex', alignItems: 'flex-end' }}>
                      <div
                        style={{
                          width: '100%',
                          height: `${Math.round((count / hourMax) * 100)}%`,
                          minHeight: count > 0 ? 6 : 0,
                          background: count > 0 ? 'var(--accent)' : 'var(--bg-base)',
                          borderRadius: '4px 4px 0 0',
                          transition: 'height 0.3s ease',
                          cursor: 'pointer',
                        }}
                        title={`${String(h).padStart(2, '0')}:00 — ${count} detections`}
                      />
                    </div>
                    {h % 3 === 0 && (
                      <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 500 }}>
                        {String(h).padStart(2, '0')}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Daily Activity */}
            <div className="card">
              <div className="card-header">
                <span className="card-title">Daily Activity (Last 7 Days)</span>
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Peak: {Math.max(...dayBuckets.map(d => d.count))} detections</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, height: 140, padding: '10px 0' }}>
                {dayBuckets.map((day, i) => (
                  <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                    <div style={{ flex: 1, width: '100%', display: 'flex', alignItems: 'flex-end' }}>
                      <div
                        style={{
                          width: '100%',
                          height: `${Math.round((day.count / dayMax) * 100)}%`,
                          minHeight: day.count > 0 ? 8 : 0,
                          background: day.count > 0 ? 'var(--green)' : 'var(--bg-base)',
                          borderRadius: '4px 4px 0 0',
                          transition: 'height 0.3s ease',
                          cursor: 'pointer',
                        }}
                        title={`${day.date} — ${day.count} detections`}
                      />
                    </div>
                    <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 500, textAlign: 'center' }}>
                      {day.date}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Top Detected Persons & System Health */}
          <div className="grid-2">
            {/* Top Detected Persons */}
            <div className="card">
              <div className="card-header"><span className="card-title">Top Detected Persons</span></div>
              {topPersons.length === 0 ? (
                <div className="empty-state" style={{ padding: 40 }}>No person detections yet</div>
              ) : (
                <div className="flex-col" style={{ gap: 14 }}>
                  {topPersons.map((person, i) => (
                    <div key={person.id}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                          <span style={{ 
                            fontSize: 16, 
                            fontWeight: 700, 
                            color: 'var(--text-muted)', 
                            minWidth: 24 
                          }}>
                            #{i + 1}
                          </span>
                          <span style={{ fontSize: 14, color: 'var(--text-primary)', fontWeight: 500 }}>
                            {person.name}
                          </span>
                        </div>
                        <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--accent)' }}>
                          {person.count} detections
                        </span>
                      </div>
                      <Bar value={person.count} max={topPersons[0]?.count || 1} color="var(--accent)" />
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* System Health */}
            <div className="card">
              <div className="card-header"><span className="card-title">System Health</span></div>
              <div className="flex-col" style={{ gap: 20 }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Average Confidence</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--green)' }}>
                      {(avgConfidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Bar value={avgConfidence * 100} max={100} color="var(--green)" />
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Recognition Rate</span>
                    <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--blue)' }}>
                      {total > 0 ? ((known / total) * 100).toFixed(1) : 0}%
                    </span>
                  </div>
                  <Bar value={known} max={total} color="var(--blue)" />
                </div>
                <div style={{ 
                  display: 'grid', 
                  gridTemplateColumns: '1fr 1fr', 
                  gap: 16, 
                  marginTop: 10,
                  padding: 16,
                  background: 'var(--bg-base)',
                  borderRadius: 8
                }}>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Total Alerts</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--accent)' }}>{alerts.length}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Active Cameras</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--green)' }}>1</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
