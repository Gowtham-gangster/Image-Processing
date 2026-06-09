import { useState, useEffect } from 'react'

import { API } from '../config'
const EMPTY_FORM = { person_id: '', name: '', gender: '', age: '', phone: '', address: '' }

export default function PersonsManager() {
  const [persons, setPersons] = useState([])
  const [loading, setLoading] = useState(true)
  const [form, setForm] = useState(EMPTY_FORM)
  const [editingId, setEditingId] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [toast, setToast] = useState(null)
  const [expandedId, setExpandedId] = useState(null)

  const load = async () => {
    setLoading(true)
    try {
      const r = await fetch(`${API}/persons`)
      if (!r.ok) {
        const err = await r.json().catch(() => ({}))
        showToast(err.detail || `API error ${r.status} — check Railway backend`, 'error')
        setPersons([])
        return
      }
      const d = await r.json()
      setPersons(d.persons || [])
    } catch (e) {
      showToast(`Cannot reach API at ${API} — is Railway running?`, 'error')
      setPersons([])
    }
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3000)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.person_id || !form.name) return
    setSubmitting(true)
    
    try {
      const url = editingId ? `${API}/persons/${editingId}` : `${API}/persons`
      const method = editingId ? 'PUT' : 'POST'
      
      const r = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      
      if (r.ok) {
        showToast(editingId ? `Updated ${form.name} ✓` : `Added ${form.name} ✓`)
        setForm(EMPTY_FORM)
        setEditingId(null)
        load()
      } else {
        showToast(editingId ? 'Failed to update person' : 'Failed to add person', 'error')
      }
    } catch (_) {
      showToast('Network error', 'error')
    }
    setSubmitting(false)
  }

  const handleEdit = (person) => {
    setForm({
      person_id: person.id || person.person_id,
      name: person.name,
      gender: person.gender || '',
      age: person.age || '',
      phone: person.phone || '',
      address: person.address || '',
    })
    setEditingId(person.id || person.person_id)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleCancel = () => {
    setForm(EMPTY_FORM)
    setEditingId(null)
  }

  const toggleExpand = (id) => {
    setExpandedId(expandedId === id ? null : id)
  }

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Persons Manager</h1>
        <p className="page-subtitle">Manage enrolled identities in the recognition database</p>
      </div>

      <div className="grid-2" style={{ alignItems: 'start', gap: 24 }}>
        {/* Enroll/Edit form */}
        <div className="card sticky-form-card" style={{ position: 'sticky', top: 20 }}>
          <div className="card-header" style={{ background: editingId ? 'rgba(59, 130, 246, 0.1)' : 'transparent', borderRadius: '8px 8px 0 0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 24 }}>{editingId ? '✏️' : '➕'}</span>
              <span className="card-title">{editingId ? 'Edit Person' : 'Enroll New Person'}</span>
            </div>
            {editingId && (
              <button className="btn btn-ghost btn-sm" onClick={handleCancel}>
                ✕ Cancel
              </button>
            )}
          </div>
          <form onSubmit={handleSubmit} className="flex-col">
            <div className="form-grid">
              <div className="form-group" style={{ gridColumn: '1 / -1' }}>
                <label className="form-label">Person ID *</label>
                <input 
                  className="form-input" 
                  placeholder="person_001" 
                  value={form.person_id}
                  onChange={e => setForm(p => ({ ...p, person_id: e.target.value }))} 
                  required 
                  disabled={editingId !== null}
                  style={{ opacity: editingId ? 0.6 : 1 }}
                />
                {editingId && (
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                    Person ID cannot be changed
                  </span>
                )}
              </div>
              <div className="form-group" style={{ gridColumn: '1 / -1' }}>
                <label className="form-label">Full Name *</label>
                <input 
                  className="form-input" 
                  placeholder="John Doe" 
                  value={form.name}
                  onChange={e => setForm(p => ({ ...p, name: e.target.value }))} 
                  required 
                />
              </div>
              <div className="form-group">
                <label className="form-label">Gender</label>
                <select 
                  className="form-input" 
                  value={form.gender}
                  onChange={e => setForm(p => ({ ...p, gender: e.target.value }))}
                  style={{ cursor: 'pointer' }}
                >
                  <option value="">Select...</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Age</label>
                <input 
                  className="form-input" 
                  type="number" 
                  placeholder="28" 
                  value={form.age}
                  onChange={e => setForm(p => ({ ...p, age: e.target.value }))} 
                />
              </div>
              <div className="form-group" style={{ gridColumn: '1 / -1' }}>
                <label className="form-label">Phone</label>
                <input 
                  className="form-input" 
                  placeholder="+1 555 0100" 
                  value={form.phone}
                  onChange={e => setForm(p => ({ ...p, phone: e.target.value }))} 
                />
              </div>
              <div className="form-group" style={{ gridColumn: '1 / -1' }}>
                <label className="form-label">Address</label>
                <textarea 
                  className="form-input" 
                  placeholder="Block A, Floor 2" 
                  value={form.address}
                  onChange={e => setForm(p => ({ ...p, address: e.target.value }))}
                  rows={2}
                  style={{ resize: 'vertical', fontFamily: 'inherit' }}
                />
              </div>
            </div>
            <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
              <button 
                className="btn btn-primary" 
                type="submit" 
                disabled={submitting}
                style={{ flex: 1 }}
              >
                {submitting ? (editingId ? 'Updating…' : 'Enrolling…') : (editingId ? '💾 Update Person' : '➕ Enroll Person')}
              </button>
              {editingId && (
                <button 
                  className="btn btn-ghost" 
                  type="button" 
                  onClick={handleCancel}
                >
                  Cancel
                </button>
              )}
            </div>
          </form>
        </div>

        {/* Enrolled list */}
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 20 }}>👥</span>
              <span className="card-title">Enrolled Persons</span>
              <span style={{ 
                fontSize: 12, 
                padding: '2px 8px', 
                background: 'var(--accent)', 
                color: 'white', 
                borderRadius: 12,
                fontWeight: 600
              }}>
                {persons.length}
              </span>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={load}>↻ Refresh</button>
          </div>
          
          {loading ? (
            <div className="empty-state">Loading…</div>
          ) : persons.length === 0 ? (
            <div className="empty-state" style={{ padding: 60 }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>👤</div>
              <div style={{ fontSize: 16, color: 'var(--text-secondary)', marginBottom: 8 }}>No persons enrolled yet</div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>Add your first person using the form on the left</div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {persons.map(p => {
                const personId = p.id || p.person_id
                const isExpanded = expandedId === personId
                
                return (
                  <div 
                    key={personId}
                    style={{
                      background: 'var(--bg-elevated)',
                      border: '1px solid var(--border)',
                      borderRadius: 8,
                      padding: 16,
                      transition: 'all 0.2s ease',
                      cursor: 'pointer',
                    }}
                    onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--accent)'}
                    onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
                  >
                    <div 
                      style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                      onClick={() => toggleExpand(personId)}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1 }}>
                        <div style={{
                          width: 40,
                          height: 40,
                          borderRadius: 20,
                          background: 'var(--accent)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: 18,
                          fontWeight: 700,
                          color: 'white'
                        }}>
                          {p.name?.charAt(0)?.toUpperCase() || '?'}
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2 }}>
                            {p.name}
                          </div>
                          <div style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                            ID: {personId}
                          </div>
                        </div>
                        <div style={{ display: 'flex', gap: 16, marginRight: 16 }}>
                          {p.gender && p.gender !== 'N/A' && (
                            <div style={{ textAlign: 'center' }}>
                              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>Gender</div>
                              <div style={{ fontSize: 13, fontWeight: 500 }}>{p.gender}</div>
                            </div>
                          )}
                          {p.age && p.age !== 'N/A' && (
                            <div style={{ textAlign: 'center' }}>
                              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 2 }}>Age</div>
                              <div style={{ fontSize: 13, fontWeight: 500 }}>{p.age}</div>
                            </div>
                          )}
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <button
                          className="btn btn-ghost btn-sm"
                          onClick={(e) => {
                            e.stopPropagation()
                            handleEdit(p)
                          }}
                          style={{ padding: '6px 12px' }}
                        >
                          ✏️ Edit
                        </button>
                        <span style={{ fontSize: 18, color: 'var(--text-muted)', transition: 'transform 0.2s', transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}>
                          ▼
                        </span>
                      </div>
                    </div>
                    
                    {isExpanded && (
                      <div style={{ 
                        marginTop: 16, 
                        paddingTop: 16, 
                        borderTop: '1px solid var(--border)',
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr',
                        gap: 12
                      }}>
                        <div>
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>📞 Phone</div>
                          <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                            {p.phone && p.phone !== 'N/A' ? p.phone : '—'}
                          </div>
                        </div>
                        <div>
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>📍 Address</div>
                          <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                            {p.address && p.address !== 'N/A' ? p.address : '—'}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {toast && (
        <div className="toast-container">
          <div className={`toast ${toast.type}`}>{toast.msg}</div>
        </div>
      )}
    </div>
  )
}
