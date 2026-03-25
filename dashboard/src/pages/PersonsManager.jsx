import { useState, useEffect } from 'react'

import { API } from '../config'
const EMPTY_FORM = { person_id: '', name: '', gender: '', age: '', phone: '', address: '' }

export default function PersonsManager() {
  const [persons, setPersons]   = useState([])
  const [loading, setLoading]   = useState(true)
  const [form, setForm]         = useState(EMPTY_FORM)
  const [submitting, setSubmitting] = useState(false)
  const [toast, setToast]       = useState(null)

  const load = async () => {
    setLoading(true)
    try {
      const r = await fetch(`${API}/persons`)
      const d = await r.json()
      setPersons(d.persons || [])
    } catch (_) {}
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
      const r = await fetch(`${API}/persons`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (r.ok) {
        showToast(`Added ${form.name} ✓`)
        setForm(EMPTY_FORM)
        load()
      } else {
        showToast('Failed to add person', 'error')
      }
    } catch (_) {
      showToast('Network error', 'error')
    }
    setSubmitting(false)
  }

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Persons Manager</h1>
        <p className="page-subtitle">Manage enrolled identities in the recognition database</p>
      </div>

      <div className="grid-2" style={{ alignItems: 'start' }}>
        {/* Enroll form */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Enroll New Person</span>
          </div>
          <form onSubmit={handleSubmit} className="flex-col">
            <div className="form-grid">
              <div className="form-group">
                <label className="form-label">Person ID *</label>
                <input className="form-input" placeholder="person_001" value={form.person_id}
                  onChange={e => setForm(p => ({ ...p, person_id: e.target.value }))} required />
              </div>
              <div className="form-group">
                <label className="form-label">Full Name *</label>
                <input className="form-input" placeholder="John Doe" value={form.name}
                  onChange={e => setForm(p => ({ ...p, name: e.target.value }))} required />
              </div>
              <div className="form-group">
                <label className="form-label">Gender</label>
                <input className="form-input" placeholder="M / F / Other" value={form.gender}
                  onChange={e => setForm(p => ({ ...p, gender: e.target.value }))} />
              </div>
              <div className="form-group">
                <label className="form-label">Age</label>
                <input className="form-input" placeholder="28" value={form.age}
                  onChange={e => setForm(p => ({ ...p, age: e.target.value }))} />
              </div>
              <div className="form-group">
                <label className="form-label">Phone</label>
                <input className="form-input" placeholder="+1 555 0100" value={form.phone}
                  onChange={e => setForm(p => ({ ...p, phone: e.target.value }))} />
              </div>
              <div className="form-group">
                <label className="form-label">Address</label>
                <input className="form-input" placeholder="Block A, Floor 2" value={form.address}
                  onChange={e => setForm(p => ({ ...p, address: e.target.value }))} />
              </div>
            </div>
            <button className="btn btn-primary" type="submit" disabled={submitting} style={{ marginTop: 4 }}>
              {submitting ? 'Enrolling…' : '+ Enroll Person'}
            </button>
          </form>
        </div>

        {/* Enrolled list */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Enrolled Persons ({persons.length})</span>
            <button className="btn btn-ghost btn-sm" onClick={load}>↻</button>
          </div>
          {loading ? <div className="empty-state">Loading…</div> :
           persons.length === 0 ? <div className="empty-state">No persons enrolled yet.</div> : (
            <table className="data-table">
              <thead><tr><th>ID</th><th>Name</th><th>Gender</th><th>Age</th></tr></thead>
              <tbody>
                {persons.map(p => (
                  <tr key={p.person_id}>
                    <td style={{ fontFamily: 'monospace', fontSize: 12 }}>{p.person_id}</td>
                    <td style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{p.name}</td>
                    <td>{p.gender || '—'}</td>
                    <td>{p.age || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
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
