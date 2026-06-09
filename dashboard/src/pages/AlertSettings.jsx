import { useState, useEffect } from 'react'

import { API } from '../config'

const EMPTY_FORM = {
  slack_webhook_url: '',
  webhook_url: '',
  email_host: '',
  email_port: '587',
  email_sender: '',
  email_password: '',
  email_recipients: '',
}

export default function AlertSettings() {
  const [form, setForm]   = useState(EMPTY_FORM)
  const [saving, setSaving]   = useState(false)
  const [testing, setTesting] = useState(false)
  const [toast, setToast] = useState(null)

  useEffect(() => {
    fetch(`${API}/alerts/config`)
      .then(r => r.json())
      .then(cfg => {
        setForm({
          slack_webhook_url: cfg.slack_webhook_url || '',
          webhook_url:       cfg.webhook_url || '',
          email_host:        cfg.email?.smtp_host || '',
          email_port:        String(cfg.email?.smtp_port || 587),
          email_sender:      cfg.email?.sender || '',
          email_password:    '',   // never pre-fill password
          email_recipients:  (cfg.email?.recipients || []).join(', '),
        })
      })
      .catch(() => {})
  }, [])

  const showToast = (msg, type = 'success') => {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 3500)
  }

  const handleSave = async (e) => {
    e.preventDefault()
    setSaving(true)
    const body = {
      slack_webhook_url: form.slack_webhook_url || null,
      webhook_url:       form.webhook_url || null,
    }
    if (form.email_host) {
      body.email = {
        smtp_host:   form.email_host,
        smtp_port:   Number(form.email_port),
        sender:      form.email_sender,
        recipients:  form.email_recipients.split(',').map(s => s.trim()).filter(Boolean),
      }
      if (form.email_password.trim()) {
        body.email.password = form.email_password.trim()
      }
    }
    try {
      const r = await fetch(`${API}/alerts/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (r.ok) { showToast('Alert config saved ✓') }
      else       { showToast('Save failed', 'error') }
    } catch (_) { showToast('Network error', 'error') }
    setSaving(false)
  }

  const handleTest = async () => {
    setTesting(true)
    try {
      const r = await fetch(`${API}/alerts/test`, { method: 'POST' })
      const data = await r.json().catch(() => ({}))
      if (r.ok && data.status === 'ok') {
        showToast(data.message || 'Test alert sent ✓')
      } else {
        showToast(data.message || data.errors?.email || 'Test failed — save config first', 'error')
      }
    } catch (_) { showToast('Network error', 'error') }
    setTesting(false)
  }

  const f = (key) => ({ value: form[key], onChange: e => setForm(p => ({ ...p, [key]: e.target.value })) })

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Alert Settings</h1>
        <p className="page-subtitle">Configure Slack, Email, and webhook notifications</p>
      </div>

      <form onSubmit={handleSave} className="flex-col">
        {/* Slack */}
        <div className="channel-card">
          <div className="channel-header">
            <span className="channel-icon">💬</span>
            Slack
            <div className={`status-dot ${form.slack_webhook_url ? 'on' : 'off'}`} />
          </div>
          <div className="form-group">
            <label className="form-label">Incoming Webhook URL</label>
            <input className="form-input" placeholder="https://hooks.slack.com/services/..." {...f('slack_webhook_url')} />
          </div>
        </div>

        {/* Webhook */}
        <div className="channel-card">
          <div className="channel-header">
            <span className="channel-icon">🔗</span>
            HTTP Webhook
            <div className={`status-dot ${form.webhook_url ? 'on' : 'off'}`} />
          </div>
          <div className="form-group">
            <label className="form-label">Endpoint URL</label>
            <input className="form-input" placeholder="https://my-siem.example.com/ingest" {...f('webhook_url')} />
          </div>
        </div>

        {/* Email */}
        <div className="channel-card">
          <div className="channel-header">
            <span className="channel-icon">✉️</span>
            Email (SMTP)
            <div className={`status-dot ${form.email_host ? 'on' : 'off'}`} />
          </div>
          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">SMTP Host</label>
              <input className="form-input" placeholder="smtp.gmail.com" {...f('email_host')} />
            </div>
            <div className="form-group">
              <label className="form-label">Port</label>
              <input className="form-input" placeholder="587" {...f('email_port')} />
            </div>
            <div className="form-group">
              <label className="form-label">Sender Email</label>
              <input className="form-input" placeholder="system@example.com" {...f('email_sender')} />
            </div>
            <div className="form-group">
              <label className="form-label">App Password</label>
              <input className="form-input" type="password" placeholder="Gmail app password (16 chars)" {...f('email_password')} />
              <p className="form-hint" style={{ marginTop: 6, fontSize: 12, color: '#888' }}>
                Gmail: use an App Password, not your login password. Leave blank when saving other fields to keep the existing password.
              </p>
            </div>
            <div className="form-group" style={{ gridColumn: '1 / -1' }}>
              <label className="form-label">Recipient Emails (comma-separated)</label>
              <input className="form-input" placeholder="security@example.com, soc@example.com" {...f('email_recipients')} />
            </div>
          </div>
        </div>

        <div className="flex-row">
          <button className="btn btn-primary" type="submit" disabled={saving}>
            {saving ? 'Saving…' : '💾 Save Config'}
          </button>
          <button className="btn btn-ghost" type="button" onClick={handleTest} disabled={testing}>
            {testing ? 'Sending…' : '🧪 Send Test Alert'}
          </button>
        </div>
      </form>

      {toast && (
        <div className="toast-container">
          <div className={`toast ${toast.type}`}>{toast.msg}</div>
        </div>
      )}
    </div>
  )
}
