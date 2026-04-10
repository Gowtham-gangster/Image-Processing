import { useState, useEffect } from 'react'
import { API } from '../config'

export default function CameraSettings() {
  const [cameras, setCameras] = useState([])
  const [loading, setLoading] = useState(true)
  const [editingId, setEditingId] = useState(null)
  const [formData, setFormData] = useState({
    id: '',
    name: '',
    type: 'local',
    source: '',
    enabled: true,
    location: '',
    description: '',
    device_id: '',
    mac_address: '',
    serial_number: ''
  })
  const [showAddForm, setShowAddForm] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)

  useEffect(() => {
    loadCameras()
  }, [])

  const loadCameras = async () => {
    try {
      const res = await fetch(`${API}/cameras`)
      const data = await res.json()
      setCameras(data.cameras || [])
    } catch (err) {
      setError('Failed to load cameras')
    } finally {
      setLoading(false)
    }
  }

  const handleEdit = (camera) => {
    setEditingId(camera.id)
    setFormData(camera)
    setShowAddForm(false)
  }

  const handleAdd = () => {
    setEditingId(null)
    setFormData({
      id: `CAM${String(cameras.length + 1).padStart(3, '0')}`,
      name: '',
      type: 'local',
      source: '0',
      enabled: true,
      location: '',
      description: '',
      device_id: '',
      mac_address: '',
      serial_number: ''
    })
    setShowAddForm(true)
  }

  const handleSave = async () => {
    try {
      setError(null)
      setSuccess(null)

      if (editingId) {
        // Update existing camera
        const res = await fetch(`${API}/cameras/${editingId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData)
        })
        if (!res.ok) throw new Error('Failed to update camera')
        setSuccess('Camera updated successfully')
      } else {
        // Add new camera
        const res = await fetch(`${API}/cameras`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData)
        })
        if (!res.ok) throw new Error('Failed to add camera')
        setSuccess('Camera added successfully')
      }

      await loadCameras()
      setEditingId(null)
      setShowAddForm(false)
    } catch (err) {
      setError(err.message)
    }
  }

  const handleDelete = async (cameraId) => {
    if (!confirm(`Are you sure you want to delete camera ${cameraId}?`)) return

    try {
      setError(null)
      const res = await fetch(`${API}/cameras/${cameraId}`, {
        method: 'DELETE'
      })
      if (!res.ok) throw new Error('Failed to delete camera')
      setSuccess('Camera deleted successfully')
      await loadCameras()
    } catch (err) {
      setError(err.message)
    }
  }

  const handleToggleEnabled = async (camera) => {
    try {
      setError(null)
      const updatedCamera = { ...camera, enabled: !camera.enabled }
      const res = await fetch(`${API}/cameras/${camera.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedCamera)
      })
      if (!res.ok) throw new Error('Failed to update camera')
      setSuccess(`Camera ${updatedCamera.enabled ? 'enabled' : 'disabled'} successfully`)
      await loadCameras()
    } catch (err) {
      setError(err.message)
    }
  }

  const handleCancel = () => {
    setEditingId(null)
    setShowAddForm(false)
  }

  const isEditing = editingId !== null || showAddForm

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Camera Settings</h1>
          <p className="page-subtitle">Manage camera sources and CCTV connections</p>
        </div>
        {!isEditing && (
          <button className="btn btn-primary" onClick={handleAdd}>
            ➕ Add Camera
          </button>
        )}
      </div>

      {error && (
        <div style={{
          padding: '12px 16px',
          background: 'var(--red)',
          color: 'white',
          borderRadius: 8,
          marginBottom: 16
        }}>
          ⚠️ {error}
        </div>
      )}

      {success && (
        <div style={{
          padding: '12px 16px',
          background: 'var(--green)',
          color: 'white',
          borderRadius: 8,
          marginBottom: 16
        }}>
          ✓ {success}
        </div>
      )}

      {/* Edit/Add Form */}
      {isEditing && (
        <div className="card" style={{ marginBottom: 24, position: 'sticky', top: 20, zIndex: 10 }}>
          <div className="card-header">
            <span className="card-title">
              {editingId ? `Edit Camera: ${editingId}` : 'Add New Camera'}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <label style={{ display: 'block', fontSize: 14, fontWeight: 700, marginBottom: 8, color: 'var(--text-primary)' }}>
                Camera ID *
              </label>
              <input
                type="text"
                value={formData.id}
                onChange={(e) => setFormData({ ...formData, id: e.target.value })}
                disabled={editingId !== null}
                placeholder="CAM001"
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  borderRadius: 6,
                  border: '1px solid var(--border)',
                  background: editingId ? 'var(--bg-base)' : 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '15px',
                  fontWeight: 500
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: 14, fontWeight: 700, marginBottom: 8, color: 'var(--text-primary)' }}>
                Camera Name *
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="Main Entrance Camera"
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  borderRadius: 6,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '15px',
                  fontWeight: 500
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: 14, fontWeight: 700, marginBottom: 8, color: 'var(--text-primary)' }}>
                Type *
              </label>
              <select
                value={formData.type}
                onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  borderRadius: 6,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-secondary)',
                  color: '#ffffff',
                  fontSize: '15px',
                  fontWeight: 600,
                  cursor: 'pointer'
                }}
              >
                <option value="local" style={{ fontSize: '15px', fontWeight: 600, padding: '8px', background: '#ffffff', color: '#000000' }}>
                  Local Camera (Webcam)
                </option>
                <option value="rtsp" style={{ fontSize: '15px', fontWeight: 600, padding: '8px', background: '#ffffff', color: '#000000' }}>
                  RTSP Stream (CCTV)
                </option>
                <option value="http" style={{ fontSize: '15px', fontWeight: 600, padding: '8px', background: '#ffffff', color: '#000000' }}>
                  HTTP Stream
                </option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: 14, fontWeight: 700, marginBottom: 8, color: 'var(--text-primary)' }}>
                Source *
              </label>
              <input
                type="text"
                value={formData.source}
                onChange={(e) => setFormData({ ...formData, source: e.target.value })}
                placeholder={formData.type === 'local' ? '0' : 'rtsp://admin:password@192.168.1.100:554/stream'}
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  borderRadius: 6,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '15px',
                  fontWeight: 500,
                  fontFamily: 'monospace'
                }}
              />
              <small style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginTop: 6 }}>
                {formData.type === 'local' ? 'Camera index (0, 1, 2...)' : 'RTSP/HTTP URL'}
              </small>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: 14, fontWeight: 700, marginBottom: 8, color: 'var(--text-primary)' }}>
                Location
              </label>
              <input
                type="text"
                value={formData.location}
                onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                placeholder="Main Entrance"
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  borderRadius: 6,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '15px',
                  fontWeight: 500
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: 14, fontWeight: 700, marginBottom: 8, color: 'var(--text-primary)' }}>
                Description
              </label>
              <input
                type="text"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="Camera description"
                style={{
                  width: '100%',
                  padding: '10px 14px',
                  borderRadius: 6,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '15px',
                  fontWeight: 500
                }}
              />
            </div>

            {formData.type !== 'local' && (
              <>
                <div>
                  <label style={{ display: 'block', fontSize: 14, fontWeight: 700, marginBottom: 8, color: 'var(--text-primary)' }}>
                    Device ID
                  </label>
                  <input
                    type="text"
                    value={formData.device_id}
                    onChange={(e) => setFormData({ ...formData, device_id: e.target.value })}
                    placeholder="Device unique identifier"
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      borderRadius: 6,
                      border: '1px solid var(--border)',
                      background: 'var(--bg-secondary)',
                      color: 'var(--text-primary)',
                      fontSize: '15px',
                      fontWeight: 500
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 14, fontWeight: 700, marginBottom: 8, color: 'var(--text-primary)' }}>
                    MAC Address
                  </label>
                  <input
                    type="text"
                    value={formData.mac_address}
                    onChange={(e) => setFormData({ ...formData, mac_address: e.target.value })}
                    placeholder="00:11:22:33:44:55"
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      borderRadius: 6,
                      border: '1px solid var(--border)',
                      background: 'var(--bg-secondary)',
                      color: 'var(--text-primary)',
                      fontSize: '15px',
                      fontWeight: 500,
                      fontFamily: 'monospace'
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 14, fontWeight: 700, marginBottom: 8, color: 'var(--text-primary)' }}>
                    Serial Number
                  </label>
                  <input
                    type="text"
                    value={formData.serial_number}
                    onChange={(e) => setFormData({ ...formData, serial_number: e.target.value })}
                    placeholder="Serial number"
                    style={{
                      width: '100%',
                      padding: '10px 14px',
                      borderRadius: 6,
                      border: '1px solid var(--border)',
                      background: 'var(--bg-secondary)',
                      color: 'var(--text-primary)',
                      fontSize: '15px',
                      fontWeight: 500
                    }}
                  />
                </div>
              </>
            )}

            <div>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={formData.enabled}
                  onChange={(e) => setFormData({ ...formData, enabled: e.target.checked })}
                  style={{ width: 18, height: 18 }}
                />
                <span style={{ fontSize: 13, fontWeight: 600 }}>Camera Enabled</span>
              </label>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 12, marginTop: 16, justifyContent: 'flex-end' }}>
            <button className="btn btn-ghost" onClick={handleCancel}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={handleSave}>
              {editingId ? 'Update Camera' : 'Add Camera'}
            </button>
          </div>
        </div>
      )}

      {/* Camera List */}
      <div className="grid-1" style={{ gap: 16 }}>
        {loading ? (
          <div className="empty-state">Loading cameras...</div>
        ) : cameras.length === 0 ? (
          <div className="empty-state">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ width: 48, height: 48, margin: '0 auto 1rem' }}>
              <rect x="2" y="4" width="20" height="16" rx="2"/>
              <circle cx="12" cy="12" r="3"/>
            </svg>
            <p>No cameras configured</p>
            <button className="btn btn-primary" onClick={handleAdd} style={{ marginTop: 16 }}>
              Add Your First Camera
            </button>
          </div>
        ) : (
          cameras.map(camera => (
            <div 
              key={camera.id}
              className="card"
              style={{
                opacity: editingId === camera.id ? 0.5 : 1,
                transition: 'opacity 0.2s'
              }}
            >
              <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
                {/* Camera Icon */}
                <div style={{
                  width: 64,
                  height: 64,
                  borderRadius: 12,
                  background: camera.enabled ? 'var(--green)20' : 'var(--red)20',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 28,
                  flexShrink: 0
                }}>
                  📹
                </div>

                {/* Camera Details */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                    <h3 style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                      {camera.name}
                    </h3>
                    <span style={{
                      fontSize: 11,
                      padding: '4px 10px',
                      borderRadius: 6,
                      background: camera.enabled ? 'var(--green)20' : 'var(--red)20',
                      color: camera.enabled ? 'var(--green)' : 'var(--red)',
                      fontWeight: 600
                    }}>
                      {camera.enabled ? 'ENABLED' : 'DISABLED'}
                    </span>
                    <span style={{
                      fontSize: 11,
                      padding: '4px 10px',
                      borderRadius: 6,
                      background: 'var(--bg-base)',
                      color: 'var(--text-secondary)',
                      fontWeight: 600
                    }}>
                      {camera.id}
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '8px 16px', fontSize: 13, color: 'var(--text-secondary)' }}>
                    <span style={{ fontWeight: 600 }}>Type:</span>
                    <span>{camera.type === 'local' ? 'Local Camera' : camera.type.toUpperCase() + ' Stream'}</span>
                    
                    <span style={{ fontWeight: 600 }}>Source:</span>
                    <span style={{ fontFamily: 'monospace', fontSize: 12 }}>{camera.source}</span>
                    
                    {camera.location && (
                      <>
                        <span style={{ fontWeight: 600 }}>Location:</span>
                        <span>{camera.location}</span>
                      </>
                    )}
                    
                    {camera.description && (
                      <>
                        <span style={{ fontWeight: 600 }}>Description:</span>
                        <span>{camera.description}</span>
                      </>
                    )}

                    {camera.type !== 'local' && (camera.device_id || camera.mac_address || camera.serial_number) && (
                      <>
                        {camera.device_id && (
                          <>
                            <span style={{ fontWeight: 600 }}>Device ID:</span>
                            <span style={{ fontFamily: 'monospace', fontSize: 12 }}>{camera.device_id}</span>
                          </>
                        )}
                        {camera.mac_address && (
                          <>
                            <span style={{ fontWeight: 600 }}>MAC Address:</span>
                            <span style={{ fontFamily: 'monospace', fontSize: 12 }}>{camera.mac_address}</span>
                          </>
                        )}
                        {camera.serial_number && (
                          <>
                            <span style={{ fontWeight: 600 }}>Serial Number:</span>
                            <span style={{ fontFamily: 'monospace', fontSize: 12 }}>{camera.serial_number}</span>
                          </>
                        )}
                      </>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
                  <button 
                    className={`btn ${camera.enabled ? 'btn-ghost' : 'btn-primary'} btn-sm`}
                    onClick={() => handleToggleEnabled(camera)}
                    disabled={isEditing}
                    title={camera.enabled ? 'Disable camera' : 'Enable camera'}
                  >
                    {camera.enabled ? '⏸️ Disable' : '▶️ Enable'}
                  </button>
                  <button 
                    className="btn btn-ghost btn-sm"
                    onClick={() => handleEdit(camera)}
                    disabled={isEditing}
                  >
                    ✏️ Edit
                  </button>
                  <button 
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDelete(camera.id)}
                    disabled={isEditing}
                  >
                    🗑️ Delete
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
