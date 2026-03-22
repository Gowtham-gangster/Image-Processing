import { useState, useRef, useEffect } from 'react'

const API = 'http://localhost:8000'

export default function Sandbox() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [imageSrc, setImageSrc] = useState(null)
  
  const fileInputRef = useRef(null)
  const canvasRef = useRef(null)

  // Draw bounding boxes when results or image change
  useEffect(() => {
    if (!imageSrc || !canvasRef.current || !results) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const img = new Image()
    img.onload = () => {
      // Set canvas dimensions to match image natural size
      canvas.width = img.width
      canvas.height = img.height
      
      // Draw base image
      ctx.drawImage(img, 0, 0)

      // Draw bounding boxes
      results.forEach((det, i) => {
        const [x, y, w, h] = det.bbox
        const color = det.is_known ? '#10b981' : '#f43f5e' // emerald-500 or rose-500
        
        ctx.strokeStyle = color
        ctx.lineWidth = Math.max(3, Math.floor(canvas.width / 300))
        ctx.strokeRect(x, y, w, h)
        
        // Draw label background
        const label = `${det.name} (${(det.confidence * 100).toFixed(1)}%)`
        ctx.font = `600 ${Math.max(14, Math.floor(canvas.width / 60))}px Inter, sans-serif`
        const textWidth = ctx.measureText(label).width
        const textHeight = parseInt(ctx.font, 10)
        
        ctx.fillStyle = color
        ctx.fillRect(x, Math.max(0, y - textHeight - 8), textWidth + 16, textHeight + 8)
        
        // Draw label text
        ctx.fillStyle = '#ffffff'
        ctx.textBaseline = 'middle'
        ctx.fillText(label, x + 8, Math.max(0, y - textHeight - 8) + (textHeight + 8) / 2)
      })
    }
    img.src = imageSrc
  }, [imageSrc, results])

  const handleFileDrop = (e) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0])
    }
  }

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0])
    }
  }

  const processFile = async (file) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file.')
      return
    }

    setError(null)
    setResults(null)
    
    // Create local object URL for instant preview
    const objectUrl = URL.createObjectURL(file)
    setImageSrc(objectUrl)
    setLoading(true)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('camera_id', 'Dashboard-Sandbox')

    try {
      const res = await fetch(`${API}/recognize/image`, {
        method: 'POST',
        body: formData
      })
      
      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'API request failed')
      }
      
      const data = await res.json()
      setResults(data.results)
      
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-container">
      <header style={{ marginBottom: 32 }}>
        <h1 className="page-title">Testing Sandbox</h1>
        <p style={{ color: 'var(--text-muted)' }}>Upload an image to test the detection, recognition, and liveness models directly in the browser.</p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.5fr) minmax(0, 1fr)', gap: 24, alignItems: 'start' }}>
        
        {/* Left Column: Image Area */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          
          <div 
            style={{ 
              border: `2px dashed ${imageSrc ? 'transparent' : 'var(--border)'}`,
              borderRadius: 8,
              minHeight: 400,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexDirection: 'column',
              backgroundColor: imageSrc ? '#000' : 'rgba(255,255,255,0.02)',
              position: 'relative',
              overflow: 'hidden',
              cursor: imageSrc ? 'default' : 'pointer',
              transition: 'all 0.2s'
            }}
            onDragOver={e => e.preventDefault()}
            onDrop={handleFileDrop}
            onClick={() => !imageSrc && fileInputRef.current.click()}
          >
            {imageSrc ? (
              <canvas 
                ref={canvasRef} 
                style={{ maxWidth: '100%', maxHeight: '600px', display: 'block' }}
              />
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ opacity: 0.5, marginBottom: 16 }}>
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <div style={{ fontSize: 16, fontWeight: 500, color: 'var(--text)' }}>Click or drag to upload an image</div>
                <div style={{ fontSize: 13, marginTop: 4 }}>JPG, PNG, WEBP supported</div>
              </div>
            )}
            
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept="image/*"
              onChange={handleFileSelect}
            />
          </div>

          <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
              {loading ? 'Processing image with AI pipeline...' : 
               results ? `Detected ${results.length} person(s)` : 
               'Awaiting image'}
            </div>
            <button 
              className="btn btn-secondary" 
              onClick={() => fileInputRef.current.click()}
              disabled={loading}
              style={{ padding: '8px 16px' }}
            >
              {imageSrc ? 'Upload Different Image' : 'Select Image'}
            </button>
          </div>
        </div>

        {/* Right Column: Results Area */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          
          {error && (
            <div style={{ padding: 16, backgroundColor: 'rgba(244,63,94,0.1)', color: '#f43f5e', borderRadius: 8, border: '1px solid rgba(244,63,94,0.2)' }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {loading && !results && (
            <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
              <div style={{ width: 24, height: 24, border: '2px solid', borderTopColor: 'transparent', borderRadius: '50%', margin: '0 auto 16px', animation: 'spin 1s linear infinite' }} />
              Running detection, alignment, liveness, and embedding models...
            </div>
          )}

          {results && results.length === 0 && (
            <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
              No faces detected in this image.
            </div>
          )}

          {results && results.map((det, i) => (
            <div key={i} className="card" style={{ padding: 20 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
                <div>
                  <h3 style={{ fontSize: 18, fontWeight: 600, color: '#fff', margin: '0 0 4px' }}>{det.name}</h3>
                  <div style={{ color: 'var(--text-muted)', fontSize: 13, fontFamily: 'monospace' }}>
                    {det.person_id}
                  </div>
                </div>
                <div style={{ fontSize: 18, fontWeight: 600, color: det.is_known ? '#10b981' : '#f43f5e' }}>
                  {(det.confidence * 100).toFixed(1)}%
                </div>
              </div>

              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
                {/* Identity Badge */}
                <span className={`badge ${det.is_known ? 'badge-success' : 'badge-danger'}`} style={{ fontSize: 13, padding: '4px 10px' }}>
                  {det.is_known ? 'Known Identity' : 'Unknown Person'}
                </span>
                
                {/* Liveness Badge */}
                <span className={`badge ${det.is_live ? 'badge-success' : 'badge-danger'}`} style={{ fontSize: 13, padding: '4px 10px' }}>
                  {det.is_live ? 'Live Person' : 'Spoof Detected'}
                </span>
                
                {/* Mask Badge */}
                <span className="badge" style={{ backgroundColor: det.is_masked ? 'rgba(56,189,248,0.1)' : 'rgba(255,255,255,0.05)', color: det.is_masked ? '#38bdf8' : 'var(--text-muted)', fontSize: 13, padding: '4px 10px' }}>
                  {det.is_masked ? 'Masked' : 'No Mask'}
                </span>
              </div>

              {/* Attributes Section */}
              {det.attributes && Object.keys(det.attributes).length > 0 && det.person_id !== 'SPOOF DETECTED' && (
                <div style={{ backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: 6, padding: 12 }}>
                  <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, color: 'var(--text-muted)', marginBottom: 8, fontWeight: 600 }}>Attributes</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 16px' }}>
                    {Object.entries(det.attributes).map(([key, val]) => {
                      if (key === 'name' || key === 'embeddings_count') return null; // Skip redundant info
                      const displayVal = val === 'N/A' || !val ? '-' : val;
                      return (
                        <div key={key}>
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'capitalize' }}>{key}</div>
                          <div style={{ fontSize: 13, color: 'var(--text)' }}>{displayVal}</div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
              
              {/* Spoof Section */}
              {det.person_id === 'SPOOF DETECTED' && (
                <div style={{ backgroundColor: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.2)', borderRadius: 6, padding: 12, marginTop: 12 }}>
                  <div style={{ fontSize: 13, color: '#f43f5e', fontWeight: 500 }}>⚠️ Presentation Attack Failed</div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>The liveness detector blocked this recognition attempt to prevent spoofing.</div>
                </div>
              )}
            </div>
          ))}

          {!results && !loading && (
            <div style={{ 
              padding: 40, 
              border: '1px dashed var(--border)', 
              borderRadius: 8, 
              textAlign: 'center', 
              color: 'var(--text-muted)',
              fontSize: 14
            }}>
              Upload an image to see results here.
            </div>
          )}

        </div>
      </div>
    </div>
  )
}
