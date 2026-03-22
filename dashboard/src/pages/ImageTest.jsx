import React, { useState } from 'react';
import ImageUpload from '../components/ImageUpload';
import ResultCard from '../components/ResultCard';

export default function ImageTest() {
  const [file, setFile] = useState(null);
  const [previewSrc, setPreviewSrc] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (selectedFile, previewUrl) => {
    setFile(selectedFile);
    setPreviewSrc(previewUrl);
    setPrediction(null);
    setError(null);
  };

  const handleClear = () => {
    setFile(null);
    setPreviewSrc(null);
    setPrediction(null);
    setError(null);
  };

  const handleRunDetection = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server Error: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <div className="page-header">
        <h1 className="page-title">Image Upload Testing</h1>
        <p className="page-subtitle">Test the hybrid detection pipeline with custom images.</p>
      </div>

      {!file ? (
        <ImageUpload onFileSelect={handleFileSelect} />
      ) : (
        <div className="grid-2" style={{ alignItems: 'start' }}>
          {/* Left Column: Image Preview */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="card-header" style={{ marginBottom: 0 }}>
              <div className="card-title">Input Image</div>
              <button className="btn btn-ghost btn-sm" onClick={handleClear}>Clear</button>
            </div>
            
            <div style={{ 
              width: '100%', 
              background: 'var(--bg-base)', 
              borderRadius: 'var(--radius-sm)',
              overflow: 'hidden',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              border: '1px solid var(--border)',
              padding: 8
            }}>
              <img 
                src={previewSrc} 
                alt="Preview" 
                style={{ width: '100%', maxHeight: 350, objectFit: 'contain', borderRadius: 'var(--radius-sm)' }} 
              />
            </div>

            <button 
              className="btn btn-primary" 
              style={{ width: '100%', justifyContent: 'center', padding: '12px' }}
              onClick={handleRunDetection}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <div className="pulse-dot" style={{ width: 10, height: 10, background: 'currentColor', boxShadow: 'none', animation: 'none', opacity: 0.5 }}></div>
                  Processing...
                </>
              ) : (
                'Run Detection'
              )}
            </button>
          </div>

          {/* Right Column: Results */}
          <div>
            {isLoading && (
              <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '80px 20px', gap: 16 }}>
                <div className="pulse-dot" style={{ width: 16, height: 16 }}></div>
                <div style={{ color: 'var(--text-secondary)' }}>Analyzing image using hybrid AI...</div>
              </div>
            )}

            {!isLoading && error && (
              <div className="card" style={{ borderLeft: '4px solid var(--red)' }}>
                <div style={{ color: 'var(--red)', fontWeight: 600, marginBottom: 8 }}>Detection Failed</div>
                <div style={{ color: 'var(--text-secondary)', fontSize: 13 }}>{error}</div>
              </div>
            )}

            {!isLoading && prediction && (
              <>
                <ResultCard prediction={prediction} />
              </>
            )}

            {!isLoading && !error && !prediction && (
              <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '80px 20px' }}>
                <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>Click 'Run Detection' to analyze</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
