import React from 'react';

export default function ResultCard({ prediction }) {
  if (!prediction) return null;

  const { person, confidence, mask, age, gender, phone, address } = prediction;
  const isUnknown = person === 'Unknown' || person === 'Unknown Person';

  return (
    <div className="card">
      <div className="card-header" style={{ marginBottom: 20 }}>
        <div className="card-title">Prediction Results</div>
        {mask ? (
          <span className="badge masked">Mask Detected</span>
        ) : (
          <span className="badge">No Mask</span>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 24 }}>
        <div style={{ 
          width: 50, height: 50, borderRadius: '50%', 
          background: isUnknown ? 'rgba(248,81,73,0.1)' : 'rgba(63, 185, 80, 0.1)', 
          color: isUnknown ? 'var(--red)' : 'var(--green)',
          display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
            <circle cx="12" cy="7" r="4"></circle>
          </svg>
        </div>
        <div>
          <div style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>
            {person}
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            Confidence: {(confidence * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: 16 }}>
        <div className="stat-card">
          <div className="stat-label">Age</div>
          <div className="stat-value" style={{ fontSize: 18 }}>{age || 'N/A'}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Gender</div>
          <div className="stat-value" style={{ fontSize: 18 }}>{gender || 'N/A'}</div>
        </div>
      </div>
      
      <div className="grid-2">
        <div className="stat-card">
          <div className="stat-label">Phone</div>
          <div className="stat-value" style={{ fontSize: 14 }}>{phone || 'N/A'}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Address</div>
          <div className="stat-value" style={{ fontSize: 14 }}>{address || 'N/A'}</div>
        </div>
      </div>
    </div>
  );
}
