import React, { useState, useRef } from 'react';

export default function ImageUpload({ onFileSelect }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const processFile = (file) => {
    if (!file) return;
    if (file.type !== 'image/jpeg' && file.type !== 'image/png') {
      alert('Only JPG and PNG files are allowed.');
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    onFileSelect(file, previewUrl);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  return (
    <div 
      className={`card ${isDragging ? 'drag-active' : ''}`}
      style={{ 
        borderStyle: 'dashed', 
        borderWidth: 2, 
        borderColor: isDragging ? 'var(--accent)' : 'var(--border)', 
        backgroundColor: isDragging ? 'var(--accent-glow)' : 'var(--bg-elevated)',
        textAlign: 'center',
        padding: '60px 20px',
        cursor: 'pointer',
        transition: 'all 0.2s ease'
      }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current.click()}
    >
      <input 
        type="file" 
        ref={fileInputRef} 
        style={{ display: 'none' }} 
        accept="image/jpeg, image/png" 
        onChange={handleFileChange} 
      />
      <div style={{ marginBottom: 12 }}>
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--text-secondary)" strokeWidth="1.5">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
      </div>
      <h3 style={{ fontSize: 16, color: 'var(--text-primary)', marginBottom: 6 }}>Drag and drop an image here</h3>
      <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>or click to browse from your computer (JPG, PNG)</p>
    </div>
  );
}
