/** Normalize API base URL — Vite env vars are baked in at build time. */
function normalizeApiUrl(raw) {
  const fallback = 'http://localhost:8000';
  if (!raw || !String(raw).trim()) return fallback;

  const trimmed = String(raw).trim().replace(/\/$/, '');

  // Railway/Vercel URLs must include a protocol or fetch treats them as relative paths
  // e.g. "foo.up.railway.app" → Vercel 404 on /foo.up.railway.app/persons
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  return `https://${trimmed}`;
}

const rawApi = import.meta.env.VITE_API_URL;
export const API = normalizeApiUrl(rawApi);
export const API_CONFIGURED = Boolean(rawApi && String(rawApi).trim());
