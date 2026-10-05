import { useEffect, useRef, useState } from 'react';

// Production API base URL is provided at build time via VITE_API_BASE_URL
// (set it on Render). The localhost fallback applies only to local
// development when the variable is not configured.
const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const HISTORY_PAGE_SIZE = 10;

function formatClass(record) {
  return `${record.predicted_class.toUpperCase()} — ${record.predicted_class_name}`;
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function formatDateTime(iso) {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime())
    ? String(iso)
    : parsed.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

function apiEndpoint(apiBase, path) {
  return new URL(path, apiBase.trim()).toString();
}

async function readJsonOrEmpty(response) {
  return response.json().catch(() => ({}));
}

// Render one class-probability bar list from real API values; used by the
// live result view and the stored-history detail view alike.
function ProbabilityList({ probabilities }) {
  const entries = Object.entries(probabilities ?? {}).sort(([, a], [, b]) => b - a);
  return (
    <ul className="probability-list">
      {entries.map(([label, value]) => (
        <li key={label}>
          <span>{label.toUpperCase()}</span>
          <span className="prob-bar" style={{ '--pct': formatPercent(value) }} />
          <span>{formatPercent(value)}</span>
        </li>
      ))}
    </ul>
  );
}

export default function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [submitting, setSubmitting] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const [historyStatus, setHistoryStatus] = useState('');
  const [historyError, setHistoryError] = useState(null);
  const [historyItems, setHistoryItems] = useState(null); // null = not loaded yet
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyOffset, setHistoryOffset] = useState(0);
  const [clearing, setClearing] = useState(false);
  const [detail, setDetail] = useState(null);

  const fileInputRef = useRef(null);
  const resultSectionRef = useRef(null);
  const detailSectionRef = useRef(null);

  // Load stored history so previous analyses survive page refreshes.
  useEffect(() => {
    loadHistory(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadHistory(offset = 0) {
    setHistoryStatus('Loading prediction history…');
    setHistoryError(null);
    setDetail(null);
    try {
      const url = new URL('/predictions', apiBase.trim());
      url.searchParams.set('limit', String(HISTORY_PAGE_SIZE));
      url.searchParams.set('offset', String(offset));
      const response = await fetch(url);
      const payload = await readJsonOrEmpty(response);
      if (!response.ok) {
        throw new Error(payload.detail || `The API returned status ${response.status}.`);
      }
      setHistoryItems(payload.items ?? []);
      setHistoryTotal(payload.total ?? (payload.items ?? []).length);
      setHistoryOffset(offset);
    } catch (err) {
      setHistoryItems(null);
      if (err instanceof TypeError) {
        setHistoryError(
          `Could not reach the API at ${apiBase.trim()}. Prediction history ` +
            'will appear here once the backend is running.'
        );
      } else {
        setHistoryError(err instanceof Error ? err.message : 'Unable to load prediction history.');
      }
    } finally {
      setHistoryStatus('');
    }
  }

  async function loadPredictionDetail(id) {
    setHistoryStatus('Loading prediction details…');
    setHistoryError(null);
    try {
      const response = await fetch(apiEndpoint(apiBase, `/predictions/${id}`));
      const payload = await readJsonOrEmpty(response);
      if (!response.ok) {
        throw new Error(payload.detail || `The API returned status ${response.status}.`);
      }
      setDetail(payload);
      requestAnimationFrame(() => {
        detailSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      });
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Unable to load the prediction details.');
    } finally {
      setHistoryStatus('');
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError(null);
    setResult(null);

    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      setError('Choose an image before requesting an explanation.');
      return;
    }
    if (!ALLOWED_TYPES.has(file.type)) {
      setError('Upload a JPEG, PNG, or WebP image.');
      return;
    }

    let endpoint;
    try {
      endpoint = apiEndpoint(apiBase, '/explain');
    } catch {
      setError(`Enter a valid API address, such as ${DEFAULT_API_BASE}.`);
      return;
    }

    const formData = new FormData();
    formData.append('file', file, file.name);
    setSubmitting(true);
    setLoadingStatus('Generating prediction and Grad-CAM visualization…');

    try {
      const response = await fetch(endpoint, { method: 'POST', body: formData });
      const payload = await readJsonOrEmpty(response);
      if (!response.ok) {
        throw new Error(payload.detail || `The API returned status ${response.status}.`);
      }
      if (!payload.original_image?.data_url || !payload.gradcam_visualization?.data_url) {
        throw new Error('The API response did not include the requested explanation images.');
      }
      setResult(payload);
      // Refresh the stored history so the just-completed analysis appears.
      loadHistory(0);
      requestAnimationFrame(() => {
        resultSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    } catch (err) {
      if (err instanceof TypeError) {
        // The browser could not connect at all: the API is unreachable.
        setError(
          `Could not reach the API at ${apiBase.trim()}. ` +
            'Ensure the backend is running and the API address is correct.'
        );
      } else {
        setError(err instanceof Error ? err.message : 'Unable to generate an AI explanation.');
      }
    } finally {
      setSubmitting(false);
      setLoadingStatus('');
    }
  }

  async function handleClearHistory() {
    const confirmed = window.confirm(
      'Delete every stored prediction record? This cannot be undone. (Uploaded images were never stored.)'
    );
    if (!confirmed) return;
    setClearing(true);
    setHistoryStatus('Clearing prediction history…');
    setHistoryError(null);
    try {
      const response = await fetch(apiEndpoint(apiBase, '/predictions'), { method: 'DELETE' });
      const payload = await readJsonOrEmpty(response);
      if (!response.ok) {
        throw new Error(payload.detail || `The API returned status ${response.status}.`);
      }
      loadHistory(0);
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Unable to clear prediction history.');
    } finally {
      setClearing(false);
      setHistoryStatus('');
    }
  }

  const hasHistoryRows = historyItems !== null && historyItems.length > 0;

  return (
    <main>
      <header>
        <h1>Skin-Lesion AI Classification</h1>
        <p className="lead">
          Upload one image to receive an educational/research AI classification and an explanation
          of the image regions that influenced that prediction.
        </p>
      </header>

      <section className="notice" aria-label="Important medical notice">
        This is an educational and research AI classification system, not a medical diagnosis or a
        substitute for clinical assessment.
      </section>

      <section className="card" aria-labelledby="upload-heading">
        <h2 id="upload-heading">Analyze an image</h2>
        <form id="explain-form" className="upload-form" onSubmit={handleSubmit}>
          <div className="form-row">
            <label className="field-label" htmlFor="api-base">
              API address
              <input
                id="api-base"
                type="url"
                value={apiBase}
                onChange={(event) => setApiBase(event.target.value)}
                autoComplete="url"
                required
              />
            </label>
            <label className="field-label" htmlFor="image-file">
              Skin image (JPEG, PNG, or WebP)
              <input
                id="image-file"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                ref={fileInputRef}
                required
              />
            </label>
          </div>
          <button id="submit-button" type="submit" disabled={submitting}>
            Classify and explain
          </button>
          <p id="loading-status" className="status" role="status" aria-live="polite">
            {loadingStatus}
          </p>
          {error ? (
            <p id="error-message" className="error" role="alert">
              {error}
            </p>
          ) : null}
        </form>
      </section>

      {result ? (
        <section
          id="result-section"
          ref={resultSectionRef}
          className="card"
          aria-labelledby="result-heading"
        >
          <h2 id="result-heading">Prediction result</h2>
          <div className="result-summary">
            <span id="predicted-class" className="class-name">
              {formatClass(result)}
            </span>
            <span id="confidence" className="confidence">
              Confidence: {formatPercent(result.confidence)}
            </span>
          </div>

          <section aria-labelledby="probabilities-heading">
            <h3 id="probabilities-heading">All-class probabilities</h3>
            <p className="history-summary">
              Model probabilities for every supported class, highest first.
            </p>
            <ProbabilityList probabilities={result.probabilities} />
          </section>

          <section aria-labelledby="explanation-heading">
            <h2 id="explanation-heading">AI Explanation</h2>
            <div className="images-grid">
              <figure className="image-card">
                <img
                  id="original-image"
                  src={result.original_image.data_url}
                  alt="Original uploaded skin image"
                />
                <figcaption>Original uploaded image</figcaption>
              </figure>
              <figure className="image-card">
                <img
                  id="gradcam-image"
                  src={result.gradcam_visualization.data_url}
                  alt="Grad-CAM visualization showing model-influenced image regions"
                />
                <figcaption>Grad-CAM visualization</figcaption>
              </figure>
            </div>
            <p id="explanation-text" className="explanation-copy">
              {result.explanation ||
                'Highlighted regions indicate areas that influenced the model prediction.'}
            </p>
            <p className="safe-copy">
              The visualization shows model influence only. It does not prove that a disease is
              present.
            </p>
          </section>
        </section>
      ) : null}

      <section id="history-section" className="card" aria-labelledby="history-heading">
        <div className="history-header">
          <h2 id="history-heading">Prediction history</h2>
          <div className="history-actions">
            <button
              id="refresh-history"
              type="button"
              className="secondary-button"
              onClick={() => loadHistory(0)}
              disabled={clearing}
            >
              Refresh
            </button>
            <button
              id="clear-history"
              type="button"
              className="danger-button"
              onClick={handleClearHistory}
              disabled={clearing}
            >
              Clear history
            </button>
          </div>
        </div>
        <p id="history-status" className="status" role="status" aria-live="polite">
          {historyStatus}
        </p>
        {historyError ? (
          <p id="history-error" className="error" role="alert">
            {historyError}
          </p>
        ) : null}

        {historyItems !== null && historyItems.length === 0 ? (
          <p id="history-empty" className="history-empty">
            No predictions stored yet. Completed analyses will appear here after each
            classification.
          </p>
        ) : null}

        {hasHistoryRows ? (
          <div id="history-content">
            <p id="history-summary" className="history-summary">
              {historyTotal} stored prediction{historyTotal === 1 ? '' : 's'}.
            </p>
            <div className="table-scroll">
              <table className="history-table">
                <thead>
                  <tr>
                    <th scope="col">Image file</th>
                    <th scope="col">Predicted category</th>
                    <th scope="col">Confidence</th>
                    <th scope="col">Date/time</th>
                    <th scope="col">
                      <span className="visually-hidden">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {historyItems.map((item) => (
                    <tr key={item.id}>
                      <td className="filename">{item.image_filename}</td>
                      <td>{formatClass(item)}</td>
                      <td>{formatPercent(item.confidence)}</td>
                      <td>{formatDateTime(item.created_at)}</td>
                      <td>
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() => loadPredictionDetail(item.id)}
                        >
                          Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <button
                id="history-prev"
                type="button"
                className="secondary-button"
                disabled={historyOffset <= 0}
                onClick={() => loadHistory(Math.max(0, historyOffset - HISTORY_PAGE_SIZE))}
              >
                Previous
              </button>
              <span id="pagination-label">
                Showing {historyOffset + 1}–{historyOffset + historyItems.length} of {historyTotal}
              </span>
              <button
                id="history-next"
                type="button"
                className="secondary-button"
                disabled={historyOffset + historyItems.length >= historyTotal}
                onClick={() => loadHistory(historyOffset + HISTORY_PAGE_SIZE)}
              >
                Next
              </button>
            </div>
          </div>
        ) : null}

        {detail ? (
          <section
            id="history-detail"
            ref={detailSectionRef}
            className="history-detail"
            aria-labelledby="detail-heading"
          >
            <h3 id="detail-heading">Prediction details</h3>
            <dl id="detail-fields" className="detail-grid">
              <dt>Record ID</dt>
              <dd>#{detail.id}</dd>
              <dt>Image file</dt>
              <dd>{detail.image_filename}</dd>
              <dt>Predicted category</dt>
              <dd>{formatClass(detail)}</dd>
              <dt>Confidence</dt>
              <dd>{formatPercent(detail.confidence)}</dd>
              <dt>Date/time</dt>
              <dd>{formatDateTime(detail.created_at)}</dd>
            </dl>
            <h4>Class probabilities</h4>
            <ProbabilityList probabilities={detail.class_probabilities} />
            <button
              id="close-detail"
              type="button"
              className="secondary-button"
              onClick={() => setDetail(null)}
            >
              Close details
            </button>
          </section>
        ) : null}
      </section>
    </main>
  );
}
