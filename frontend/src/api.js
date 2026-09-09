/**
 * Typed-ish client for the FastAPI backend.
 *
 * The base URL comes from `VITE_API_BASE_URL` (see `.env.example`) and defaults
 * to `/api`, which the Vite dev server proxies to the backend. Nothing here
 * hard-codes `localhost`, so the same build runs behind any reverse proxy.
 *
 * Every call funnels through `request()` so that timeouts, aborts, non-JSON
 * responses and the backend's uniform `{error, detail, request_id}` envelope are
 * handled in exactly one place -- the UI then only ever sees an `ApiError` with
 * a message worth showing a human.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/+$/, '');

/** Milliseconds before a request is aborted. SHAP is genuinely slow. */
export const TIMEOUTS = {
  default: 20000,
  predict: 60000,
  shap: 180000,
};

/** Error carrying a user-presentable message plus machine-readable details. */
export class ApiError extends Error {
  constructor(message, { status = 0, kind = 'unknown', requestId = null, detail = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.kind = kind; // 'network' | 'timeout' | 'validation' | 'unavailable' | 'server'
    this.requestId = requestId;
    this.detail = detail;
  }
}

async function request(path, { method = 'GET', body = null, timeout = TIMEOUTS.default } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (error) {
    clearTimeout(timer);
    if (error.name === 'AbortError') {
      throw new ApiError(
        `The request took longer than ${Math.round(timeout / 1000)} s and was cancelled. ` +
          'SHAP explanations are expensive — try LIME or the fast token-importance mode.',
        { kind: 'timeout' },
      );
    }
    throw new ApiError(
      'Could not reach the backend. Check that the API is running and that ' +
        'VITE_API_BASE_URL points at it.',
      { kind: 'network' },
    );
  }
  clearTimeout(timer);

  const requestId = response.headers.get('X-Request-ID');
  let payload = null;
  const text = await response.text();
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    const detail = payload?.detail ?? text ?? response.statusText;
    const kind =
      response.status === 422 ? 'validation' : response.status === 503 ? 'unavailable' : 'server';
    throw new ApiError(detail || `Request failed with status ${response.status}`, {
      status: response.status,
      kind,
      requestId,
      detail: payload,
    });
  }
  return payload;
}

export const api = {
  /** Liveness + readiness. Never throws for a *degraded* backend, only for an unreachable one. */
  health: () => request('/health', { timeout: 8000 }),
  /** Architecture, retrieval and explainability metadata. */
  modelInfo: () => request('/model/info'),
  /** Demonstration inputs for the demo mode. */
  examples: () => request('/examples'),
  /** Saved experiment results for the dashboard. */
  results: () => request('/results', { timeout: 15000 }),
  /** Retrieval only, no classification. */
  retrieve: (payload) => request('/retrieve', { method: 'POST', body: payload }),
  /** Classify + explain. */
  predict: (payload) =>
    request('/predict', {
      method: 'POST',
      body: payload,
      timeout: payload.explanation_method === 'shap' ? TIMEOUTS.shap : TIMEOUTS.predict,
    }),
};

export { BASE_URL };
