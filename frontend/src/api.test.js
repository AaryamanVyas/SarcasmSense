import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { api, ApiError, BASE_URL, TIMEOUTS } from './api';

beforeEach(() => {
  vi.spyOn(globalThis, 'fetch').mockImplementation(() =>
    Promise.resolve(
      new Response(JSON.stringify({ status: 'ok' }), { status: 200 })
    )
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('api client', () => {
  it('exposes a default BASE_URL of /api', () => {
    expect(BASE_URL).toBe('/api');
  });

  it('configures sensible timeouts', () => {
    expect(TIMEOUTS.default).toBe(20000);
    expect(TIMEOUTS.predict).toBe(60000);
    expect(TIMEOUTS.shap).toBe(180000);
  });

  it('health performs a GET to /health', async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ status: 'ok' }), { status: 200 })
    );

    await api.health();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/health`);
    expect(opts.method).toBe('GET');
    expect(opts.signal).toBeDefined();
  });

  it('predict performs a POST with a JSON body', async () => {
    const fetchMock = vi.mocked(globalThis.fetch);
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify({ label: 1 }), { status: 200 })
    );

    await api.predict({
      text: 'Great, my laptop crashed.',
      context: ['I have a presentation at three.'],
      explanation_method: 'lime',
      explain: true,
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, opts] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE_URL}/predict`);
    expect(opts.method).toBe('POST');
    expect(opts.headers['Content-Type']).toBe('application/json');
    const body = JSON.parse(opts.body);
    expect(body.text).toBe('Great, my laptop crashed.');
    expect(body.context).toEqual(['I have a presentation at three.']);
    expect(body.explanation_method).toBe('lime');
  });

  it('maps a 422 response to a validation ApiError', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'bad input' }), { status: 422 })
    );

    const error = await api.health().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.kind).toBe('validation');
    expect(error.status).toBe(422);
    expect(error.message).toBe('bad input');
  });

  it('maps a 503 response to an unavailable ApiError', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'model not loaded' }), {
        status: 503,
      })
    );

    const error = await api.health().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.kind).toBe('unavailable');
    expect(error.status).toBe(503);
  });

  it('maps any other non-ok response to a server ApiError', async () => {
    vi.mocked(globalThis.fetch).mockResolvedValueOnce(
      new Response('boom', { status: 500 })
    );

    const error = await api.health().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.kind).toBe('server');
    expect(error.status).toBe(500);
  });

  it('maps a fetch failure to a network ApiError', async () => {
    vi.mocked(globalThis.fetch).mockRejectedValueOnce(
      new TypeError('Failed to fetch')
    );

    const error = await api.health().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.kind).toBe('network');
  });
});
