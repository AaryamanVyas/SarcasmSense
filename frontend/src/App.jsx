import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, api, BASE_URL } from './api';
import { Banner } from './components/common';
import DashboardPanel from './components/DashboardPanel';
import InputPanel from './components/InputPanel';
import ModelInfoPanel from './components/ModelInfoPanel';
import ResultPanel from './components/ResultPanel';

const TABS = [
  { id: 'analyse', label: 'Analyse' },
  { id: 'model', label: 'Model information' },
  { id: 'dashboard', label: 'Performance dashboard' },
];

const EMPTY_CONTEXT = ['', '', ''];

/**
 * Root component.
 *
 * Owns all cross-panel state: backend health (polled while unhealthy), model
 * metadata, the current analysis, and every loading/error flag. Each UI state
 * required by the specification — loading, model initialising, invalid input,
 * API error, retrieval failure, no context, no evidence, low confidence, slow
 * explanation, backend unavailable — is surfaced explicitly somewhere below.
 */
export default function App() {
  const [tab, setTab] = useState('analyse');
  const [focusedTabIndex, setFocusedTabIndex] = useState(0);
  const tabRefs = useRef([]);

  const activateTab = (id, index) => {
    setTab(id);
    setFocusedTabIndex(index);
  };

  const handleTabKeyDown = (event, index) => {
    const lastIndex = TABS.length - 1;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      event.preventDefault();
      moveFocus(index === lastIndex ? 0 : index + 1);
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      event.preventDefault();
      moveFocus(index === 0 ? lastIndex : index - 1);
    } else if (event.key === 'Home') {
      event.preventDefault();
      moveFocus(0);
    } else if (event.key === 'End') {
      event.preventDefault();
      moveFocus(lastIndex);
    }
  };

  const moveFocus = (next) => {
    setFocusedTabIndex(next);
    tabRefs.current[next]?.focus();
  };

  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);

  const [modelInfo, setModelInfo] = useState(null);
  const [modelInfoError, setModelInfoError] = useState(null);
  const [modelInfoLoading, setModelInfoLoading] = useState(true);

  const [examples, setExamples] = useState([]);
  const [examplesLoading, setExamplesLoading] = useState(true);

  const [text, setText] = useState('');
  const [context, setContext] = useState(EMPTY_CONTEXT);
  const [method, setMethod] = useState('lime');

  const [result, setResult] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [analysing, setAnalysing] = useState(false);

  const [results, setResults] = useState(null);
  const [resultsError, setResultsError] = useState(null);
  const [resultsLoading, setResultsLoading] = useState(false);

  /** Poll health until the backend is ready, then stop. */
  const checkHealth = useCallback(async () => {
    try {
      const payload = await api.health();
      setHealth(payload);
      setHealthError(null);
      return payload;
    } catch (error) {
      setHealth(null);
      setHealthError(error);
      return null;
    }
  }, []);

  const loadModelInfo = useCallback(async () => {
    setModelInfoLoading(true);
    try {
      setModelInfo(await api.modelInfo());
      setModelInfoError(null);
    } catch (error) {
      setModelInfo(null);
      setModelInfoError(error);
    } finally {
      setModelInfoLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    const tick = async () => {
      const payload = await checkHealth();
      if (cancelled) return;
      if (payload?.model_loaded) {
        loadModelInfo();
      } else {
        // Retry while the model is still loading or the backend is down.
        timer = setTimeout(tick, 4000);
      }
    };
    tick();

    api
      .examples()
      .then((data) => !cancelled && setExamples(data))
      .catch(() => !cancelled && setExamples([]))
      .finally(() => !cancelled && setExamplesLoading(false));

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [checkHealth, loadModelInfo]);

  const loadResults = useCallback(async () => {
    setResultsLoading(true);
    try {
      setResults(await api.results());
      setResultsError(null);
    } catch (error) {
      setResults(null);
      setResultsError(error);
    } finally {
      setResultsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (tab === 'dashboard' && results === null && !resultsLoading) {
      loadResults();
    }
  }, [tab, results, resultsLoading, loadResults]);

  const analyze = async () => {
    setAnalysing(true);
    setAnalysisError(null);
    try {
      const payload = await api.predict({
        text: text.trim(),
        context: context.map((turn) => turn.trim()).filter(Boolean),
        explanation_method: method,
        explain: true,
      });
      setResult(payload);
    } catch (error) {
      setResult(null);
      setAnalysisError(error instanceof ApiError ? error : new ApiError(String(error)));
      // A 503 usually means the model dropped out; re-check readiness.
      if (error?.kind === 'unavailable' || error?.kind === 'network') checkHealth();
    } finally {
      setAnalysing(false);
    }
  };

  const clearAll = () => {
    setText('');
    setContext(EMPTY_CONTEXT);
    setResult(null);
    setAnalysisError(null);
  };

  const pickExample = (example) => {
    setText(example.text);
    const turns = [...example.context];
    while (turns.length < 3) turns.unshift('');
    setContext(turns.slice(-3));
    setResult(null);
    setAnalysisError(null);
  };

  const backendDown = Boolean(healthError);
  const modelLoading = health && !health.model_loaded && health.status === 'loading';
  const modelFailed = health && !health.model_loaded && health.status === 'error';
  const degraded = health?.status === 'degraded';

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header__inner">
          <h1>Context-Aware Sarcasm Detector</h1>
          <p>
            RoBERTa + conversational context + retrieval-augmented commonsense reasoning +
            explainable AI. A shared RoBERTa encoder reads the utterance and up to three
            preceding turns; a FAISS index supplies top-<i>k</i> commonsense facts; a trainable
            cross-attention layer fuses them; LIME/SHAP and the fusion attention explain the
            result. Text only.
          </p>
          <nav className="tabs" role="tablist" aria-label="Sections">
            {TABS.map((entry, index) => {
              const isSelected = tab === entry.id;
              const isFocused = focusedTabIndex === index;
              return (
                <button
                  key={entry.id}
                  id={`tab-${entry.id}`}
                  role="tab"
                  type="button"
                  className={`tab${isSelected ? ' tab--selected' : ''}`}
                  aria-selected={isSelected}
                  aria-controls={`tabpanel-${entry.id}`}
                  tabIndex={isFocused ? 0 : -1}
                  onClick={() => activateTab(entry.id, index)}
                  onKeyDown={(event) => handleTabKeyDown(event, index)}
                  ref={(el) => (tabRefs.current[index] = el)}
                >
                  {entry.label}
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="main">
        {backendDown && (
          <Banner tone="error" title="Backend unavailable.">
            Could not reach the API at <code>{BASE_URL}</code>. Start it with{' '}
            <code>uvicorn backend.main:app --port 8000</code>, or set{' '}
            <code>VITE_API_BASE_URL</code> to the right origin. Retrying automatically…
          </Banner>
        )}
        {modelLoading && (
          <Banner tone="info" title="Model is initialising.">
            The backend is loading the checkpoint, tokenizer and FAISS index. Analysis will
            become available shortly.
          </Banner>
        )}
        {modelFailed && (
          <Banner tone="error" title="Model failed to load.">
            {health.detail} Check <code>MODEL_PATH</code> and the backend logs.
          </Banner>
        )}
        {degraded && (
          <Banner tone="warn" title="Running in degraded mode.">
            {health.detail}
          </Banner>
        )}

        {tab === 'analyse' && (
          <section id="tabpanel-analyse" role="tabpanel" aria-labelledby="tab-analyse">
            <div className="grid">
              <InputPanel
              text={text}
              setText={setText}
              context={context}
              setContext={setContext}
              method={method}
              setMethod={setMethod}
              onAnalyze={analyze}
              onClear={clearAll}
              loading={analysing}
              disabled={!health?.model_loaded}
              examples={examples}
              examplesLoading={examplesLoading}
              onPickExample={pickExample}
            />
            <div>
              <ResultPanel
                result={result}
                loading={analysing}
                error={analysisError}
                modelInfo={modelInfo}
              />
            </div>
          </div>
          </section>
        )}

        {tab === 'model' && (
          <section id="tabpanel-model" role="tabpanel" aria-labelledby="tab-model">
            <ModelInfoPanel
            info={modelInfo}
            loading={modelInfoLoading}
            error={modelInfoError}
          />
          </section>
        )}

        {tab === 'dashboard' && (
          <section id="tabpanel-dashboard" role="tabpanel" aria-labelledby="tab-dashboard">
            <DashboardPanel
            results={results}
            loading={resultsLoading}
            error={resultsError}
            onRefresh={loadResults}
          />
          </section>
        )}
      </main>

      <footer className="footer">
        Context-Aware Explainable Sarcasm Detection · API base <code>{BASE_URL}</code>
        {health?.device ? ` · serving on ${health.device}` : ''}
        {modelInfo?.is_development_asset ? ' · DEVELOPMENT CHECKPOINT' : ''}
      </footer>
    </div>
  );
}
