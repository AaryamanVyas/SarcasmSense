import { useState } from 'react';
import { Card, Skeleton } from './common';

const MAX_TEXT = 2000;
const METHODS = [
  {
    value: 'lime',
    label: 'LIME (fast, default)',
    hint: 'Local linear surrogate over ~200 perturbations. Typically under a second.',
  },
  {
    value: 'occlusion',
    label: 'Token occlusion (fastest, exact)',
    hint: 'Leave-one-word-out. One forward pass per word; exact and deterministic.',
  },
  {
    value: 'shap',
    label: 'SHAP (audit mode, slow)',
    hint: 'Partition SHAP / Owen values. Principled but can take tens of seconds.',
  },
];

/**
 * Input form: target utterance, up to three preceding turns, explanation method,
 * and one-click demonstration examples.
 */
export default function InputPanel({
  text,
  setText,
  context,
  setContext,
  method,
  setMethod,
  onAnalyze,
  onClear,
  loading,
  disabled,
  examples,
  examplesLoading,
  onPickExample,
}) {
  const [touched, setTouched] = useState(false);
  const trimmed = text.trim();
  const tooLong = text.length > MAX_TEXT;
  const empty = trimmed.length === 0;
  const invalid = touched && (empty || tooLong);

  const setTurn = (index, value) => {
    const next = [...context];
    next[index] = value;
    setContext(next);
  };

  const submit = (event) => {
    event.preventDefault();
    setTouched(true);
    if (empty || tooLong || disabled) return;
    onAnalyze();
  };

  const selected = METHODS.find((m) => m.value === method);

  return (
    <form onSubmit={submit} noValidate>
      <Card
        title="Analyse an utterance"
        hint="Enter the statement to classify. Conversational context is optional but usually decisive."
      >
        <div className="field">
          <div className="field__row">
            <label htmlFor="target">Target utterance</label>
            <span className="field__count">
              {text.length} / {MAX_TEXT}
            </span>
          </div>
          <textarea
            id="target"
            rows={3}
            value={text}
            placeholder="e.g. Great, my laptop crashed five minutes before the presentation."
            onChange={(event) => setText(event.target.value)}
            onBlur={() => setTouched(true)}
            aria-invalid={invalid ? 'true' : 'false'}
            aria-describedby={invalid ? 'target-error' : undefined}
          />
          {invalid && (
            <p className="field__error" id="target-error" role="alert">
              {empty
                ? 'Enter some text to analyse.'
                : `Text must be at most ${MAX_TEXT} characters.`}
            </p>
          )}
        </div>

        <div className="card__section">
          <label style={{ marginBottom: 10 }}>
            Conversational context — up to 3 preceding turns, oldest first (optional)
          </label>
          {[0, 1, 2].map((index) => (
            <div className="field" key={index}>
              <label htmlFor={`turn-${index}`} style={{ fontWeight: 500 }}>
                Previous turn {index + 1}
                {index === 2 ? ' (immediately before the target)' : ''}
              </label>
              <textarea
                id={`turn-${index}`}
                rows={2}
                value={context[index] ?? ''}
                placeholder={
                  index === 2 ? 'The turn the target utterance replies to' : 'Earlier turn'
                }
                onChange={(event) => setTurn(index, event.target.value)}
              />
            </div>
          ))}
        </div>

        <div className="card__section">
          <div className="field">
            <label htmlFor="method">Explanation method</label>
            <select
              id="method"
              value={method}
              onChange={(event) => setMethod(event.target.value)}
            >
              {METHODS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            {selected && <p className="card__hint" style={{ marginTop: 6 }}>{selected.hint}</p>}
          </div>

          <div className="actions">
            <button className="button" type="submit" disabled={loading || disabled}>
              {loading && <span className="spinner" aria-hidden="true" />}
              {loading ? 'Analysing…' : 'Analyze Sarcasm'}
            </button>
            <button
              className="button button--ghost"
              type="button"
              onClick={onClear}
              disabled={loading}
            >
              Clear
            </button>
          </div>
          {method === 'shap' && (
            <p className="card__hint" style={{ marginTop: 10 }}>
              SHAP runs in audit mode and may take tens of seconds on CPU. The request
              will not be abandoned silently — a timeout is reported explicitly.
            </p>
          )}
        </div>
      </Card>

      <Card
        title="Demonstration examples"
        hint="Illustrative inputs for the demo. They are not drawn from any dataset split and no reported metric uses them."
      >
        {examplesLoading ? (
          <Skeleton lines={4} />
        ) : examples.length === 0 ? (
          <p className="card__hint">Examples are unavailable while the backend is unreachable.</p>
        ) : (
          <div className="example-list">
            {examples.map((example) => (
              <button
                key={example.id}
                type="button"
                className="example"
                onClick={() => onPickExample(example)}
                title={example.note}
              >
                <span className="example__title">{example.title}</span>
                <span className="example__text">
                  “{example.text}”
                  {example.context.length > 0 && ` · ${example.context.length} context turn(s)`}
                </span>
              </button>
            ))}
          </div>
        )}
      </Card>
    </form>
  );
}
