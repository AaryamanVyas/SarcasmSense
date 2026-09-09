import { Banner, Card, Empty, Skeleton, pct } from './common';

/** Colour a token by the sign and magnitude of its attribution. */
function tokenStyle(weight) {
  const magnitude = Math.min(Math.abs(weight), 1);
  if (magnitude < 0.04) return {};
  const alpha = 0.12 + magnitude * 0.5;
  return weight > 0
    ? { background: `rgba(183, 121, 31, ${alpha.toFixed(3)})` }
    : { background: `rgba(43, 108, 176, ${alpha.toFixed(3)})` };
}

/** Render the utterance with per-token attribution highlighting. */
function HighlightedText({ text, importance, method }) {
  const weights = new Map();
  importance.forEach((item) => {
    const key = item.token;
    if (!weights.has(key) || Math.abs(item.weight) > Math.abs(weights.get(key))) {
      weights.set(key, item.weight);
    }
  });
  const words = text.split(/(\s+)/);

  return (
    <>
      <p className="tokens">
        {words.map((word, index) => {
          if (/^\s+$/.test(word)) return <span key={index}>{word}</span>;
          const weight = weights.get(word) ?? 0;
          const supports = weight > 0.04;
          const opposes = weight < -0.04;
          return (
            <span
              key={index}
              className={`token ${supports ? 'token--supports' : ''} ${
                opposes ? 'token--opposes' : ''
              }`}
              style={tokenStyle(weight)}
              title={
                weight
                  ? `${method.toUpperCase()} weight ${weight >= 0 ? '+' : ''}${weight.toFixed(3)}`
                  : 'no measurable influence'
              }
            >
              {word}
            </span>
          );
        })}
      </p>
      <div className="legend">
        <span>
          <span className="legend__swatch" style={{ background: 'rgba(183,121,31,0.5)' }} />
          pushed towards the predicted label
        </span>
        <span>
          <span className="legend__swatch" style={{ background: 'rgba(43,108,176,0.5)' }} />
          pushed away from it
        </span>
        <span>weights normalised so the strongest token is ±1.00</span>
      </div>
    </>
  );
}

/** Full result view: verdict, explanation, tokens, context, evidence. */
export default function ResultPanel({ result, loading, error, modelInfo }) {
  if (loading) {
    return (
      <Card title="Analysing…" hint="Retrieval → RoBERTa → cross-attention fusion → attribution.">
        <Skeleton lines={6} />
      </Card>
    );
  }

  if (error) {
    const title =
      { network: 'Backend unavailable', timeout: 'Request timed out', validation: 'Invalid input', unavailable: 'Model not ready' }[
        error.kind
      ] ?? 'Request failed';
    return (
      <Card title="No result">
        <Banner tone="error" title={`${title}:`}>
          {error.message}
          {error.requestId && (
            <div style={{ marginTop: 6, fontSize: '0.78rem' }}>
              Request id: <code>{error.requestId}</code>
            </div>
          )}
        </Banner>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card
        title="Result"
        hint="Enter an utterance on the left, or pick a demonstration example, and press Analyze Sarcasm."
      >
        <Empty>Nothing analysed yet.</Empty>
      </Card>
    );
  }

  const sarcastic = result.label === 1;
  const totalContextAttention = result.context_influence.reduce((sum, c) => sum + c.attention, 0);
  const strongestTurn = result.context_influence.reduce(
    (best, c) => (best === null || c.attention > best.attention ? c : best),
    null,
  );
  const lowConfidence = result.confidence < 0.6;
  const usesKnowledge = modelInfo?.model?.uses_knowledge ?? true;

  return (
    <>
      <Card title="Prediction">
        <div className="verdict">
          <span
            className={`verdict__badge verdict__badge--${sarcastic ? 'sarcastic' : 'literal'}`}
          >
            {result.prediction}
          </span>
          <div className="verdict__confidence">
            <div className="verdict__confidence-value">{pct(result.confidence, 1)}</div>
            <div className="verdict__confidence-label">confidence in the predicted class</div>
            <div className="meter">
              <div
                className={`meter__fill meter__fill--${sarcastic ? 'sarcastic' : 'literal'}`}
                style={{ width: `${Math.max(2, result.confidence * 100)}%` }}
                role="meter"
                aria-valuenow={Math.round(result.confidence * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label="Prediction confidence"
              />
            </div>
            <div className="probabilities">
              <span>
                Sarcastic <b>{pct(result.probabilities.sarcastic, 1)}</b>
              </span>
              <span>
                Not sarcastic <b>{pct(result.probabilities.not_sarcastic, 1)}</b>
              </span>
            </div>
          </div>
        </div>

        {lowConfidence && (
          <div style={{ marginTop: 14 }}>
            <Banner tone="warn" title="Low confidence.">
              The model is close to its decision boundary here. Treat this label as a weak
              signal rather than a verdict.
            </Banner>
          </div>
        )}

        {result.warnings.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <Banner tone="warn" title="System notes">
              <ul>
                {result.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </Banner>
          </div>
        )}
      </Card>

      <Card
        title={`Why was this classified as ${result.prediction.toLowerCase()}?`}
        hint={`Generated from this prediction's own attribution, attention and retrieved evidence — attribution method: ${result.explanation_method.toUpperCase()}.`}
      >
        <p className="explanation">{result.explanation}</p>

        {result.incongruity && (
          <div className="card__section">
            <h3 className="card__title" style={{ fontSize: '0.9rem' }}>
              Incongruity check
            </h3>
            <p className="card__hint" style={{ marginBottom: 8 }}>
              Literal wording polarity vs situational polarity (SentiWordNet + retrieved
              connotation facts).
            </p>
            <Banner tone={result.incongruity.detected ? 'ok' : 'info'}>
              {result.incongruity.detected
                ? `Incongruity detected — ${result.incongruity.description}.`
                : `No incongruity asserted — ${result.incongruity.description}.`}
              <div style={{ marginTop: 6, fontSize: '0.78rem' }}>
                wording polarity {result.incongruity.utterance_polarity.toFixed(2)} · situation
                polarity {result.incongruity.situation_polarity.toFixed(2)} · source:{' '}
                {result.incongruity.situation_source}
              </div>
            </Banner>
          </div>
        )}
      </Card>

      <Card
        title="Influential words"
        hint="Colour intensity is proportional to the normalised attribution weight. Hover a word for its exact value."
      >
        {result.token_importance.length === 0 ? (
          <Empty>No token attribution was produced for this input.</Empty>
        ) : (
          <HighlightedText
            text={result.analysed_text}
            importance={result.token_importance}
            method={result.explanation_method}
          />
        )}
      </Card>

      <Card
        title="Context influence"
        hint="Share of the fusion layer's cross-attention that each preceding turn received."
      >
        {result.context_used.length === 0 ? (
          <Empty>
            No conversational context was supplied. The model used its learned
            empty-context representation and judged the utterance in isolation.
          </Empty>
        ) : (
          <ul className="context-list">
            {result.context_influence.map((turn) => {
              const share = totalContextAttention > 0 ? turn.attention / totalContextAttention : 0;
              const isTop = strongestTurn && turn.turn_index === strongestTurn.turn_index;
              return (
                <li
                  key={turn.turn_index}
                  className={`context-list__item ${isTop ? 'context-list__item--top' : ''}`}
                >
                  <div className="context-list__head">
                    <span>Turn {turn.turn_index}</span>
                    <span>{pct(share, 0)} of context attention</span>
                  </div>
                  <div className="meter" style={{ height: 6 }}>
                    <div
                      className="meter__fill meter__fill--literal"
                      style={{ width: `${Math.max(1, share * 100)}%` }}
                    />
                  </div>
                  <p className="context-list__text">{turn.text}</p>
                </li>
              );
            })}
          </ul>
        )}
      </Card>

      <Card
        title="Retrieved commonsense evidence"
        hint={
          usesKnowledge
            ? 'Top facts retrieved from the offline knowledge index and consumed by the fusion layer.'
            : 'Retrieved for transparency. This model configuration does not feed evidence to the classifier.'
        }
      >
        {result.retrieved_evidence.length === 0 ? (
          <Banner tone="warn" title="No evidence available.">
            Retrieval status: <code>{result.retrieval_status}</code>. The prediction was made
            without commonsense evidence.
          </Banner>
        ) : (
          <ul className="evidence">
            {result.retrieved_evidence.map((fact, index) => (
              <li className="evidence__item" key={fact.fact_id}>
                <p className="evidence__text">{fact.text}</p>
                <div className="evidence__meta">
                  <span className="chip chip--source">{fact.source}</span>
                  {fact.relation && <span className="chip">{fact.relation}</span>}
                  <span>similarity {fact.score.toFixed(3)}</span>
                  {result.evidence_attention[index] !== undefined && (
                    <span className="chip chip--attn">
                      attention {pct(result.evidence_attention[index], 1)}
                    </span>
                  )}
                  <span style={{ color: 'var(--text-faint)' }}>id {fact.fact_id}</span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Timing" hint="Measured for this request, in milliseconds.">
        <dl className="kv">
          {Object.entries(result.timings_ms).map(([stage, value]) => (
            <div key={stage} style={{ display: 'contents' }}>
              <dt>{stage.replace(/_/g, ' ')}</dt>
              <dd>{value.toFixed(1)} ms</dd>
            </div>
          ))}
          <dt>request id</dt>
          <dd>{result.request_id}</dd>
        </dl>
      </Card>
    </>
  );
}
