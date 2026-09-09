import { Banner, Card, Empty, Skeleton } from './common';

/** Format a boolean as a readable Yes/No. */
const yesNo = (value) => (value ? 'Yes' : 'No');

/**
 * "Model information" view — the architecture card required by the project
 * specification, filled entirely from `GET /model/info` rather than hard-coded.
 */
export default function ModelInfoPanel({ info, loading, error }) {
  if (loading) {
    return (
      <Card title="Model information">
        <Skeleton lines={8} />
      </Card>
    );
  }
  if (error || !info) {
    return (
      <Card title="Model information">
        <Banner tone="error" title="Unavailable.">
          {error?.message ?? 'The backend did not return model information.'}
        </Banner>
      </Card>
    );
  }

  const model = info.model ?? {};
  const retrieval = info.retrieval ?? {};
  const index = retrieval.index ?? null;
  const encoder = retrieval.encoder ?? null;
  const explanation = info.explanation ?? {};

  const rows = [
    ['Model', model.encoder_name],
    ['Task', info.task],
    ['Input', info.input_modality],
    ['Hidden size', model.hidden_size],
    ['Max utterance length', `${model.max_utterance_length} tokens`],
    ['Max context length', `${model.max_context_length} tokens`],
    ['Context', `Up to ${info.max_context_turns} previous turns · used: ${yesNo(model.uses_context)}`],
    ['Shared encoder', yesNo(model.shared_encoder)],
    ['Knowledge', `Consumed by classifier: ${yesNo(model.uses_knowledge)}`],
    ['Retrieval', encoder ? `${encoder.name} + FAISS` : 'not configured'],
    ['Fusion', `${info.fusion} · ${model.fusion_layers} layer(s) · ${model.attention_heads} heads`],
    ['Classifier', `${model.classifier_input_size} → ${model.classifier_hidden} → ${model.num_labels}`],
    ['Explanation', Object.entries(explanation.available_methods ?? {}).filter(([, ok]) => ok).map(([name]) => name.toUpperCase()).join(' / ') || 'none'],
    [
      'Parameters',
      model.n_parameters
        ? `${(model.n_parameters / 1e6).toFixed(1)} M` +
          // The served model has gradients disabled, so a trainable count of
          // zero is expected and would otherwise read as a bug.
          (model.n_trainable_parameters > 0
            ? ` (${(model.n_trainable_parameters / 1e6).toFixed(1)} M trainable)`
            : ' (inference mode — gradients disabled)')
        : '—',
    ],
    ['Device', info.device],
    ['Experiment', info.experiment],
    ['Checkpoint', info.checkpoint],
  ];

  return (
    <>
      {info.is_development_asset && (
        <Banner tone="error" title="Development / test checkpoint.">
          This server is running a tiny randomly-initialised encoder built for pipeline
          testing. Its predictions are structurally valid but carry no research meaning,
          and must not be reported as results.
        </Banner>
      )}
      {(info.startup_warnings ?? []).length > 0 && (
        <Banner tone="warn" title="Startup warnings">
          <ul>
            {info.startup_warnings.map((warning, i) => (
              <li key={i}>{warning}</li>
            ))}
          </ul>
        </Banner>
      )}

      <div className="grid">
        <Card title="Architecture" hint="Reported live by the running backend.">
          <dl className="kv">
            {rows.map(([label, value]) => (
              <div key={label} style={{ display: 'contents' }}>
                <dt>{label}</dt>
                <dd>{value ?? '—'}</dd>
              </div>
            ))}
          </dl>
        </Card>

        <div>
          <Card title="Commonsense knowledge index">
            {!index ? (
              <Empty>
                No knowledge index is loaded. Predictions run without retrieved evidence.
              </Empty>
            ) : (
              <>
                {index.is_sample_index && (
                  <Banner tone="warn" title="Sample index.">
                    This is the small smoke-test index, not a research knowledge base.
                  </Banner>
                )}
                {index.encoder_is_fallback && (
                  <Banner tone="warn" title="Fallback sentence encoder.">
                    The index was built with the lexical hashing encoder rather than the
                    configured Sentence-Transformer. Retrieval is lexical, not semantic.
                  </Banner>
                )}
                <dl className="kv">
                  <dt>Facts indexed</dt>
                  <dd>{Number(index.n_facts).toLocaleString()}</dd>
                  <dt>Sources</dt>
                  <dd>
                    {Object.entries(index.sources ?? {})
                      .map(([source, count]) => `${source}: ${Number(count).toLocaleString()}`)
                      .join(', ') || '—'}
                  </dd>
                  <dt>Index type</dt>
                  <dd>
                    {index.index_type}
                    {index.nlist ? ` (nlist=${index.nlist}, nprobe=${index.nprobe})` : ''}
                  </dd>
                  <dt>Embedding dim</dt>
                  <dd>{index.dimension}</dd>
                  <dt>Encoder</dt>
                  <dd>{index.encoder_name}</dd>
                  <dt>Top-k per query</dt>
                  <dd>{retrieval.top_k}</dd>
                  <dt>Query mode</dt>
                  <dd>{retrieval.query_mode}</dd>
                  <dt>Retrieval calls</dt>
                  <dd>
                    {retrieval.calls} ({retrieval.failures} failed, {retrieval.cache_hits} cached)
                  </dd>
                </dl>
              </>
            )}
          </Card>

          <Card
            title="Pipeline"
            hint="What happens between your text and the answer."
          >
            <ol style={{ paddingLeft: 20, margin: 0, fontSize: '0.88rem', lineHeight: 1.85 }}>
              <li>Conservative preprocessing (punctuation, casing and emoji preserved)</li>
              <li>Offline commonsense retrieval: sentence embedding → FAISS → top-k facts</li>
              <li>Shared RoBERTa encodes the utterance and the flattened context</li>
              <li>
                {info.fusion === 'cross_attention'
                  ? 'Cross-attention fusion: utterance queries attend over context tokens + evidence vectors'
                  : 'Concatenation fusion (ablation variant)'}
              </li>
              <li>Classification head → 2-class softmax → confidence</li>
              <li>Token attribution (LIME / SHAP / occlusion) over the utterance</li>
              <li>Evidence-grounded natural-language explanation</li>
            </ol>
          </Card>
        </div>
      </div>
    </>
  );
}
