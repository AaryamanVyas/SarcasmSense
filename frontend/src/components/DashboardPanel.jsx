import { Banner, Card, Empty, Skeleton, num, pct } from './common';

const LABELS = ['Not Sarcastic', 'Sarcastic'];

/** Render a 2x2 confusion matrix from the saved metrics. */
function ConfusionMatrix({ matrix }) {
  if (!Array.isArray(matrix) || matrix.length !== 2) return null;
  return (
    <table className="matrix">
      <thead>
        <tr>
          <th />
          <th colSpan={2}>Predicted</th>
        </tr>
        <tr>
          <th>True</th>
          {LABELS.map((label) => (
            <th key={label}>{label}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={i}>
            <th>{LABELS[i]}</th>
            {row.map((value, j) => (
              <td key={j} className={i === j ? 'diag' : undefined}>
                {Number(value).toLocaleString()}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Table of runs / ablation variants. */
function MetricsTable({ rows, firstColumn }) {
  if (rows.length === 0) return <Empty>No rows.</Empty>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>{firstColumn}</th>
            <th>n</th>
            <th>Accuracy</th>
            <th>Macro-F1</th>
            <th>Precision (sarc.)</th>
            <th>Recall (sarc.)</th>
            <th>ROC-AUC</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key}>
              <td>
                {row.name}
                {row.dev && (
                  <span className="chip" style={{ marginLeft: 8 }}>
                    dev asset
                  </span>
                )}
              </td>
              <td className="num">{row.metrics.n_examples ?? '—'}</td>
              <td className="num">{num(row.metrics.accuracy)}</td>
              <td className="num">
                <b>{num(row.metrics.macro_f1)}</b>
              </td>
              <td className="num">{num(row.metrics.sarcastic_precision)}</td>
              <td className="num">{num(row.metrics.sarcastic_recall)}</td>
              <td className="num">{num(row.metrics.roc_auc)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Performance dashboard. Displays **only** results that exist on disk; when no
 * experiment has been run it says so instead of showing placeholder numbers.
 */
export default function DashboardPanel({ results, loading, error, onRefresh }) {
  if (loading) {
    return (
      <Card title="Experiment results">
        <Skeleton lines={8} />
      </Card>
    );
  }
  if (error) {
    return (
      <Card title="Experiment results">
        <Banner tone="error" title="Could not load results.">
          {error.message}
        </Banner>
      </Card>
    );
  }
  if (!results?.available) {
    return (
      <Card title="Experiment results">
        <Empty>
          No experiment results available yet.
          <br />
          <span style={{ fontSize: '0.82rem' }}>
            {results?.message ??
              'Train a model with `python scripts/train.py --config configs/full_model.yaml`.'}
          </span>
        </Empty>
      </Card>
    );
  }

  const runs = results.experiments ?? [];
  const withTest = runs.filter((run) => run.test_metrics && run.test_metrics.n_examples);
  const anyDev = runs.some((run) => run.is_development_asset);
  const latest = withTest.length > 0 ? withTest[withTest.length - 1] : null;
  const ablation = results.ablation;
  const agreement = ablation?.explanation_agreement;
  const benchmark = results.benchmark;
  const dataset = results.dataset;

  return (
    <>
      {anyDev && (
        <Banner tone="warn" title="Some runs used a development/test checkpoint.">
          Rows marked “dev asset” were produced with a tiny randomly-initialised encoder to
          verify the pipeline. They are not research results.
        </Banner>
      )}

      <Card
        title="Runs"
        hint="Every training run recorded in results/experiments.json, in chronological order."
        right={
          <button className="button button--ghost button--small" onClick={onRefresh} type="button">
            Refresh
          </button>
        }
      >
        {withTest.length === 0 ? (
          <Empty>
            No completed training run has test-set metrics yet.
            <br />
            <span style={{ fontSize: '0.82rem' }}>
              Runs appear here after <code>python scripts/train.py</code> finishes without{' '}
              <code>--skip-test</code>.
            </span>
          </Empty>
        ) : (
          <>
            <MetricsTable
              firstColumn="Run"
              rows={withTest.map((run) => ({
                key: run.run_id,
                name: `${run.model_type} · ${run.experiment}`,
                metrics: run.test_metrics,
                dev: run.is_development_asset,
              }))}
            />
            <p className="table-note">
              Metrics are on the held-out <b>test</b> split. Primary metric: macro-F1.
            </p>
          </>
        )}
      </Card>

      {latest && (
        <div className="grid">
          <Card title={`Confusion matrix — ${latest.experiment}`} hint="Most recent run, test split.">
            <ConfusionMatrix matrix={latest.test_metrics.confusion_matrix} />
            <p className="table-note">
              Support: {LABELS.map((l) => `${l} ${latest.test_metrics.support?.[l] ?? '—'}`).join(' · ')}
            </p>
          </Card>
          <Card title="Per-dataset breakdown" hint="Same run, split by source corpus.">
            {Object.keys(latest.per_dataset_metrics ?? {}).length === 0 ? (
              <Empty>No per-dataset breakdown was recorded for this run.</Empty>
            ) : (
              <MetricsTable
                firstColumn="Dataset"
                rows={Object.entries(latest.per_dataset_metrics).map(([name, metrics]) => ({
                  key: name,
                  name,
                  metrics,
                  dev: false,
                }))}
              />
            )}
          </Card>
        </div>
      )}

      <Card title="Ablation study" hint="From results/ablation_results.json.">
        {!ablation ? (
          <Empty>
            No ablation results yet. Run{' '}
            <code>python scripts/run_ablation.py --base configs/full_model.yaml</code>.
          </Empty>
        ) : (
          <>
            <MetricsTable
              firstColumn="Variant"
              rows={(ablation.variants ?? [])
                .filter((v) => !v.error && v.test)
                .map((v) => ({
                  key: v.variant,
                  name: v.variant,
                  metrics: v.test,
                  dev: v.is_development_asset,
                }))}
            />
            <p className="table-note">Base config: {ablation.base_config}</p>
          </>
        )}
      </Card>

      {agreement && agreement.available && (
        <Card
          title="SHAP vs LIME agreement"
          hint={`Controlled sample of ${agreement.n_samples} test examples (seed ${agreement.seed}).`}
        >
          <dl className="kv">
            <dt>Jaccard overlap (top-5 tokens)</dt>
            <dd>{num(agreement.mean_jaccard_top5)}</dd>
            <dt>Rank correlation</dt>
            <dd>{num(agreement.mean_rank_correlation)}</dd>
            <dt>Sign agreement</dt>
            <dd>{pct(agreement.mean_sign_agreement, 1)}</dd>
            <dt>Mean latency (LIME / SHAP)</dt>
            <dd>
              {agreement.mean_lime_ms} ms / {agreement.mean_shap_ms} ms
            </dd>
          </dl>
        </Card>
      )}

      {benchmark && (
        <Card title="Inference latency" hint="From results/benchmark.json — measured, not estimated.">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>Mean (ms)</th>
                  <th>Median (ms)</th>
                  <th>p95 (ms)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(benchmark.stages ?? {}).map(([stage, stats]) => (
                  <tr key={stage}>
                    <td>{stage.replace(/_/g, ' ')}</td>
                    <td className="num">{num(stats.mean_ms, 1)}</td>
                    <td className="num">{num(stats.median_ms, 1)}</td>
                    <td className="num">{num(stats.p95_ms, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="table-note">
            Device: {benchmark.device} · {benchmark.n_runs} runs ·{' '}
            {benchmark.meets_300ms_target === null
              ? 'target not evaluated'
              : benchmark.meets_300ms_target
                ? 'classification + retrieval p95 is within the 300 ms CPU target'
                : 'classification + retrieval p95 exceeds the 300 ms CPU target'}
          </p>
        </Card>
      )}

      {dataset && (
        <Card title="Corpus" hint="From results/dataset_report.json.">
          <dl className="kv">
            <dt>Records</dt>
            <dd>{Number(dataset.n_records).toLocaleString()}</dd>
            <dt>Split sizes</dt>
            <dd>
              {Object.entries(dataset.split?.sizes ?? {})
                .map(([name, size]) => `${name} ${Number(size).toLocaleString()}`)
                .join(' · ')}
            </dd>
            <dt>Leakage check</dt>
            <dd>{dataset.leakage?.clean ? 'PASS' : 'FAIL — see dataset_report.json'}</dd>
            <dt>Context availability</dt>
            <dd>{pct(dataset.merged_report?.context_availability ?? 0, 1)}</dd>
          </dl>
        </Card>
      )}
    </>
  );
}
