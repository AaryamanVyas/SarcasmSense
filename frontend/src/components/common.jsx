/** Small presentational primitives shared across panels. */

/** A titled card section. */
export function Card({ title, hint, children, right = null }) {
  return (
    <section className="card">
      {(title || right) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12 }}>
          {title && <h2 className="card__title">{title}</h2>}
          {right}
        </div>
      )}
      {hint && <p className="card__hint">{hint}</p>}
      {children}
    </section>
  );
}

/** Coloured status banner. `tone` is one of info | warn | error | ok. */
export function Banner({ tone = 'info', title, children }) {
  const icon = { info: 'i', warn: '!', error: '×', ok: '✓' }[tone] ?? 'i';
  return (
    <div className={`banner banner--${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <span className="banner__icon" aria-hidden="true">
        {icon}
      </span>
      <div>
        {title && <strong>{title}</strong>}
        {title && children ? ' ' : null}
        {children}
      </div>
    </div>
  );
}

/** Placeholder shown while a panel is loading. */
export function Skeleton({ lines = 3 }) {
  return (
    <div aria-hidden="true">
      {Array.from({ length: lines }, (_, i) => (
        <div
          key={i}
          className="skeleton skeleton--line"
          style={{ width: `${100 - i * 12}%` }}
        />
      ))}
    </div>
  );
}

/** Empty-state message. */
export function Empty({ children }) {
  return <p className="empty">{children}</p>;
}

/** Format a 0..1 number as a percentage string. */
export function pct(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

/** Format a number to fixed decimals, tolerating null. */
export function num(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}
