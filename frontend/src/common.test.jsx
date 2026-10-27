import { describe, it, expect } from 'vitest';
import React from 'react';
import ReactDOMServer from 'react-dom/server';
import { Card, Banner, Skeleton, Empty, pct, num } from './components/common.jsx';

describe('pct', () => {
  it('formats a 0..1 value as a percentage', () => {
    expect(pct(0.5)).toBe('50%');
    expect(pct(0.5, 1)).toBe('50.0%');
    expect(pct(0.1234, 2)).toBe('12.34%');
    expect(pct(1)).toBe('100%');
    expect(pct(0)).toBe('0%');
  });

  it('returns a dash for null / undefined / NaN', () => {
    expect(pct(null)).toBe('—');
    expect(pct(undefined)).toBe('—');
    expect(pct(NaN)).toBe('—');
  });
});

describe('num', () => {
  it('formats a number to fixed decimals', () => {
    expect(num(0.5)).toBe('0.5000');
    expect(num(0.5, 1)).toBe('0.5');
    expect(num(1 / 3, 3)).toBe('0.333');
    expect(num(0)).toBe('0.0000');
  });

  it('returns a dash for null / undefined / NaN', () => {
    expect(num(null)).toBe('—');
    expect(num(undefined)).toBe('—');
    expect(num(NaN)).toBe('—');
  });
});

describe('presentational primitives', () => {
  it('renders Card with title, hint and children', () => {
    const html = ReactDOMServer.renderToString(
      <Card title="T" hint="H">content</Card>
    );
    expect(html).toContain('T');
    expect(html).toContain('H');
    expect(html).toContain('content');
  });

  it('renders Banner with tone and title', () => {
    const html = ReactDOMServer.renderToString(
      <Banner tone="error" title="Bad">detail</Banner>
    );
    expect(html).toContain('Bad');
    expect(html).toContain('detail');
    expect(html).toContain('banner--error');
  });

  it('renders Skeleton with the requested line count', () => {
    const html = ReactDOMServer.renderToString(<Skeleton lines={5} />);
    expect(html).toContain('skeleton--line');
  });

  it('renders Empty with children', () => {
    const html = ReactDOMServer.renderToString(
      <Empty>nothing here</Empty>
    );
    expect(html).toContain('nothing here');
  });
});
