import { describe, it, expect } from 'vitest';
import React from 'react';
import ReactDOMServer from 'react-dom/server';
import { HighlightedText } from './ResultPanel';

describe('HighlightedText', () => {
  it('renders each token with its attribution weight', () => {
    const html = ReactDOMServer.renderToString(
      <HighlightedText
        text="Great, my laptop crashed."
        importance={[
          { token: 'Great,', weight: 0.8, position: 0 },
          { token: 'my', weight: 0.0, position: 1 },
          { token: 'laptop', weight: -0.6, position: 2 },
          { token: 'crashed.', weight: 0.9, position: 3 },
        ]}
        method="lime"
      />
    );

    expect(html).toContain('Great,');
    expect(html).toContain('my');
    expect(html).toContain('laptop');
    expect(html).toContain('crashed.');
    // LIME method named in the tooltip
    expect(html).toContain('LIME weight');
  });

  it('assigns weights by position so repeated tokens keep distinct weights', () => {
    const html = ReactDOMServer.renderToString(
      <HighlightedText
        text="great great"
        importance={[
          { token: 'great', weight: 0.9, position: 0 },
          { token: 'great', weight: -0.9, position: 1 },
        ]}
        method="occlusion"
      />
    );

    // First occurrence is positive (supports), second is negative (opposes)
    expect(html).toContain('token--supports');
    expect(html).toContain('token--opposes');
    expect(html).toContain('OCCLUSION weight');
  });

  it('leaves unattributed tokens with no influence class', () => {
    const html = ReactDOMServer.renderToString(
      <HighlightedText
        text="hello world"
        importance={[{ token: 'hello', weight: 0.5, position: 0 }]}
        method="shap"
      />
    );

    // 'world' (position 1) has no attribution -> no supports/opposes class
    expect(html).toContain('token--supports');
    expect(html).not.toContain('token--opposes');
    // SHAP named in tooltip
    expect(html).toContain('SHAP weight');
  });

  it('preserves whitespace between tokens', () => {
    const html = ReactDOMServer.renderToString(
      <HighlightedText
        text="a  b"
        importance={[
          { token: 'a', weight: 0.5, position: 0 },
          { token: 'b', weight: 0.5, position: 1 },
        ]}
        method="lime"
      />
    );

    // Two spaces between 'a' and 'b' are preserved as their own span
    expect(html).toContain('<span>  </span>');
  });
});
