#!/usr/bin/env python
"""Render the architecture diagrams into ``docs/``.

The figures are *generated*, not drawn by hand, so they stay in sync with the
code and can be regenerated at any resolution:

    python scripts/make_diagrams.py

Produces:
    docs/architecture.png                  -- system architecture (three lanes)
    docs/detailed_model_architecture.png   -- tensor-level model architecture
    docs/data_pipeline.png                 -- corpus normalisation and splitting
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.utils.paths import ensure_dir  # noqa: E402

# --------------------------------------------------------------------- palette
INK = "#1a202c"
MUTED = "#5b6673"
LANE_BG = "#f7f9fb"
LANE_EDGE = "#dfe3e8"

COLOURS = {
    "input": ("#e7f0f9", "#2b6cb0"),
    "encoder": ("#e7f4ec", "#2f855a"),
    "retrieval": ("#fdf3e0", "#b7791f"),
    "fusion": ("#f3e8fb", "#6b46c1"),
    "head": ("#fde8e8", "#c53030"),
    "explain": ("#e8f4f8", "#2c7a7b"),
    "output": ("#eef1f5", "#4a5568"),
    "storage": ("#f0f2f5", "#718096"),
}


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    kind: str = "input",
    title_size: float = 9.5,
    subtitle_size: float = 7.6,
    dashed: bool = False,
):
    """Draw a rounded, two-line labelled node and return its centre."""
    face, edge = COLOURS[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5,
            edgecolor=edge,
            facecolor=face,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    )
    centre_x, centre_y = x + width / 2, y + height / 2
    if subtitle:
        # Anchor the title's baseline just above the centre and the subtitle's
        # top just below it. Using bottom/top anchors (rather than fixed offsets
        # from the centre) keeps them separated no matter how many lines the
        # subtitle has -- fixed offsets are what made multi-line labels collide.
        ax.text(
            centre_x, centre_y + 0.005, title,
            ha="center", va="bottom", fontsize=title_size, fontweight="bold",
            color=INK, zorder=3,
        )
        ax.text(
            centre_x, centre_y + 0.001, subtitle,
            ha="center", va="top", fontsize=subtitle_size, color=MUTED, zorder=3,
        )
    else:
        ax.text(
            centre_x, centre_y, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=INK, zorder=3,
        )
    return centre_x, centre_y


def arrow(ax, start, end, label: str = "", colour: str = MUTED, style: str = "-", rad: float = 0.0):
    """Draw a curved arrow between two points, with an optional label."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=colour,
            linestyle=style,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
            shrinkA=2,
            shrinkB=4,
        )
    )
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(
            mid_x, mid_y, f" {label} ", ha="center", va="center", fontsize=7,
            color=colour, zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.92),
        )


class Lane:
    """A labelled swimlane that stacks its nodes automatically.

    Manual y-coordinates for a dozen boxes are how diagrams end up with labels
    behind boxes, so the lane owns the vertical layout: nodes are added top to
    bottom and the lane keeps track of the cursor.
    """

    def __init__(
        self,
        ax,
        x: float,
        bottom: float,
        width: float,
        top: float,
        title: str,
        inner_pad: float = 0.018,
    ) -> None:
        self.ax = ax
        self.x = x
        self.width = width
        self.top = top
        self.bottom = bottom
        self.inner_pad = inner_pad
        ax.add_patch(
            FancyBboxPatch(
                (x, bottom), width, top - bottom,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                linewidth=1.1, edgecolor=LANE_EDGE, facecolor=LANE_BG, zorder=0,
            )
        )
        # The title sits ABOVE the lane, never on top of a node.
        ax.text(
            x + 0.004, top + 0.008, title, ha="left", va="bottom",
            fontsize=8.6, fontweight="bold", color=MUTED, zorder=1,
        )
        self._cursor = top - inner_pad

    def add(
        self,
        title: str,
        subtitle: str = "",
        kind: str = "input",
        height: float = 0.060,
        gap: float = 0.030,
        inset: float = 0.016,
        **kwargs,
    ):
        """Add a full-width node below the previous one; returns its centre."""
        y = self._cursor - height
        centre = box(
            self.ax, self.x + inset, y, self.width - 2 * inset, height,
            title, subtitle, kind, **kwargs,
        )
        self._cursor = y - gap
        return centre

    def add_pair(
        self,
        left: tuple[str, str, str],
        right: tuple[str, str, str],
        height: float = 0.068,
        gap: float = 0.030,
        inset: float = 0.016,
        **kwargs,
    ):
        """Add two side-by-side nodes; returns ``(left_centre, right_centre)``."""
        y = self._cursor - height
        usable = self.width - 2 * inset
        each = (usable - 0.012) / 2
        a = box(self.ax, self.x + inset, y, each, height, *left, **kwargs)
        b = box(self.ax, self.x + inset + each + 0.012, y, each, height, *right, **kwargs)
        self._cursor = y - gap
        return a, b

    def skip(self, amount: float) -> None:
        """Leave extra vertical space."""
        self._cursor -= amount


def new_canvas(width: float, height: float, title: str, subtitle: str):
    """Create a figure with a consistent title block."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.985, title, ha="center", va="top", fontsize=14, fontweight="bold", color=INK)
    ax.text(0.5, 0.951, subtitle, ha="center", va="top", fontsize=8.8, color=MUTED)
    return fig, ax


def save(fig, path: Path, dpi: int) -> Path:
    """Save and close a figure."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ============================================================ system diagram
# Vertical rhythm. FancyBboxPatch adds ~0.012 of padding on each side, so the
# gap between two nodes must exceed 0.024 or they visually touch.
GAP = 0.040


def draw_system_architecture(output: Path, dpi: int) -> Path:
    """Five-lane system architecture: data prep, knowledge, model, XAI, serving."""
    fig, ax = new_canvas(
        16.0, 10.5,
        "Context-Aware Explainable Sarcasm Detection \u2014 System Architecture",
        "RoBERTa \u00b7 conversational context \u00b7 retrieval-augmented commonsense reasoning \u00b7 "
        "cross-attention fusion \u00b7 LIME/SHAP explainability   (text-only)",
    )

    # ---------------------------------------------------------- lane 1: offline
    data_lane = Lane(ax, 0.015, 0.628, 0.270, 0.900,
                     "\u2460  OFFLINE DATA PREPARATION   (run once)")
    d1 = data_lane.add("Four sarcasm corpora",
                       "News-Headlines \u00b7 MUStARD \u00b7 SARC \u00b7 FigLang2020", "storage",
                       height=0.052, gap=GAP, title_size=9.0, subtitle_size=7.2)
    d2 = data_lane.add("Adapters \u2192 common schema",
                       "{id, text, context[\u22643], label, dataset, domain}", "input",
                       height=0.052, gap=GAP, title_size=9.0, subtitle_size=7.0)
    d3 = data_lane.add("Conversation-level split 70/15/15",
                       "stratified \u00b7 seeded \u00b7 leakage-verified", "input",
                       height=0.052, gap=GAP, title_size=9.0, subtitle_size=7.2)

    knowledge_lane = Lane(ax, 0.015, 0.245, 0.270, 0.517,
                          "\u2461  KNOWLEDGE INDEX   (run once)")
    k1 = knowledge_lane.add("ConceptNet 5.7 / ATOMIC / WordNet",
                            "relation tuples \u2192 natural-language facts", "storage",
                            height=0.052, gap=GAP, title_size=8.8, subtitle_size=7.2)
    k2 = knowledge_lane.add("Sentence-Transformer",
                            "all-MiniLM-L6-v2 \u00b7 384-d \u00b7 L2-normalised", "retrieval",
                            height=0.052, gap=GAP, title_size=9.0, subtitle_size=7.2)
    k3 = knowledge_lane.add("FAISS index (inner product)",
                            "flat < 200k facts \u00b7 IVF above", "retrieval",
                            height=0.052, gap=GAP, title_size=9.0, subtitle_size=7.2)

    for a, b in ((d1, d2), (d2, d3), (k1, k2), (k2, k3)):
        arrow(ax, (a[0], a[1] - 0.038), (b[0], b[1] + 0.038))

    # ------------------------------------------------------- lane 2: the model
    model_lane = Lane(ax, 0.320, 0.322, 0.352, 0.900,
                      "\u2462  MODEL   (trained \u2014 the project's contribution)")
    m_in = model_lane.add("Preprocessing",
                          "casing \u00b7 punctuation \u00b7 emoji \u00b7 hashtags preserved", "input",
                          height=0.046, gap=GAP, title_size=9.2, subtitle_size=7.2)
    m_u, m_c = model_lane.add_pair(
        ("Target utterance", "\u2264128 tokens", "input"),
        ("Context (\u22643 turns)", "turn-tagged, \u2264128 tok", "input"),
        height=0.056, gap=GAP, title_size=8.4, subtitle_size=7.0,
    )
    m_e = model_lane.add("Shared RoBERTa-base encoder",
                         "12 layers \u00b7 hidden 768 \u00b7 one set of weights", "encoder",
                         height=0.052, gap=GAP, title_size=9.4, subtitle_size=7.2)
    m_r = model_lane.add("Top-5 commonsense evidence",
                         "query = last context turn + utterance", "retrieval",
                         height=0.048, gap=GAP, title_size=9.2, subtitle_size=7.2)
    m_p = model_lane.add("Evidence projection 384 \u2192 768",
                         "Linear + LayerNorm", "retrieval",
                         height=0.044, gap=GAP, title_size=9.2, subtitle_size=7.2)
    m_f = model_lane.add("CROSS-ATTENTION FUSION",
                         "Q = utterance tokens\nK, V = context tokens \u2295 evidence \u2295 null slot",
                         "fusion", height=0.070, gap=GAP,
                         title_size=10.2, subtitle_size=7.2)
    m_h = model_lane.add("Classification head",
                         "concat[utt \u2016 ctx \u2016 attended] 2304 \u2192 256 \u2192 2", "head",
                         height=0.050, gap=GAP, title_size=9.4, subtitle_size=7.2)
    m_o = model_lane.add("Softmax", "Sarcastic / Not-Sarcastic + confidence", "output",
                         height=0.046, gap=0.020, title_size=9.4, subtitle_size=7.2)

    arrow(ax, (m_in[0], m_in[1] - 0.035), (m_u[0], m_u[1] + 0.040))
    arrow(ax, (m_in[0], m_in[1] - 0.035), (m_c[0], m_c[1] + 0.040))
    arrow(ax, (m_u[0], m_u[1] - 0.040), (m_u[0], m_e[1] + 0.038))
    arrow(ax, (m_c[0], m_c[1] - 0.040), (m_c[0], m_e[1] + 0.038))
    arrow(ax, (m_e[0], m_e[1] - 0.038), (m_r[0], m_r[1] + 0.036))
    arrow(ax, (m_r[0], m_r[1] - 0.036), (m_p[0], m_p[1] + 0.034))
    arrow(ax, (m_p[0], m_p[1] - 0.034), (m_f[0], m_f[1] + 0.047))
    arrow(ax, (m_f[0], m_f[1] - 0.047), (m_h[0], m_h[1] + 0.037))
    arrow(ax, (m_h[0], m_h[1] - 0.037), (m_o[0], m_o[1] + 0.035))

    # Skip connection: the pooled utterance and context vectors bypass fusion and
    # are concatenated straight into the head (this is what makes the head input
    # 3 x 768 = 2304). Routed down the lane's left gutter.
    gutter = 0.302
    arrow(ax, (0.336, m_e[1]), (gutter, m_e[1]), colour="#2f855a")
    ax.plot([gutter, gutter], [m_h[1], m_e[1]], color="#2f855a", linewidth=1.4, zorder=1)
    arrow(ax, (gutter, m_h[1]), (0.336, m_h[1]), colour="#2f855a")
    ax.text(0.290, (m_e[1] + m_h[1]) / 2, "pooled utterance & context vectors",
            ha="center", va="center", fontsize=6.4, color="#2f855a", rotation=90)

    # ------------------------------------------------- lane 3: explainability
    xai_lane = Lane(ax, 0.700, 0.500, 0.270, 0.900, "\u2463  EXPLAINABILITY")
    x1 = xai_lane.add("Token attribution",
                      "LIME / occlusion (fast) \u00b7 SHAP (audit)", "explain",
                      height=0.048, gap=GAP, title_size=9.2, subtitle_size=7.2)
    x2 = xai_lane.add("Fusion attention read-out",
                      "per-turn and per-fact influence", "explain",
                      height=0.048, gap=GAP, title_size=9.2, subtitle_size=7.2)
    x3 = xai_lane.add("Incongruity check",
                      "wording polarity vs situation polarity", "explain",
                      height=0.048, gap=GAP, title_size=9.2, subtitle_size=7.2)
    x4 = xai_lane.add("Evidence-grounded explanation",
                      "composed from this prediction's own\nattribution, attention and cited facts",
                      "explain", height=0.056, gap=GAP,
                      title_size=9.2, subtitle_size=7.0)
    x5 = xai_lane.add("Structured JSON response",
                      "prediction \u00b7 confidence \u00b7 tokens \u00b7 evidence", "output",
                      height=0.048, gap=0.020, title_size=9.2, subtitle_size=7.2)
    for a, b in ((x1, x2), (x2, x3), (x3, x4), (x4, x5)):
        arrow(ax, (a[0], a[1] - 0.036), (b[0], b[1] + 0.036))

    # ---------------------------------------------------- lane 4: application
    app_lane = Lane(ax, 0.700, 0.075, 0.270, 0.420, "\u2464  APPLICATION")
    a1 = app_lane.add("FastAPI backend",
                      "/health  /model/info  /predict  /explain  /retrieve", "output",
                      height=0.052, gap=GAP, title_size=9.2, subtitle_size=6.8)
    a2 = app_lane.add("React + Vite dashboard",
                      "verdict \u00b7 highlighted tokens \u00b7 evidence \u00b7 results", "output",
                      height=0.052, gap=GAP, title_size=9.2, subtitle_size=7.0)
    a3 = app_lane.add("Docker / cloud deployment",
                      "model + index mounted, never baked in", "storage",
                      height=0.052, gap=0.020, title_size=9.2, subtitle_size=7.0)
    arrow(ax, (a1[0], a1[1] - 0.038), (a2[0], a2[1] + 0.038))
    arrow(ax, (a2[0], a2[1] - 0.038), (a3[0], a3[1] + 0.038))

    # ----------------------------------------------------- cross-lane arrows
    arrow(ax, (0.285, d3[1]), (0.318, m_e[1] + 0.058), "train / eval", "#2b6cb0", rad=-0.18)
    arrow(ax, (0.285, k2[1]), (0.318, m_r[1] - 0.005), "top-5 facts", "#b7791f", rad=-0.14)
    arrow(ax, (0.660, m_h[1] + 0.006), (0.696, x1[1] - 0.030), "perturbation\nqueries",
          "#2c7a7b", rad=-0.26)
    arrow(ax, (0.660, m_f[1] + 0.020), (0.696, x2[1] + 0.020), "attention\nweights",
          "#6b46c1", rad=-0.14)
    arrow(ax, (0.660, m_o[1]), (0.696, x5[1] - 0.028), "prediction", "#4a5568", rad=0.22)
    arrow(ax, (x5[0], x5[1] - 0.036), (a1[0], a1[1] + 0.038), "", "#4a5568")

    # -------------------------------------------------------- research note
    ax.text(
        0.026, 0.155,
        "WHAT IS NEW HERE\n\n"
        "RoBERTa, FAISS, LIME, SHAP, ConceptNet and\n"
        "Sentence-Transformers are all prior work and are\n"
        "used as published.\n\n"
        "The contribution is their INTEGRATION: conversational\n"
        "context and retrieval-grounded commonsense fused by a\n"
        "TRAINABLE cross-attention layer whose evidence is\n"
        "carried through into a human-readable explanation \u2014\n"
        "plus the controlled experimental evaluation of that\n"
        "integration against three ablated baselines.",
        ha="left", va="center", fontsize=7.4, color=MUTED,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f7f9fb", edgecolor=LANE_EDGE),
    )

    legend = [
        mpatches.Patch(facecolor=COLOURS[k][0], edgecolor=COLOURS[k][1], label=v)
        for k, v in [
            ("input", "input / data"),
            ("encoder", "shared RoBERTa"),
            ("retrieval", "commonsense retrieval"),
            ("fusion", "trainable fusion (novel)"),
            ("head", "classifier"),
            ("explain", "explainability"),
            ("output", "output / serving"),
            ("storage", "external artefact"),
        ]
    ]
    ax.legend(
        handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.045),
        ncol=8, frameon=False, fontsize=7.6,
    )
    return save(fig, output, dpi)


# ====================================================== detailed model diagram
def draw_detailed_architecture(output: Path, dpi: int) -> Path:
    """Tensor-level view with explicit shapes, matching the proposal."""
    fig, ax = new_canvas(
        13.0, 12.0,
        "Detailed Model Architecture — tensor shapes",
        "B = batch · L = 128 utterance tokens · Lc = 128 context tokens · "
        "H = 768 (RoBERTa-base) · K = 5 retrieved facts · D = 384 (MiniLM)",
    )

    # ---- inputs
    t_u = box(ax, 0.045, 0.855, 0.255, 0.058,
              "Target tokens", "[B, L] = [B, 128]", "input")
    t_c = box(ax, 0.355, 0.855, 0.255, 0.058,
              "Context tokens", "[B, Lc] = [B, 128]", "input")
    t_k = box(ax, 0.665, 0.855, 0.290, 0.058,
              "Retrieval query", "utterance + last context turn", "retrieval")

    # ---- encoders / retrieval
    e_u = box(ax, 0.045, 0.752, 0.255, 0.062,
              "RoBERTa-base", "shared weights → [B, L, H]", "encoder")
    e_c = box(ax, 0.355, 0.752, 0.255, 0.062,
              "RoBERTa-base (same weights)", "→ [B, Lc, H]", "encoder")
    r_1 = box(ax, 0.665, 0.752, 0.290, 0.062,
              "Sentence-Transformer + FAISS", "top-5 facts → [B, K, D] = [B, 5, 384]",
              "retrieval")

    v_u = box(ax, 0.045, 0.655, 0.255, 0.055,
              "Utterance vector", "<s> pooled → [B, 768]", "encoder")
    v_c = box(ax, 0.355, 0.655, 0.255, 0.055,
              "Context vector", "[B, 768] (learned vector if absent)", "encoder",
              subtitle_size=7.0)
    v_k = box(ax, 0.665, 0.655, 0.290, 0.055,
              "Evidence projection", "Linear 384→768 + LayerNorm → [B, 5, 768]", "retrieval",
              subtitle_size=7.0)

    for a, b in ((t_u, e_u), (t_c, e_c), (t_k, r_1), (e_u, v_u), (e_c, v_c), (r_1, v_k)):
        arrow(ax, (a[0], a[1] - 0.030), (b[0], b[1] + 0.030))

    # ---- memory assembly
    mem = box(ax, 0.300, 0.548, 0.400, 0.062,
              "Key/Value memory  =  null ⊕ context tokens ⊕ evidence",
              "[B, 1 + Lc + K, 768] = [B, 134, 768]", "fusion", title_size=9.0)
    arrow(ax, (v_c[0], v_c[1] - 0.028), (0.430, mem[1] + 0.031), "[B, Lc, 768]", rad=-0.10)
    arrow(ax, (v_k[0], v_k[1] - 0.028), (0.600, mem[1] + 0.031), "[B, 5, 768]", rad=0.10)
    ax.text(0.030, 0.575,
            "the learned null slot keeps\nthe softmax well-defined when\n"
            "context AND evidence are absent",
            ha="left", va="center", fontsize=7.0, color="#6b46c1", style="italic")

    # ---- fusion
    fus = box(ax, 0.190, 0.400, 0.620, 0.098,
              "MULTI-HEAD CROSS-ATTENTION FUSION  (8 heads, pre-norm, residual + FFN)",
              "Q = utterance tokens [B, L, 768]      K, V = memory [B, 134, 768]\n"
              "attention [B, L, 134]  →  masked mean over query tokens  →  attended [B, 768]",
              "fusion", title_size=9.6, subtitle_size=7.6)
    arrow(ax, (v_u[0], v_u[1] - 0.028), (0.300, fus[1] + 0.049), "Q: [B, L, 768]", rad=-0.15)
    arrow(ax, (mem[0], mem[1] - 0.031), (0.560, fus[1] + 0.049), "K, V")

    # ---- concat + head
    cat = box(ax, 0.230, 0.300, 0.540, 0.058,
              "concat[ utterance ‖ context ‖ attended ]", "[B, 3 × 768] = [B, 2304]", "head")
    arrow(ax, (fus[0], fus[1] - 0.049), (cat[0], cat[1] + 0.030))
    arrow(ax, (0.100, v_u[1] - 0.028), (0.270, cat[1] + 0.030), "", "#2f855a", rad=-0.42)
    arrow(ax, (0.420, v_c[1] - 0.028), (0.400, cat[1] + 0.030), "", "#2f855a", rad=-0.30)

    h1 = box(ax, 0.230, 0.214, 0.540, 0.052,
             "LayerNorm → Linear 2304→256 → ReLU → Dropout(0.2) → Linear 256→2",
             "", "head", title_size=8.6)
    h2 = box(ax, 0.315, 0.140, 0.370, 0.048,
             "Softmax → [B, 2]", "P(sarcastic), P(not sarcastic)", "output")
    arrow(ax, (cat[0], cat[1] - 0.030), (h1[0], h1[1] + 0.027))
    arrow(ax, (h1[0], h1[1] - 0.027), (h2[0], h2[1] + 0.025))

    # ---- outputs
    o1 = box(ax, 0.045, 0.040, 0.255, 0.058,
             "Prediction + confidence", "argmax and max probability", "output")
    o2 = box(ax, 0.355, 0.040, 0.255, 0.058,
             "LIME / SHAP attribution", "over the utterance, evidence fixed", "explain")
    o3 = box(ax, 0.665, 0.040, 0.290, 0.058,
             "Evidence + attention read-out", "cited facts, per-turn influence", "explain")
    arrow(ax, (0.400, h2[1] - 0.026), (o1[0], o1[1] + 0.030), "", rad=0.22)
    arrow(ax, (h2[0], h2[1] - 0.026), (o2[0], o2[1] + 0.030))
    arrow(ax, (0.600, h2[1] - 0.026), (o3[0], o3[1] + 0.030), "", rad=-0.22)
    arrow(ax, (0.810, fus[1]), (0.955, fus[1]), "", "#6b46c1")
    ax.plot([0.955, 0.955], [fus[1], o3[1]], color="#6b46c1", linewidth=1.3, zorder=1)
    arrow(ax, (0.955, o3[1]), (0.955, o3[1] - 0.001), "", "#6b46c1")
    ax.text(0.968, (fus[1] + o3[1]) / 2, "attention weights",
            ha="center", va="center", fontsize=6.8, color="#6b46c1", rotation=90)

    ax.text(
        0.5, 0.010,
        "Ablation variants change only this diagram's shaded blocks: "
        "concat fusion replaces cross-attention; disabling context or evidence removes that "
        "branch from the memory and shrinks the concatenation accordingly.",
        ha="center", va="bottom", fontsize=7.4, color=MUTED, style="italic",
    )
    return save(fig, output, dpi)


# ========================================================== data pipeline
def draw_data_pipeline(output: Path, dpi: int) -> Path:
    """Corpus normalisation, deduplication, splitting and leakage verification."""
    fig, ax = new_canvas(
        14.0, 7.0,
        "Data Pipeline — heterogeneous corpora to a leakage-free split",
        "scripts/download_datasets.py → scripts/prepare_datasets.py",
    )

    sources = [
        ("News-Headlines", "28,619 rows · no context\nKaggle (manual)"),
        ("MUStARD (text)", "690 rows · speaker-tagged\nGitHub (auto)"),
        ("SARC", "up to 1.01M rows · 1 parent turn\nKaggle / Princeton (manual)"),
        ("FigLang2020", "9,400 rows · 2–3 turns\nGitHub (auto)"),
    ]
    xs = [0.030, 0.275, 0.520, 0.765]
    tops = []
    for x, (name, detail) in zip(xs, sources):
        tops.append(box(ax, x, 0.752, 0.205, 0.100, name, detail, "storage",
                        title_size=9.0, subtitle_size=6.9))

    adapters = box(ax, 0.130, 0.622, 0.740, 0.070,
                   "Dataset adapters + conservative preprocessing",
                   "URLs/mentions normalised · casing, punctuation, emoji and hashtags preserved",
                   "input", subtitle_size=7.2)
    for top in tops:
        arrow(ax, (top[0], top[1] - 0.050), (top[0], adapters[1] + 0.035))

    schema = box(ax, 0.180, 0.492, 0.640, 0.076,
                 "Common schema (JSONL)",
                 "{id, text, context[≤3], label∈{0,1}, dataset, domain, language, "
                 "conversation_id, metadata}", "input", subtitle_size=7.0)
    arrow(ax, (adapters[0], adapters[1] - 0.035), (schema[0], schema[1] + 0.038))

    dedup = box(ax, 0.180, 0.362, 0.300, 0.078,
                "Deduplication", "identical (context, text) removed;\n"
                "label-conflicting groups dropped", "input", subtitle_size=6.9)
    report = box(ax, 0.520, 0.362, 0.300, 0.078,
                 "Quality report", "class balance · missing values ·\ncontext availability",
                 "output", subtitle_size=6.9)
    arrow(ax, (0.330, schema[1] - 0.038), (dedup[0], dedup[1] + 0.039))
    arrow(ax, (0.670, schema[1] - 0.038), (report[0], report[1] + 0.039))

    cluster = box(ax, 0.180, 0.238, 0.640, 0.070,
                  "Conversation clustering (union-find over shared utterances)",
                  "two records that share any substantive utterance join the same cluster",
                  "fusion", subtitle_size=7.0)
    arrow(ax, (dedup[0], dedup[1] - 0.039), (0.330, cluster[1] + 0.035))

    split = box(ax, 0.180, 0.118, 0.300, 0.070,
                "Stratified 70 / 15 / 15", "by (dataset, majority label) · fixed seed",
                "encoder", subtitle_size=7.0)
    leak = box(ax, 0.520, 0.118, 0.300, 0.078,
               "Leakage verification", "shared conversations · duplicates ·\ncontext↔target overlap",
               "head", subtitle_size=6.9)
    arrow(ax, (0.330, cluster[1] - 0.035), (split[0], split[1] + 0.035))
    arrow(ax, (0.486, split[1]), (0.514, leak[1]), "")
    ax.text(0.500, split[1] + 0.052, "verify", ha="center", va="bottom", fontsize=7,
            color=MUTED)

    out = box(ax, 0.300, 0.020, 0.400, 0.062,
              "data/splits/{train,val,test}.txt + manifest.json",
              "corpus fingerprint pins the split to the exact corpus", "output",
              title_size=8.6, subtitle_size=7.0)
    arrow(ax, (leak[0], leak[1] - 0.039), (0.600, out[1] + 0.031), "", rad=0.18)
    return save(fig, output, dpi)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a POSIX exit code."""
    args = parse_args(argv)
    directory = ensure_dir(args.output_dir)
    print("Rendering architecture diagrams ...")
    draw_system_architecture(directory / "architecture.png", args.dpi)
    draw_detailed_architecture(directory / "detailed_model_architecture.png", args.dpi)
    draw_data_pipeline(directory / "data_pipeline.png", args.dpi)
    print("Done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
