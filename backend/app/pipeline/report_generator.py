"""
report_generator.py — PDF export for pipeline runs.

Two entry points:
  build_run_report_pdf(out, dataset_name, audit_events)   -> single-run deep dive
  build_comparison_report_pdf(rows, missing)               -> cross-dataset comparison

`out` is the full raw pipeline output dict (same shape returned by
orchestrator.run_pipeline / cached in api/pipeline.py's _CACHE).
`rows` is the list of per-run dicts built by api/pipeline.py's
_row_from_cache / _row_from_db helpers.

Design notes:
  - Uses reportlab (Platypus) for layout/tables, matplotlib (Agg backend,
    already a project dependency) purely to rasterise charts to PNG bytes
    that get embedded as reportlab Images. No interactive chart state is
    kept — everything is one-shot bytes-in, bytes-out.
  - Every value is read defensively (.get with fallback) because DB-sourced
    rows for runs created before this feature existed won't have every key
    (see api/pipeline.py _row_from_db) and audit/mining data is only ever
    present when the run is still in the in-process cache.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, ListFlowable, ListItem,
)

TEAL = colors.HexColor("#085041")
TEAL_LIGHT = colors.HexColor("#E6F2EF")
GRAY = colors.HexColor("#6B7280")
AMBER = colors.HexColor("#B45309")
RED = colors.HexColor("#D85A30")

_styles = getSampleStyleSheet()
_styles.add(ParagraphStyle("H1", parent=_styles["Heading1"], textColor=TEAL, spaceAfter=6))
_styles.add(ParagraphStyle("H2", parent=_styles["Heading2"], textColor=TEAL, spaceBefore=14, spaceAfter=6, fontSize=13))
_styles.add(ParagraphStyle("H3", parent=_styles["Heading3"], textColor=colors.HexColor("#374151"), spaceBefore=8, spaceAfter=4, fontSize=10.5))
_styles.add(ParagraphStyle("Body", parent=_styles["BodyText"], fontSize=9, leading=13))
_styles.add(ParagraphStyle("Small", parent=_styles["BodyText"], fontSize=7.5, leading=10, textColor=GRAY))
_styles.add(ParagraphStyle("Cover", parent=_styles["Title"], textColor=TEAL, fontSize=20, alignment=TA_LEFT))
_styles.add(ParagraphStyle("CoverSub", parent=_styles["Normal"], textColor=GRAY, fontSize=10, spaceAfter=2))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fmt(v: Any, default: str = "—") -> str:
    if v is None:
        return default
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _pct(v: Any, default: str = "—") -> str:
    if v is None:
        return default
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return default


def _base_table_style(header_bg=TEAL, header_fg=colors.white) -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), header_fg),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ])


def _kv_table_style() -> TableStyle:
    """For two-column key/value tables with NO header row (first label is a
    row, not a header) — distinct from _base_table_style which teal-shades row 0."""
    return TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), GRAY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5E7EB")),
    ])


def _chart_png(fig) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=160 * mm, height=68 * mm)


def _fig_to_base64(fig) -> str:
    """Same rasterisation as _chart_png, but as a data: URI for the HTML report."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _build_reward_curve_fig(rl: dict):
    curve = rl.get("reward_curve") or []
    if not curve:
        return None
    smoothed = rl.get("reward_curve_smoothed") or []
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.plot(range(1, len(curve) + 1), curve, color="#B9DCD3", linewidth=0.8, label="raw episode reward")
    if smoothed:
        ax.plot(range(1, len(smoothed) + 1), smoothed, color="#085041", linewidth=1.6, label="smoothed")
    ax.set_xlabel("episode", fontsize=8)
    ax.set_ylabel("reward", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _reward_curve_chart(rl: dict) -> Optional[Image]:
    fig = _build_reward_curve_fig(rl)
    return _chart_png(fig) if fig is not None else None


def _comparisons_table(comparisons: dict) -> Table:
    header = ["Baseline", "Mean Advantage", "Win Rate", "Cohen's d", "Rel. Improvement", "MAE", "Verdict"]
    data = [header]
    for name, c in (comparisons or {}).items():
        better = c.get("learned_better")
        data.append([
            name.replace("_", " ").title(),
            _fmt(c.get("mean_advantage")),
            _pct(c.get("win_rate")),
            _fmt(c.get("cohens_d")),
            f"{c.get('relative_improvement_pct', '—')}%" if c.get("relative_improvement_pct") is not None else "—",
            _fmt(c.get("mae_vs_learned")),
            "Learned wins" if better else "Baseline competitive",
        ])
    t = Table(data, colWidths=[65, 68, 55, 55, 75, 50, 90])
    style = _base_table_style()
    for i, (name, c) in enumerate(list((comparisons or {}).items()), start=1):
        if not c.get("learned_better"):
            style.add("TEXTCOLOR", (0, i), (-1, i), RED)
    t.setStyle(style)
    return t


def build_run_report_pdf(out: dict, dataset_name: str, audit_events: Optional[list[dict]] = None) -> bytes:
    """Full single-run report: every stage the pipeline executed, in order."""
    audit_events = audit_events or out.get("audit_log") or []
    log_summary = out.get("log_summary", {}) or {}
    mining = out.get("mining", {}) or {}
    features = out.get("features", {}) or {}
    rl = out.get("rl", {}) or {}
    agent = out.get("agent", {}) or {}

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             topMargin=18 * mm, bottomMargin=16 * mm,
                             leftMargin=16 * mm, rightMargin=16 * mm,
                             title="BPMN Agentic Orchestration — Run Report")
    story: list[Any] = []

    # ── Cover ──
    story.append(Paragraph("BPMN Agentic Orchestration Platform", _styles["Cover"]))
    story.append(Paragraph("Pipeline Run Report", _styles["CoverSub"]))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", color=TEAL, thickness=1))
    story.append(Spacer(1, 10))
    meta_rows = [
        ["Dataset", dataset_name or log_summary.get("dataset_id", "—")],
        ["Source file", log_summary.get("source_filename", "—")],
        ["Run ID", out.get("run_id", "—")],
        ["Content hash (lineage anchor)", out.get("content_hash", "—")],
        ["Report generated", _now()],
        ["Researcher", "Anindya Roy — University of Amsterdam, MBA in AI, Data & Analytics"],
    ]
    mt = Table(meta_rows, colWidths=[55 * mm, 115 * mm])
    mt.setStyle(_kv_table_style())
    story.append(mt)
    story.append(PageBreak())

    # ── 1. Dataset characterisation ──
    story.append(Paragraph("1. Dataset Characterisation", _styles["H1"]))
    char_rows = [
        ["Metric", "Value"],
        ["Cases (traces)", _fmt(log_summary.get("n_cases"))],
        ["Events (activity executions)", _fmt(log_summary.get("n_events"))],
        ["Unique activities", _fmt(log_summary.get("n_activities"))],
        ["Format", _fmt(log_summary.get("format"))],
        ["Timestamps available", "Yes — real durations" if log_summary.get("has_timestamps") else "No — sequence-based proxies used"],
        ["Trace attribute keys", ", ".join(log_summary.get("trace_attr_keys") or []) or "None detected"],
    ]
    ct = Table(char_rows, colWidths=[70 * mm, 100 * mm])
    ct.setStyle(_base_table_style())
    story.append(ct)
    if not log_summary.get("has_timestamps"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            "⚠ No timestamps detected in this log. Cycle time, waiting time, and throughput cannot be "
            "computed directly; time-dependent features fall back to documented sequence-based proxies.",
            _styles["Small"]))

    # ── 2. Process mining findings ──
    story.append(Paragraph("2. Process Mining Findings", _styles["H1"]))
    story.append(Paragraph(f"{len(mining.get('variants', []))} distinct process variants discovered. "
                            f"Bottleneck basis: {mining.get('bottleneck_basis', '—')}.", _styles["Body"]))

    if mining.get("bottlenecks"):
        story.append(Paragraph("Top bottleneck activities", _styles["H3"]))
        bn_rows = [["Activity", "Criticality Score"]]
        for b in mining["bottlenecks"][:5]:
            bn_rows.append([b.get("activity", "—"), _fmt(b.get("score"))])
        bt = Table(bn_rows, colWidths=[110 * mm, 60 * mm])
        bt.setStyle(_base_table_style())
        story.append(bt)

    conf = mining.get("conformance", {}) or {}
    if conf.get("has_labels"):
        story.append(Paragraph("Conformance (label-based)", _styles["H3"]))
        conf_rows = [
            ["Positive (conforming)", _fmt(conf.get("positive"))],
            ["Negative (non-conforming)", _fmt(conf.get("negative"))],
            ["Conformance rate", _pct(conf.get("conformance_rate"))],
        ]
        cft = Table(conf_rows, colWidths=[70 * mm, 100 * mm])
        cft.setStyle(_kv_table_style())
        story.append(cft)

    findings = mining.get("findings_text") or []
    if findings:
        story.append(Paragraph("Narrative findings (as embedded for agent retrieval)", _styles["H3"]))
        story.append(ListFlowable([ListItem(Paragraph(f, _styles["Body"])) for f in findings],
                                   bulletType="bullet", start="•"))

    # ── 3. Feature vector ──
    story.append(Paragraph("3. MDP State Vector", _styles["H1"]))
    values = features.get("values", {}) or {}
    provenance = features.get("provenance", {}) or {}
    fv_rows = [["Feature", "Value", "Provenance"]]
    for k, v in values.items():
        fv_rows.append([k.replace("_", " "), _fmt(v), provenance.get(k, "—")])
    fvt = Table(fv_rows, colWidths=[45 * mm, 20 * mm, 105 * mm])
    fvt.setStyle(_base_table_style())
    story.append(fvt)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Time-based features available: {'Yes' if features.get('time_based_available') else 'No'} · "
        f"Timestamp coverage: {_pct(features.get('timestamp_coverage'))}", _styles["Small"]))

    # ── 4. RL training + proof ──
    story.append(PageBreak())
    story.append(Paragraph("4. Reinforcement Learning — Training & Proof", _styles["H1"]))
    learned = rl.get("learned_eval", {}) or {}
    metrics = rl.get("metrics", {}) or {}
    training = metrics.get("training", {}) or {}
    train_rows = [
        ["Episodes", _fmt(rl.get("episodes"))],
        ["Learned policy — mean return (held out)", _fmt(learned.get("mean"))],
        ["Learned policy — std return", _fmt(learned.get("std"))],
        ["Q-table size (states visited)", _fmt(rl.get("q_table_size"))],
        ["Convergence episode (90% of final reward)", _fmt(training.get("convergence_episode"))],
        ["Policy entropy", _fmt(training.get("policy_entropy"))],
    ]
    trt = Table(train_rows, colWidths=[100 * mm, 70 * mm])
    trt.setStyle(_kv_table_style())
    story.append(trt)

    curve_img = _reward_curve_chart(rl)
    if curve_img:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Training reward curve", _styles["H3"]))
        story.append(curve_img)

    comparisons = (rl.get("proof") or {}).get("comparisons", {})
    if comparisons:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Learned policy vs baselines (held-out evaluation)", _styles["H3"]))
        story.append(_comparisons_table(comparisons))
        if not all(c.get("learned_better") for c in comparisons.values()):
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                "Research finding: at least one baseline is competitive with the learned policy on current "
                "dynamics. This is documented as an honest empirical result, not suppressed — see thesis "
                "discussion on near-myopic transition effects.", _styles["Small"]))

    # ── 5. Agent reasoning ──
    story.append(Paragraph("5. Agent Reasoning (LangGraph + Claude)", _styles["H1"]))
    if agent.get("summary"):
        story.append(Paragraph(agent["summary"], _styles["Body"]))
    if agent.get("reasoning"):
        story.append(Paragraph("Reasoning trace", _styles["H3"]))
        story.append(Paragraph(agent["reasoning"], _styles["Body"]))
    node_audit = agent.get("node_audit") or []
    if node_audit:
        story.append(Paragraph("Graph nodes executed", _styles["H3"]))
        na_rows = [["Node", "Status"]]
        for n in node_audit:
            na_rows.append([n.get("node", "—"), n.get("status", "ok")])
        nat = Table(na_rows, colWidths=[80 * mm, 90 * mm])
        nat.setStyle(_base_table_style())
        story.append(nat)

    # ── 6. Audit & lineage ──
    story.append(PageBreak())
    story.append(Paragraph("6. Audit Log & Lineage", _styles["H1"]))
    story.append(Paragraph(
        f"Every stage below was logged against the same run_id ({out.get('run_id', '—')}) and content "
        f"hash ({out.get('content_hash', '—')}), so every downstream artifact is traceable to the exact "
        f"bytes of the uploaded file.", _styles["Body"]))
    if audit_events:
        aud_rows = [["#", "Event", "Timestamp (UTC)", "Status"]]
        for e in audit_events:
            aud_rows.append([str(e.get("seq", "—")), e.get("event", "—"), e.get("ts", "—")[:19], e.get("status", "ok")])
        at = Table(aud_rows, colWidths=[10 * mm, 55 * mm, 60 * mm, 25 * mm])
        at.setStyle(_base_table_style())
        story.append(at)
    else:
        story.append(Paragraph("Audit log not available for this report (run may have been loaded from a "
                                "prior session).", _styles["Small"]))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#E5E7EB"), thickness=0.6))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Generated by the BPMN Agentic Orchestration Validation Platform — "
        "Anindya Roy, University of Amsterdam MBA in AI, Data & Analytics, 2026.", _styles["Small"]))

    doc.build(story)
    return buf.getvalue()


def _comparison_bar_chart(rows: list[dict], key: str, label: str, as_pct: bool = False) -> Optional[Image]:
    vals, names = [], []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        vals.append(v * 100 if as_pct else v)
        names.append((r.get("source_filename") or r.get("dataset_name") or r["run_id"])[:18])
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    bars = ax.bar(names, vals, color="#085041")
    ax.set_ylabel(label, fontsize=8)
    ax.tick_params(labelsize=7, axis="x", rotation=25)
    ax.tick_params(labelsize=7, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}" if not as_pct else f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=6.5)
    fig.tight_layout()
    return _chart_png(fig)


_HTML_CSS = """
:root { --teal:#085041; --teal-light:#E6F2EF; --gray:#6B7280; --red:#D85A30; --border:#E5E7EB; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
       color:#111827; max-width: 960px; margin: 0 auto; padding: 32px 24px 64px; line-height: 1.5; }
h1.cover { color: var(--teal); font-size: 26px; margin-bottom: 2px; }
.cover-sub { color: var(--gray); font-size: 14px; margin-bottom: 18px; }
h1 { color: var(--teal); font-size: 19px; margin-top: 40px; padding-top: 8px; border-top: 3px solid var(--teal); }
h1:first-of-type { border-top: none; margin-top: 24px; }
h3 { font-size: 13px; color: #374151; margin: 18px 0 6px; }
p { font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-size: 12.5px; }
th { background: var(--teal); color: white; text-align: left; padding: 8px 10px; font-weight: 600; }
td { padding: 6px 10px; border-bottom: 1px solid var(--border); }
tr:nth-child(even) td { background: #F9FAFB; }
table.kv td:first-child { font-weight: 600; color: var(--gray); width: 32%; }
table.kv tr:nth-child(even) td { background: #F9FAFB; }
.meta-table td { border-bottom: 1px solid var(--border); }
.badge { display:inline-block; padding:2px 8px; border-radius:6px; font-size:11px; font-family: monospace; }
.badge-warn { background:#FEF3C7; color:#B45309; }
.badge-bad  { background:#FEE2E2; color:#D85A30; }
.small { font-size: 11px; color: var(--gray); }
.verdict-bad { color: var(--red); font-weight: 600; }
.verdict-ok  { color: var(--teal); font-weight: 600; }
img.chart { max-width: 100%; margin: 6px 0 16px; }
ul.findings { font-size: 13px; padding-left: 20px; }
hr { border: none; border-top: 1px solid var(--border); margin: 32px 0 8px; }
.footer { color: var(--gray); font-size: 11px; }
"""


def _h(s: Any) -> str:
    """Minimal HTML-escape for values interpolated into the template."""
    if s is None:
        return "—"
    s = str(s)
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _html_table(headers: list[str], rows: list[list[str]], css_class: str = "") -> str:
    thead = "".join(f"<th>{_h(c)}</th>" for c in headers)
    trs = "".join("<tr>" + "".join(f"<td>{_h(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{css_class}"><thead><tr>{thead}</tr></thead><tbody>{trs}</tbody></table>'


def _html_kv_table(pairs: list[list[str]]) -> str:
    trs = "".join(f"<tr><td>{_h(k)}</td><td>{_h(v)}</td></tr>" for k, v in pairs)
    return f'<table class="kv"><tbody>{trs}</tbody></table>'


def build_run_report_html(out: dict, dataset_name: str, audit_events: Optional[list[dict]] = None) -> str:
    """Same content as build_run_report_pdf, as a single self-contained HTML
    file (inline CSS, charts embedded as base64 PNG) so it opens in any
    browser with no extra tooling — an alternative to the PDF export."""
    audit_events = audit_events or out.get("audit_log") or []
    log_summary = out.get("log_summary", {}) or {}
    mining = out.get("mining", {}) or {}
    features = out.get("features", {}) or {}
    rl = out.get("rl", {}) or {}
    agent = out.get("agent", {}) or {}

    parts: list[str] = []
    parts.append(f'<h1 class="cover">BPMN Agentic Orchestration Platform</h1>')
    parts.append('<div class="cover-sub">Pipeline Run Report</div>')
    parts.append(_html_kv_table([
        ["Dataset", dataset_name or log_summary.get("dataset_id", "—")],
        ["Source file", log_summary.get("source_filename")],
        ["Run ID", out.get("run_id")],
        ["Content hash (lineage anchor)", out.get("content_hash")],
        ["Report generated", _now()],
        ["Researcher", "Anindya Roy — University of Amsterdam, MBA in AI, Data & Analytics"],
    ]))

    # 1. Dataset characterisation
    parts.append("<h1>1. Dataset Characterisation</h1>")
    parts.append(_html_table(["Metric", "Value"], [
        ["Cases (traces)", _fmt(log_summary.get("n_cases"))],
        ["Events (activity executions)", _fmt(log_summary.get("n_events"))],
        ["Unique activities", _fmt(log_summary.get("n_activities"))],
        ["Format", _fmt(log_summary.get("format"))],
        ["Timestamps available", "Yes — real durations" if log_summary.get("has_timestamps") else "No — sequence-based proxies used"],
        ["Trace attribute keys", ", ".join(log_summary.get("trace_attr_keys") or []) or "None detected"],
    ]))
    if not log_summary.get("has_timestamps"):
        parts.append('<p class="small">⚠ No timestamps detected in this log. Cycle time, waiting time, and '
                     'throughput cannot be computed directly; time-dependent features fall back to documented '
                     'sequence-based proxies.</p>')

    # 2. Process mining findings
    parts.append("<h1>2. Process Mining Findings</h1>")
    parts.append(f'<p>{_h(len(mining.get("variants", [])))} distinct process variants discovered. '
                  f'Bottleneck basis: {_h(mining.get("bottleneck_basis", "—"))}.</p>')
    if mining.get("bottlenecks"):
        parts.append("<h3>Top bottleneck activities</h3>")
        parts.append(_html_table(["Activity", "Criticality Score"],
                                  [[b.get("activity", "—"), _fmt(b.get("score"))] for b in mining["bottlenecks"][:5]]))
    conf = mining.get("conformance", {}) or {}
    if conf.get("has_labels"):
        parts.append("<h3>Conformance (label-based)</h3>")
        parts.append(_html_kv_table([
            ["Positive (conforming)", _fmt(conf.get("positive"))],
            ["Negative (non-conforming)", _fmt(conf.get("negative"))],
            ["Conformance rate", _pct(conf.get("conformance_rate"))],
        ]))
    findings = mining.get("findings_text") or []
    if findings:
        parts.append("<h3>Narrative findings (as embedded for agent retrieval)</h3>")
        parts.append('<ul class="findings">' + "".join(f"<li>{_h(f)}</li>" for f in findings) + "</ul>")

    # 3. Feature vector
    parts.append("<h1>3. MDP State Vector</h1>")
    values = features.get("values", {}) or {}
    provenance = features.get("provenance", {}) or {}
    parts.append(_html_table(["Feature", "Value", "Provenance"],
                              [[k.replace("_", " "), _fmt(v), provenance.get(k, "—")] for k, v in values.items()]))
    parts.append(f'<p class="small">Time-based features available: '
                 f'{"Yes" if features.get("time_based_available") else "No"} · '
                 f'Timestamp coverage: {_pct(features.get("timestamp_coverage"))}</p>')

    # 4. RL training + proof
    parts.append("<h1>4. Reinforcement Learning — Training &amp; Proof</h1>")
    learned = rl.get("learned_eval", {}) or {}
    training = (rl.get("metrics", {}) or {}).get("training", {}) or {}
    parts.append(_html_kv_table([
        ["Episodes", _fmt(rl.get("episodes"))],
        ["Learned policy — mean return (held out)", _fmt(learned.get("mean"))],
        ["Learned policy — std return", _fmt(learned.get("std"))],
        ["Q-table size (states visited)", _fmt(rl.get("q_table_size"))],
        ["Convergence episode (90% of final reward)", _fmt(training.get("convergence_episode"))],
        ["Policy entropy", _fmt(training.get("policy_entropy"))],
    ]))
    fig = _build_reward_curve_fig(rl)
    if fig is not None:
        parts.append("<h3>Training reward curve</h3>")
        parts.append(f'<img class="chart" src="{_fig_to_base64(fig)}" alt="training reward curve"/>')

    comparisons = (rl.get("proof") or {}).get("comparisons", {})
    if comparisons:
        parts.append("<h3>Learned policy vs baselines (held-out evaluation)</h3>")
        rows = []
        for name, c in comparisons.items():
            better = c.get("learned_better")
            rows.append([
                name.replace("_", " ").title(),
                _fmt(c.get("mean_advantage")),
                _pct(c.get("win_rate")),
                _fmt(c.get("cohens_d")),
                f'{c.get("relative_improvement_pct", "—")}%' if c.get("relative_improvement_pct") is not None else "—",
                _fmt(c.get("mae_vs_learned")),
                f'<span class="{"verdict-ok" if better else "verdict-bad"}">'
                f'{"Learned wins" if better else "Baseline competitive"}</span>',
            ])
        table_html = _html_table(
            ["Baseline", "Mean Advantage", "Win Rate", "Cohen's d", "Rel. Improvement", "MAE", "Verdict"], [])
        # Build manually so the verdict span isn't HTML-escaped by _html_table.
        thead = "<tr>" + "".join(f"<th>{h}</th>" for h in
                 ["Baseline", "Mean Advantage", "Win Rate", "Cohen's d", "Rel. Improvement", "MAE", "Verdict"]) + "</tr>"
        trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        parts.append(f"<table><thead>{thead}</thead><tbody>{trs}</tbody></table>")
        if not all(c.get("learned_better") for c in comparisons.values()):
            parts.append('<p class="small">Research finding: at least one baseline is competitive with the '
                         'learned policy on current dynamics. This is documented as an honest empirical result, '
                         'not suppressed — see thesis discussion on near-myopic transition effects.</p>')

    # 5. Agent reasoning
    parts.append("<h1>5. Agent Reasoning (LangGraph + Claude)</h1>")
    if agent.get("summary"):
        parts.append(f"<p>{_h(agent['summary'])}</p>")
    if agent.get("reasoning"):
        parts.append("<h3>Reasoning trace</h3>")
        parts.append(f"<p>{_h(agent['reasoning'])}</p>")
    node_audit = agent.get("node_audit") or []
    if node_audit:
        parts.append("<h3>Graph nodes executed</h3>")
        parts.append(_html_table(["Node", "Status"], [[n.get("node", "—"), n.get("status", "ok")] for n in node_audit]))

    # 6. Audit & lineage
    parts.append("<h1>6. Audit Log &amp; Lineage</h1>")
    parts.append(f'<p>Every stage below was logged against the same run_id '
                 f'({_h(out.get("run_id"))}) and content hash ({_h(out.get("content_hash"))}), so every '
                 f'downstream artifact is traceable to the exact bytes of the uploaded file.</p>')
    if audit_events:
        parts.append(_html_table(["#", "Event", "Timestamp (UTC)", "Status"],
                                  [[e.get("seq", "—"), e.get("event", "—"), (e.get("ts") or "—")[:19], e.get("status", "ok")]
                                   for e in audit_events]))
    else:
        parts.append('<p class="small">Audit log not available for this report (run may have been loaded '
                     'from a prior session).</p>')

    parts.append("<hr/>")
    parts.append('<div class="footer">Generated by the BPMN Agentic Orchestration Validation Platform — '
                 'Anindya Roy, University of Amsterdam MBA in AI, Data &amp; Analytics, 2026.</div>')

    body = "\n".join(parts)
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'/>"
            f"<title>BPMN Run Report — {_h(out.get('content_hash'))}</title>"
            f"<style>{_HTML_CSS}</style></head><body>{body}</body></html>")


def build_comparison_report_pdf(rows: list[dict], missing: Optional[list[str]] = None) -> bytes:
    """Cross-dataset comparison report covering every run selected in the Compare page."""
    missing = missing or []
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             topMargin=18 * mm, bottomMargin=16 * mm,
                             leftMargin=14 * mm, rightMargin=14 * mm,
                             title="BPMN Agentic Orchestration — Dataset Comparison Report")
    story: list[Any] = []

    story.append(Paragraph("BPMN Agentic Orchestration Platform", _styles["Cover"]))
    story.append(Paragraph("Cross-Dataset Comparison Report", _styles["CoverSub"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated {_now()} · {len(rows)} run(s) compared", _styles["CoverSub"]))
    story.append(HRFlowable(width="100%", color=TEAL, thickness=1))
    story.append(Spacer(1, 10))

    if missing:
        story.append(Paragraph(
            f"Note: {len(missing)} selected run(s) could not be included — no cached or persisted result "
            f"was found (run_ids: {', '.join(missing)}). Open each run's results in the app once to "
            f"refresh its cached data.", _styles["Small"]))
        story.append(Spacer(1, 8))

    # ── Summary table ──
    story.append(Paragraph("Summary", _styles["H1"]))
    header = ["Dataset", "Cases", "Variants", "Conform.", "Bottleneck\nscore", "Rework\nprob.", "Learned\nmean return", "Best vs\ngreedy"]
    data = [header]
    for r in rows:
        greedy = (r.get("comparisons") or {}).get("greedy", {})
        data.append([
            (r.get("source_filename") or r.get("dataset_name") or r["run_id"])[:22],
            _fmt(r.get("n_cases")),
            _fmt(r.get("n_variants")),
            _pct(r.get("conformance_rate")),
            _fmt(r.get("bottleneck_score")),
            _fmt(r.get("rework_probability")),
            _fmt(r.get("learned_mean")),
            (f"+{greedy.get('relative_improvement_pct')}%" if greedy.get("learned_better")
             else (f"{greedy.get('relative_improvement_pct')}%" if greedy.get("relative_improvement_pct") is not None else "—")),
        ])
    st = Table(data, colWidths=[42 * mm, 16 * mm, 16 * mm, 16 * mm, 18 * mm, 16 * mm, 22 * mm, 20 * mm])
    st.setStyle(_base_table_style())
    story.append(st)
    story.append(Spacer(1, 4))
    incomplete = [r for r in rows if r.get("data_completeness") not in ("cache_full",)]
    if incomplete:
        story.append(Paragraph(
            "Rows marked with partial data were reconstructed from persisted database summaries "
            "(the full in-memory run cache had been recycled since that run executed); mining and "
            "reward-curve detail for those runs is limited to what was persisted.", _styles["Small"]))

    # ── Charts ──
    story.append(Paragraph("Charts", _styles["H1"]))
    for key, label, pct in [
        ("n_variants", "process variants", False),
        ("conformance_rate", "conformance rate", True),
        ("bottleneck_score", "bottleneck score", False),
        ("rework_probability", "rework probability", False),
        ("learned_mean", "learned mean return (held out)", False),
    ]:
        img = _comparison_bar_chart(rows, key, label, as_pct=pct)
        if img:
            story.append(Paragraph(label.title(), _styles["H3"]))
            story.append(img)
            story.append(Spacer(1, 4))

    # ── Per-run detail ──
    story.append(PageBreak())
    story.append(Paragraph("Per-Run Detail", _styles["H1"]))
    for r in rows:
        story.append(Paragraph(r.get("source_filename") or r.get("dataset_name") or r["run_id"], _styles["H2"]))
        detail_rows = [
            ["Dataset", r.get("dataset_name", "—")],
            ["Run ID", r.get("run_id", "—")],
            ["Content hash", r.get("content_hash", "—")],
            ["Cases / Events", f"{_fmt(r.get('n_cases'))} / {_fmt(r.get('n_events'))}"],
            ["Timestamps", "Yes" if r.get("has_timestamps") else "No — proxies used"],
            ["Top bottleneck", r.get("top_bottleneck", "—")],
            ["Episodes trained", _fmt(r.get("episodes"))],
            ["Data completeness", r.get("data_completeness", "—")],
        ]
        dtab = Table(detail_rows, colWidths=[45 * mm, 125 * mm])
        dtab.setStyle(_kv_table_style())
        story.append(dtab)
        comparisons = r.get("comparisons") or {}
        if comparisons:
            story.append(Spacer(1, 4))
            story.append(_comparisons_table(comparisons))
        story.append(Spacer(1, 10))

    story.append(HRFlowable(width="100%", color=colors.HexColor("#E5E7EB"), thickness=0.6))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Generated by the BPMN Agentic Orchestration Validation Platform — "
        "Anindya Roy, University of Amsterdam MBA in AI, Data & Analytics, 2026.", _styles["Small"]))

    doc.build(story)
    return buf.getvalue()
