"""
reporter.py — turns the raw results into something you actually want to look at

Generates a self-contained HTML report (no external CSS/JS) and a clean JSON
file. The HTML is deliberately simple so it renders everywhere and you can
print it or email it without drama.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML template (single-file, no external dependencies)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Plagiarism Detection Report</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#f5f7fa;color:#222;padding:2rem}}
    h1{{font-size:1.8rem;color:#1a1a2e;margin-bottom:.25rem}}
    .subtitle{{color:#555;font-size:.9rem;margin-bottom:2rem}}
    .summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin-bottom:2rem}}
    .card{{background:#fff;border-radius:10px;padding:1.2rem 1.5rem;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
    .card .label{{font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:#888}}
    .card .value{{font-size:2rem;font-weight:700;margin-top:.2rem}}
    .card.danger .value{{color:#e63946}}
    .card.safe .value{{color:#2a9d8f}}
    .card.info .value{{color:#457b9d}}
    table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;
           box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:2rem}}
    thead{{background:#1a1a2e;color:#fff}}
    th,td{{padding:.75rem 1rem;text-align:left;font-size:.88rem}}
    tr:not(:last-child) td{{border-bottom:1px solid #eee}}
    tbody tr:hover{{background:#f0f4ff}}
    .badge{{display:inline-block;padding:.2rem .6rem;border-radius:20px;font-size:.78rem;font-weight:600}}
    .badge-danger{{background:#fde8ea;color:#e63946}}
    .badge-safe{{background:#d8f3f0;color:#2a9d8f}}
    .bar-container{{width:140px;height:10px;background:#eee;border-radius:5px;display:inline-block;vertical-align:middle;margin-right:.5rem}}
    .bar{{height:100%;border-radius:5px}}
    .bar-danger{{background:#e63946}}
    .bar-warning{{background:#f4a261}}
    .bar-safe{{background:#2a9d8f}}
    .config-box{{background:#fff;border-radius:10px;padding:1.2rem 1.5rem;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:2rem}}
    .config-box h2{{font-size:1rem;color:#1a1a2e;margin-bottom:.8rem}}
    .config-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.5rem}}
    .config-item{{font-size:.83rem;color:#444}}
    .config-item strong{{color:#1a1a2e}}
    footer{{text-align:center;color:#aaa;font-size:.8rem;margin-top:2rem}}
  </style>
</head>
<body>
  <h1>📄 Plagiarism Detection Report</h1>
  <p class="subtitle">Generated on {generated_at} &nbsp;|&nbsp; Threshold: {threshold}</p>

  <div class="summary-grid">
    <div class="card info">
      <div class="label">Documents</div>
      <div class="value">{doc_count}</div>
    </div>
    <div class="card info">
      <div class="label">Pairs Checked</div>
      <div class="value">{pair_count}</div>
    </div>
    <div class="card {flag_class}">
      <div class="label">Flagged Pairs</div>
      <div class="value">{flagged_count}</div>
    </div>
    <div class="card info">
      <div class="label">Run Time</div>
      <div class="value" style="font-size:1.4rem">{elapsed}s</div>
    </div>
  </div>

  <div class="config-box">
    <h2>⚙️ Configuration</h2>
    <div class="config-grid">
      {config_items}
    </div>
  </div>

  <h2 style="margin-bottom:.75rem">📊 All Pair Results</h2>
  <table>
    <thead>
      <tr>
        <th>Document A</th>
        <th>Document B</th>
        <th>TF-IDF</th>
        <th>Fuzzy</th>
        <th>Semantic</th>
        <th>Composite</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>

  <footer>Plagiarism Detector — NLP Pipeline &nbsp;|&nbsp; github.com/your-username/plagiarism-detector</footer>
</body>
</html>"""


def _bar(score: float) -> str:
    pct = int(score * 100)
    if score >= 0.75:
        cls = "bar-danger"
    elif score >= 0.45:
        cls = "bar-warning"
    else:
        cls = "bar-safe"
    return (
        f'<div class="bar-container"><div class="bar {cls}" style="width:{pct}%"></div></div>'
        f'{pct}%'
    )


# ---------------------------------------------------------------------------
# Reporter class
# ---------------------------------------------------------------------------

class Reporter:
    """
    Generate HTML and/or JSON reports from a :class:`PipelineResult`.

    Parameters
    ----------
    output_dir : str or Path
        Directory where report files are saved (created if absent).
    """

    def __init__(self, output_dir: Union[str, Path] = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_html(self, result, filename: str = "report.html") -> Path:
        """
        Build and save an HTML report.

        Parameters
        ----------
        result : PipelineResult
        filename : str

        Returns
        -------
        Path to the saved report.
        """
        cfg = result.config
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        flag_class = "danger" if result.flagged_count > 0 else "safe"

        # Config summary
        config_items_html = "".join(
            f'<div class="config-item"><strong>{k}</strong>: {v}</div>'
            for k, v in {
                "Threshold": cfg.threshold,
                "N-gram range": cfg.ngram_range,
                "Max features": cfg.max_features,
                "Remove stop-words": cfg.remove_stopwords,
                "Lemmatize": cfg.lemmatize,
                "Semantic similarity": cfg.use_semantic,
                "Fuzzy method": cfg.fuzzy_method,
                "TF-IDF weight": round(cfg.weights.get("tfidf", 0), 3),
                "Fuzzy weight": round(cfg.weights.get("fuzzy", 0), 3),
                "Semantic weight": round(cfg.weights.get("semantic", 0), 3),
            }.items()
        )

        # Table rows
        rows_html = ""
        for r in result.similarity_results:
            badge = (
                '<span class="badge badge-danger">⚠ Plagiarism</span>'
                if r.is_plagiarism
                else '<span class="badge badge-safe">✔ Clear</span>'
            )
            rows_html += (
                f"<tr>"
                f"<td>{r.doc_a}</td>"
                f"<td>{r.doc_b}</td>"
                f"<td>{_bar(r.tfidf_score)}</td>"
                f"<td>{_bar(r.fuzzy_score)}</td>"
                f"<td>{_bar(r.semantic_score)}</td>"
                f"<td>{_bar(r.composite_score)}</td>"
                f"<td>{badge}</td>"
                f"</tr>\n"
            )

        html = _HTML_TEMPLATE.format(
            generated_at=now,
            threshold=cfg.threshold,
            doc_count=len(result.documents),
            pair_count=result.total_pairs,
            flagged_count=result.flagged_count,
            flag_class=flag_class,
            elapsed=round(result.elapsed_seconds, 2),
            config_items=config_items_html,
            rows=rows_html,
        )

        out_path = self.output_dir / filename
        out_path.write_text(html, encoding="utf-8")
        logger.info("HTML report saved → %s", out_path)
        return out_path

    def generate_json(self, result, filename: str = "report.json") -> Path:
        """
        Save a machine-readable JSON report.

        Returns
        -------
        Path to the saved report.
        """
        cfg = result.config
        payload = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "threshold": cfg.threshold,
                "ngram_range": list(cfg.ngram_range),
                "max_features": cfg.max_features,
                "remove_stopwords": cfg.remove_stopwords,
                "lemmatize": cfg.lemmatize,
                "remove_numbers": cfg.remove_numbers,
                "use_semantic": cfg.use_semantic,
                "weights": cfg.weights,
                "fuzzy_method": cfg.fuzzy_method,
            },
            "summary": {
                "documents_checked": len(result.documents),
                "total_pairs": result.total_pairs,
                "flagged_pairs": result.flagged_count,
                "elapsed_seconds": round(result.elapsed_seconds, 4),
            },
            "documents": result.documents,
            "results": [r.to_dict() for r in result.similarity_results],
        }

        out_path = self.output_dir / filename
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("JSON report saved → %s", out_path)
        return out_path

    def print_summary(self, result) -> None:
        """Print a concise summary table to stdout."""
        cfg = result.config
        sep = "-" * 80
        print(f"\n{'=' * 80}")
        print("  PLAGIARISM DETECTION REPORT")
        print(f"{'=' * 80}")
        print(f"  Documents   : {len(result.documents)}")
        print(f"  Pairs       : {result.total_pairs}")
        print(f"  Flagged     : {result.flagged_count}  (threshold={cfg.threshold})")
        print(f"  Elapsed     : {result.elapsed_seconds:.2f}s")
        print(sep)
        header = f"  {'Doc A':<25} {'Doc B':<25} {'Composite':>9}  {'TF-IDF':>7}  {'Fuzzy':>6}  Status"
        print(header)
        print(sep)
        for r in result.similarity_results:
            flag = "⚠ PLAGIARISM" if r.is_plagiarism else "  clear"
            print(
                f"  {r.doc_a:<25} {r.doc_b:<25} "
                f"{r.composite_score:>8.1%}  {r.tfidf_score:>6.1%}  "
                f"{r.fuzzy_score:>5.1%}  {flag}"
            )
        print(f"{'=' * 80}\n")
