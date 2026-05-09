#!/usr/bin/env python3
"""
cli.py — the friendly command-line front door to the plagiarism detector

I use this all the time from the terminal. Here are the patterns that come up most:

    # Quick check on a folder of essays
    python cli.py --dir sample_documents/

    # Specific files only
    python cli.py --files essay1.txt essay2.txt essay3.txt

    # Tweak the sensitivity and turn on semantic matching
    python cli.py --dir docs/ --threshold 0.65 --ngram-max 3 --semantic

    # Load everything from a config file (then override a couple things)
    python cli.py --dir docs/ --config config.yaml --threshold 0.70

    # Save reports somewhere else
    python cli.py --dir docs/ --output-dir my_reports/

    # Quiet mode for CI or scripts (just the exit code matters)
    python cli.py --dir docs/ --quiet
"""

import argparse
import logging
import sys
from pathlib import Path

# Let people run `python cli.py` straight from the project root without installing
sys.path.insert(0, str(Path(__file__).parent))

from plagiarism_detector import (
    PlagiarismDetector,
    DetectorConfig,
    Reporter,
)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plagiarism-detector",
        description="Catch copied and paraphrased text with a bit of NLP magic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input ---
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "--dir", "-d",
        metavar="DIRECTORY",
        help="Folder full of .txt files you want to scan.",
    )
    inp.add_argument(
        "--files", "-f",
        nargs="+",
        metavar="FILE",
        help="Specific files you want to compare against each other.",
    )

    # --- Config file (optional) ---
    p.add_argument(
        "--config", "-c",
        metavar="YAML",
        help="Path to a YAML config file.  CLI flags override YAML values.",
    )

    # --- Detection options ---
    p.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Composite similarity threshold for flagging plagiarism.",
    )
    p.add_argument(
        "--ngram-min",
        type=int,
        default=None,
        help="Minimum n-gram size.",
    )
    p.add_argument(
        "--ngram-max",
        type=int,
        default=None,
        help="Maximum n-gram size.",
    )
    p.add_argument(
        "--no-stopwords",
        action="store_true",
        help="Disable stop-word removal.",
    )
    p.add_argument(
        "--no-lemmatize",
        action="store_true",
        help="Disable lemmatization.",
    )
    p.add_argument(
        "--semantic",
        action="store_true",
        help="Enable spaCy semantic similarity (requires en_core_web_md).",
    )
    p.add_argument(
        "--fuzzy-method",
        choices=["ratio", "partial_ratio", "token_sort_ratio"],
        default=None,
        help="rapidfuzz scorer.",
    )
    p.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern when scanning a directory.",
    )
    p.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recurse into sub-directories.",
    )

    # --- Output ---
    p.add_argument(
        "--output-dir", "-o",
        default="reports",
        metavar="DIR",
        help="Directory to save HTML/JSON reports.",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation.",
    )
    p.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON report generation.",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose logging.",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Build config ---
    if args.config:
        config = DetectorConfig.from_yaml(args.config)
    else:
        config = DetectorConfig()

    # CLI overrides
    if args.threshold is not None:
        config.threshold = args.threshold
    if args.ngram_min is not None or args.ngram_max is not None:
        lo = args.ngram_min or config.ngram_range[0]
        hi = args.ngram_max or config.ngram_range[1]
        config.ngram_range = (lo, hi)
    if args.no_stopwords:
        config.remove_stopwords = False
    if args.no_lemmatize:
        config.lemmatize = False
    if args.semantic:
        config.use_semantic = True
    if args.fuzzy_method:
        config.fuzzy_method = args.fuzzy_method

    # --- Run pipeline ---
    detector = PlagiarismDetector(config)

    try:
        if args.dir:
            result = detector.check_directory(
                args.dir,
                pattern=args.pattern,
                recursive=args.recursive,
            )
        else:
            result = detector.check_files(args.files)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    # --- Reports ---
    reporter = Reporter(output_dir=args.output_dir)
    reporter.print_summary(result)

    if not args.no_html:
        html_path = reporter.generate_html(result)
        print(f"HTML report → {html_path}")

    if not args.no_json:
        json_path = reporter.generate_json(result)
        print(f"JSON report → {json_path}")

    # Exit code: 1 if any plagiarism detected (useful in CI pipelines)
    return 1 if result.flagged_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
