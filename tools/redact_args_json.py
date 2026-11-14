"""Redact plaintext API keys from existing args.json files under a results directory.

Dry run by default: lists files that contain secret-looking keys without modifying them.
Pass --apply to rewrite those files in place with the secret values replaced by "<redacted>".

Usage:
    PROJECT_DIR="." PYTHONPATH="code:${PYTHONPATH}" python tools/redact_args_json.py --results-dir results
    PROJECT_DIR="." PYTHONPATH="code:${PYTHONPATH}" python tools/redact_args_json.py --results-dir results --apply
"""
import argparse
import json
import logging
import os

from medscore_completeness.utils import REDACTED, redact_secrets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=os.path.join(os.environ.get("PROJECT_DIR", "."), "results"))
    parser.add_argument("--filename", default="args.json", help="File name to search for")
    parser.add_argument("--apply", action="store_true", help="Rewrite files in place (default: dry run)")
    return parser.parse_args()


def find_files(root, filename):
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            yield os.path.join(dirpath, filename)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    n_with_secrets = 0
    for path in sorted(find_files(args.results_dir, args.filename)):
        with open(path) as f:
            try:
                original = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Skipping unparseable file: {path}")
                continue
        redacted = redact_secrets(original)
        if redacted == original:
            continue
        n_with_secrets += 1
        if args.apply:
            with open(path, "w") as f:
                json.dump(redacted, f, indent=4)
            logger.info(f"Redacted: {path}")
        else:
            logger.info(f"Would redact: {path}")

    action = "Redacted" if args.apply else "Files with secrets (dry run)"
    logger.info(f"{action}: {n_with_secrets}. Replacement value: {REDACTED!r}")


if __name__ == "__main__":
    main()
