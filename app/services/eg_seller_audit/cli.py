"""CLI entry point: python -m app.services.eg_seller_audit --seller-id <id>
--marketplace wb --out report.json"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import audit_seller
from .sampler import DEFAULT_SAMPLE_TARGET


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with --seller-id (required), --marketplace (default "wb"),
    --out (default None meaning stdout), --sample-size (default DEFAULT_SAMPLE_TARGET, type int)."""
    parser = argparse.ArgumentParser(
        description="Audit a seller on a marketplace."
    )
    parser.add_argument(
        "--seller-id",
        required=True,
        help="The seller ID to audit.",
    )
    parser.add_argument(
        "--marketplace",
        default="wb",
        help="The marketplace to audit (default: wb).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output file path (default: stdout).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_TARGET,
        help=f"Sample size for the audit (default: {DEFAULT_SAMPLE_TARGET}).",
    )
    return parser


def main() -> None:
    """Parse args, validate marketplace, run the audit_seller pipeline via asyncio.run,
    write the resulting AuditReport as indented JSON to --out (or stdout if --out is None)."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.marketplace != "wb":
        print(
            f"Phase 1: only wb supported (got marketplace={args.marketplace})",
            file=sys.stderr,
        )
        raise SystemExit(1)

    generated_at = datetime.now(timezone.utc).isoformat()
    report = asyncio.run(
        audit_seller(
            seller_id=args.seller_id,
            marketplace=args.marketplace,
            generated_at=generated_at,
            sample_size=args.sample_size,
        )
    )
    json_text = report.model_dump_json(indent=2)

    if args.out:
        Path(args.out).write_text(json_text, encoding="utf-8")
    else:
        print(json_text)


if __name__ == "__main__":
    main()
