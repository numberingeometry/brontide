#!/usr/bin/env python3
"""Stage 7 runner — emergence labels (emerging/declining/saturating) from g-hat."""
from brontide import cli, emergence

if __name__ == "__main__":
    p = cli.sample_parser(__doc__)
    p.add_argument("--cutoff", default=None, help="forecast origin YYYY-MM (default: CORPUS_END)")
    p.add_argument("--model", default=None, help="roster model to rank by (default: CV-selected)")
    args = p.parse_args()
    out = emergence.run(cutoff=args.cutoff, model=args.model)
    print(f"emergence -> {out}")
