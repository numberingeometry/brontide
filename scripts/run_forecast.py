#!/usr/bin/env python3
"""Stage 6 runner — statsforecast roster, 24m horizon, rolling-origin CV."""
from lacuna import cli, forecast

if __name__ == "__main__":
    p = cli.sample_parser(__doc__)
    p.add_argument("--cutoff", default=None, help="forecast origin YYYY-MM (default: latest)")
    args = p.parse_args()
    out = forecast.run(sample=args.sample, cutoff=args.cutoff)
    print(f"forecast -> {out}")
