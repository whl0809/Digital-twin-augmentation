"""Command-line entry point: train a VAE and write synthetic patients.

Examples::

    python scripts/generate.py core35
    python scripts/generate.py mixed149 --epochs 300 --samples 200
    python scripts/generate.py all149 --no-plot --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dtaug import data, plotting, sampling, schema, train


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dtaug-generate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "schema", choices=sorted(schema.SCHEMAS), help="which dataset variant to fit"
    )
    parser.add_argument("--epochs", type=int, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, help="weight on the KL term")
    parser.add_argument("--samples", type=int, default=100, help="patients to generate")
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Gumbel-softmax temperature when sampling (mixed149 only)",
    )
    parser.add_argument(
        "--stop-below", type=float,
        help="stop early once the mean per-sample epoch loss drops below this",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed; -1 to skip")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, help="output CSV path")
    parser.add_argument("--figure", type=Path, help="output figure path")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    spec = schema.SCHEMAS[args.schema]
    prep = data.prepare(spec)
    print(
        f"{spec.name}: {spec.csv} -> {tuple(prep.X.shape)}, "
        f"{prep.continuous_dim} continuous + {len(prep.categorical_idx)} categorical "
        f"({prep.model_input_size} model inputs)"
    )

    common = dict(
        latent_dim=args.latent_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        stop_below=args.stop_below,
        device=args.device,
    )
    if spec.categorical:
        continuous, one_hots = prep.split()
        model = train.train_mixed_vae(
            continuous, one_hots,
            num_epochs=args.epochs or 300,
            kl_weight=1e-3 if args.kl_weight is None else args.kl_weight,
            **common,
        )
        fake = sampling.sample_mixed_vae(
            model, prep, args.samples, temperature=args.temperature
        )
    else:
        model = train.train_vae(
            prep.X,
            num_epochs=args.epochs or 600,
            kl_weight=1e-5 if args.kl_weight is None else args.kl_weight,
            **common,
        )
        fake = sampling.sample_vae(model, prep, args.samples)

    out = args.out or schema.GENERATED_DIR / f"{spec.name}_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fake.to_csv(out, index=False)
    print(f"wrote {len(fake)} synthetic patients to {out}")

    if not args.no_plot:
        figure = args.figure or schema.FIGURE_DIR / f"{spec.name}_marginals.png"
        plotting.plot_marginals(prep.raw, fake, figure)
        print(f"wrote marginal comparison to {figure}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
