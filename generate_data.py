"""Script for generating synthetic polynomial datasets with noise."""

import argparse
import csv
import pathlib

import numpy as np


def generate_synthetic_data(
    num_samples: int,
    x_min: float,
    x_max: float,
    coefficients: list[float],
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic x and y data for y = beta_0 + beta_1*x + beta_2*x^2 + ... + noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, size=num_samples)
    x.sort()  # Sort x values for neat dataset ordering

    y = np.zeros(num_samples, dtype=np.float64)
    for degree, beta in enumerate(coefficients):
        y += beta * (x**degree)

    if noise_std > 0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=num_samples)
        y += noise

    return x, y


def save_to_csv(filepath: pathlib.Path, x: np.ndarray, y: np.ndarray) -> None:
    """Saves x and y arrays to CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y"])
        for xi, yi in zip(x, y):
            writer.writerow([f"{xi:.6f}", f"{yi:.6f}"])


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic polynomial data with Gaussian noise."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="synthetic_data.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--num-samples", "-n", type=int, default=200, help="Number of observations"
    )
    parser.add_argument("--x-min", type=float, default=-10.0, help="Minimum X value")
    parser.add_argument("--x-max", type=float, default=10.0, help="Maximum X value")
    parser.add_argument(
        "--coefficients",
        "-c",
        type=float,
        nargs="+",
        default=[1.0, -2.0, 0.5],
        help="Polynomial coefficients in ascending order: beta_0 beta_1 beta_2 ...",
    )
    parser.add_argument(
        "--noise-std", type=float, default=5.0, help="Gaussian noise standard deviation"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    x, y = generate_synthetic_data(
        num_samples=args.num_samples,
        x_min=args.x_min,
        x_max=args.x_max,
        coefficients=args.coefficients,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    out_path = pathlib.Path(args.output)
    save_to_csv(out_path, x, y)
    print(f"Generated {args.num_samples} synthetic samples saved to '{out_path}'.")
    print(f"Coefficients (beta_0, beta_1, ...): {args.coefficients}")
    print(f"Noise std: {args.noise_std}, Seed: {args.seed}")


if __name__ == "__main__":
    main()
