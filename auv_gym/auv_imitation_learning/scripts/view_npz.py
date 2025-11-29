#!/usr/bin/env python3

import numpy as np
import argparse
import sys


def view_npz(npz_path, num_samples=5, detailed=False):
    """View contents of an NPZ file."""

    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"NPZ File: {npz_path}")
    print(f"{'='*60}\n")

    # List available arrays
    print(f"Available arrays: {list(data.keys())}\n")

    # Show shapes and dtypes
    print("Array Information:")
    print("-" * 60)
    for key in data.keys():
        arr = data[key]
        print(f"  {key}:")
        print(f"    Shape: {arr.shape}")
        print(f"    Dtype: {arr.dtype}")
        print(f"    Size:  {arr.size:,} elements ({arr.nbytes / 1024 / 1024:.2f} MB)")
    print()

    # Show statistics for each array
    print("Statistics:")
    print("-" * 60)
    for key in data.keys():
        arr = data[key]
        print(f"  {key}:")
        print(f"    Min:    {arr.min():.6f}")
        print(f"    Max:    {arr.max():.6f}")
        print(f"    Mean:   {arr.mean():.6f}")
        print(f"    Std:    {arr.std():.6f}")
        print(f"    Median: {np.median(arr):.6f}")
    print()

    # Show sample data
    if num_samples > 0:
        print(f"Sample Data (first {num_samples} samples):")
        print("-" * 60)
        for key in data.keys():
            arr = data[key]
            samples_to_show = min(num_samples, len(arr))
            print(f"\n  {key}:")
            if detailed:
                for i in range(samples_to_show):
                    print(f"    [{i}]: {arr[i]}")
            else:
                print(f"{arr[:samples_to_show]}")

    # Feature breakdown for inputs (if applicable)
    if "inputs" in data.keys() and detailed:
        print(f"\n{'='*60}")
        print("Input Feature Breakdown:")
        print("-" * 60)
        inputs = data["inputs"]
        feature_dim = inputs.shape[1]

        # Assuming structure: [target_features(4), velocity(6), orientation(3)]
        if feature_dim == 13:
            print("  Features [0-3]:  Target (x, y, z, yaw)")
            print("  Features [4-9]:  Velocity (vx, vy, vz, wx, wy, wz)")
            print("  Features [10-12]: Orientation (roll, pitch, yaw)")
            print()

            # Show statistics per feature group
            print("  Target stats:")
            print(
                f"    x:   min={inputs[:,0].min():.3f}, max={inputs[:,0].max():.3f}, mean={inputs[:,0].mean():.3f}"
            )
            print(
                f"    y:   min={inputs[:,1].min():.3f}, max={inputs[:,1].max():.3f}, mean={inputs[:,1].mean():.3f}"
            )
            print(
                f"    z:   min={inputs[:,2].min():.3f}, max={inputs[:,2].max():.3f}, mean={inputs[:,2].mean():.3f}"
            )
            print(
                f"    yaw: min={inputs[:,3].min():.3f}, max={inputs[:,3].max():.3f}, mean={inputs[:,3].mean():.3f}"
            )
            print()

            print("  Velocity stats:")
            print(
                f"    vx: min={inputs[:,4].min():.3f}, max={inputs[:,4].max():.3f}, mean={inputs[:,4].mean():.3f}"
            )
            print(
                f"    vy: min={inputs[:,5].min():.3f}, max={inputs[:,5].max():.3f}, mean={inputs[:,5].mean():.3f}"
            )
            print(
                f"    vz: min={inputs[:,6].min():.3f}, max={inputs[:,6].max():.3f}, mean={inputs[:,6].mean():.3f}"
            )
            print()

            print("  Orientation stats:")
            print(
                f"    roll:  min={inputs[:,10].min():.3f}, max={inputs[:,10].max():.3f}, mean={inputs[:,10].mean():.3f}"
            )
            print(
                f"    pitch: min={inputs[:,11].min():.3f}, max={inputs[:,11].max():.3f}, mean={inputs[:,11].mean():.3f}"
            )
            print(
                f"    yaw:   min={inputs[:,12].min():.3f}, max={inputs[:,12].max():.3f}, mean={inputs[:,12].mean():.3f}"
            )

    # Label breakdown (if applicable)
    if "labels" in data.keys() and detailed:
        print(f"\n{'='*60}")
        print("Label Feature Breakdown:")
        print("-" * 60)
        labels = data["labels"]

        if labels.shape[1] == 4:
            print("  Features: [linear.x, linear.y, linear.z, angular.z]")
            print()
            print(
                f"    linear.x:  min={labels[:,0].min():.3f}, max={labels[:,0].max():.3f}, mean={labels[:,0].mean():.3f}"
            )
            print(
                f"    linear.y:  min={labels[:,1].min():.3f}, max={labels[:,1].max():.3f}, mean={labels[:,1].mean():.3f}"
            )
            print(
                f"    linear.z:  min={labels[:,2].min():.3f}, max={labels[:,2].max():.3f}, mean={labels[:,2].mean():.3f}"
            )
            print(
                f"    angular.z: min={labels[:,3].min():.3f}, max={labels[:,3].max():.3f}, mean={labels[:,3].mean():.3f}"
            )

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View contents of NPZ files")
    parser.add_argument("npz_file", help="Path to NPZ file")
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to display (default: 5)",
    )
    parser.add_argument(
        "-d", "--detailed", action="store_true", help="Show detailed feature breakdown"
    )

    args = parser.parse_args()
    view_npz(args.npz_file, args.num_samples, args.detailed)
