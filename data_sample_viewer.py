#!/usr/bin/env python3
"""
Data Sample Viewer for MoMLNIDS Dataset
Displays sample data and feature summaries from the dataset
"""

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import glob
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
import click


def find_data_files(data_path=".", pattern="*.parquet"):
    """Find data files grouped by dataset"""
    datasets = {}
    search_patterns = [
        f"{data_path}/**/{pattern}",
        f"{data_path}/{pattern}",
        f"./data/**/{pattern}",
        f"./dataset/**/{pattern}",
        f"./src/data/**/{pattern}",
    ]

    all_files = []
    for search_pattern in search_patterns:
        files = glob.glob(search_pattern, recursive=True)
        all_files.extend(files)

    # Group files by dataset (directory name)
    for file_path in all_files:
        path_obj = Path(file_path)
        # Extract dataset name from path
        dataset_name = None
        for part in path_obj.parts:
            if any(
                keyword in part.lower()
                for keyword in ["unsw", "cicids", "nsl", "kdd", "bot", "dos"]
            ):
                dataset_name = part
                break

        if not dataset_name:
            dataset_name = path_obj.parent.name

        if dataset_name not in datasets:
            datasets[dataset_name] = []
        datasets[dataset_name].append(file_path)

    return datasets


def load_sample_data(file_path, n_samples=5):
    """Load sample data from parquet file"""
    try:
        if file_path.endswith(".parquet"):
            df = pl.read_parquet(file_path)
        else:
            df = pl.read_csv(file_path)

        # Get sample data
        sample_df = df.head(n_samples)
        return df, sample_df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def analyze_features(df):
    """Analyze features in the dataset"""
    # Based on the codebase, features are in columns 4-43, label in column 43
    feature_cols = list(range(4, 43))

    if len(df.columns) > 43:
        features = df.select(pl.nth(feature_cols))
        labels = df.select(pl.nth(43))

        # Get statistics
        stats = features.describe()

        # Get unique label values
        unique_labels = labels.unique().to_numpy().flatten()

        # Get label counts - simplified approach
        try:
            # Convert to pandas for value_counts
            label_series = labels.to_pandas().iloc[:, 0]
            label_counts = label_series.value_counts().to_dict()
        except:
            label_counts = {}

        return {
            "n_features": len(feature_cols),
            "n_samples": len(df),
            "feature_names": features.columns,
            "label_column": df.columns[43],  # Store label column name
            "stats": stats,
            "unique_labels": unique_labels,
            "label_counts": label_counts,
            "all_columns": df.columns,
        }
    else:
        # Fallback for different data structure
        numeric_cols = [
            col
            for col in df.columns
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

        if numeric_cols:
            stats = df.select(numeric_cols).describe()
            return {
                "n_features": len(numeric_cols),
                "n_samples": len(df),
                "feature_names": numeric_cols,
                "label_column": df.columns[-1] if len(df.columns) > 0 else None,
                "stats": stats,
                "unique_labels": [],
                "label_counts": {},
                "all_columns": df.columns,
            }
        else:
            return None


def display_sample_data(sample_df, n_lines=1, n_features=10, dataset_name="Dataset"):
    """Display sample data in a formatted table"""
    console = Console()

    # Convert to pandas for better display
    sample_pd = sample_df.to_pandas()

    console.print(
        Panel.fit(
            f"📊 Sample Data from {dataset_name} ({n_lines} line{'s' if n_lines > 1 else ''})",
            style="bold blue",
        )
    )

    # Always include the target feature (last column or column 43)
    target_col_idx = 43 if len(sample_pd.columns) > 43 else len(sample_pd.columns) - 1

    # Select features to display + target
    if n_features >= len(sample_pd.columns):
        display_cols = list(range(len(sample_pd.columns)))
    else:
        # Show first n_features-1 features + target column
        feature_cols = list(range(min(n_features - 1, target_col_idx)))
        if target_col_idx not in feature_cols:
            feature_cols.append(target_col_idx)
        display_cols = feature_cols

    display_data = sample_pd.iloc[:n_lines, display_cols]

    table = Table(show_header=True, header_style="bold magenta")

    # Add columns with special styling for target
    for i, col in enumerate(display_data.columns):
        col_name = str(col)
        if i == len(display_data.columns) - 1 and display_cols[-1] == target_col_idx:
            table.add_column(f"🎯 {col_name} (Target)", style="bold red")
        else:
            table.add_column(col_name, style="dim")

    # Add rows
    for _, row in display_data.iterrows():
        table.add_row(*[str(val) for val in row])

    console.print(table)

    remaining_cols = len(sample_pd.columns) - len(display_cols)
    if remaining_cols > 0:
        console.print(f"... and {remaining_cols} more columns")

    console.print(
        f"📋 Column info: Showing {len(display_cols)} out of {len(sample_pd.columns)} total columns"
    )
    return display_data


def display_feature_summary(analysis, dataset_name="Dataset", n_features_stats=5):
    """Display feature summary"""
    console = Console()

    console.print(Panel.fit(f"📈 Feature Summary - {dataset_name}", style="bold green"))

    # Basic info
    info_table = Table(show_header=False)
    info_table.add_column("Metric", style="cyan")
    info_table.add_column("Value", style="yellow")

    info_table.add_row("Number of Features", str(analysis["n_features"]))
    info_table.add_row("Number of Samples", str(analysis["n_samples"]))
    info_table.add_row("Target Column", str(analysis.get("label_column", "Unknown")))
    info_table.add_row("Unique Labels", str(analysis["unique_labels"]))

    # Add label distribution if available
    if analysis["label_counts"]:
        label_dist = ", ".join(
            [f"{k}: {v}" for k, v in analysis["label_counts"].items()]
        )
        info_table.add_row("Label Distribution", label_dist)

    console.print(info_table)

    # Feature statistics
    if analysis["stats"] is not None:
        console.print(
            f"\n📊 Feature Statistics (showing first {n_features_stats} features):"
        )
        stats_pd = analysis["stats"].to_pandas()

        # Display statistics
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Statistic", style="cyan")

        # Add columns for features
        display_features = min(n_features_stats, len(analysis["feature_names"]))
        for i in range(display_features):
            col_name = (
                analysis["feature_names"][i]
                if i < len(analysis["feature_names"])
                else f"Feature_{i}"
            )
            stats_table.add_column(str(col_name), style="dim")

        for row_idx, row in stats_pd.iterrows():
            stat_name = str(row_idx)
            values = [
                f"{val:.4f}"
                if isinstance(val, (float, np.float64, np.float32))
                else str(val)
                for val in row.iloc[:display_features]
            ]
            stats_table.add_row(stat_name, *values)

        console.print(stats_table)

        remaining_features = len(analysis["feature_names"]) - display_features
        if remaining_features > 0:
            console.print(f"... and {remaining_features} more features")


@click.command()
@click.option("--path", "-p", default=".", help="Path to search for data files")
@click.option("--samples", "-s", default=5, help="Number of sample lines to display")
@click.option("--lines", "-l", default=1, help="Number of lines to display (1 or 5)")
@click.option("--summary", "-sum", is_flag=True, help="Show feature summary")
@click.option("--pattern", default="*.parquet", help="File pattern to search for")
@click.option(
    "--features", "-f", default=10, help="Number of features to display in sample data"
)
@click.option(
    "--stats-features",
    "-sf",
    default=5,
    help="Number of features to show in statistics",
)
@click.option("--all-datasets", "-a", is_flag=True, help="Show data from all datasets")
@click.option(
    "--dataset", "-d", default=None, help="Show data from specific dataset only"
)
def main(
    path,
    samples,
    lines,
    summary,
    pattern,
    features,
    stats_features,
    all_datasets,
    dataset,
):
    """
    Display sample data and feature summaries from the MoMLNIDS dataset.

    Examples:
    python data_sample_viewer.py --lines 1 --features 15          # Show 1 line with 15 features
    python data_sample_viewer.py --lines 5 --all-datasets         # Show 5 lines from all datasets
    python data_sample_viewer.py --summary --stats-features 10    # Show summary with 10 features in stats
    python data_sample_viewer.py --dataset NF-UNSW-NB15-v2 --summary  # Show specific dataset only
    """
    console = Console()

    console.print(
        Panel.fit("🔍 Enhanced MoMLNIDS Data Sample Viewer", style="bold blue")
    )

    # Find data files grouped by dataset
    datasets = find_data_files(path, pattern)

    if not datasets:
        console.print("❌ No data files found!")
        console.print(
            "Try specifying a different path with --path or pattern with --pattern"
        )
        return

    console.print(
        f"📁 Found {len(datasets)} dataset(s) with {sum(len(files) for files in datasets.values())} total files"
    )

    # Filter datasets if specific dataset requested
    if dataset:
        if dataset in datasets:
            datasets = {dataset: datasets[dataset]}
        else:
            console.print(f"❌ Dataset '{dataset}' not found!")
            console.print(f"Available datasets: {list(datasets.keys())}")
            return

    # Process each dataset
    for dataset_name, file_list in datasets.items():
        if not all_datasets and dataset is None:
            # If not showing all datasets and no specific dataset, show only first one
            if dataset_name != list(datasets.keys())[0]:
                continue

        console.print(f"\n{'=' * 60}")
        console.print(f"📊 Processing Dataset: {dataset_name}")
        console.print(f"📁 Files in dataset: {len(file_list)}")

        # Load first available file from this dataset
        first_file = file_list[0]
        console.print(f"📖 Loading sample from: {Path(first_file).name}")

        df, sample_df = load_sample_data(first_file, samples)

        if df is None:
            console.print(f"❌ Could not load data file from {dataset_name}")
            continue

        # Display sample data
        if lines > 0:
            display_sample_data(
                sample_df, min(lines, len(sample_df)), features, dataset_name
            )

        # Display feature summary
        if summary:
            analysis = analyze_features(df)
            if analysis:
                display_feature_summary(analysis, dataset_name, stats_features)
            else:
                console.print(f"❌ Could not analyze features for {dataset_name}")

        # Show available files in this dataset
        if len(file_list) > 1:
            console.print(f"\n📂 Other files in {dataset_name}:")
            for i, file_path in enumerate(file_list[1:6]):  # Show up to 5 more files
                console.print(f"   {i + 2}. {Path(file_path).name}")
            if len(file_list) > 6:
                console.print(f"   ... and {len(file_list) - 6} more files")

    console.print(f"\n{'=' * 60}")
    console.print("✨ Data viewing completed!")

    # Show available datasets summary
    console.print("\n📋 Available Datasets Summary:")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Dataset", style="cyan")
    summary_table.add_column("Files", style="yellow")

    for name, files in datasets.items():
        summary_table.add_row(name, str(len(files)))

    console.print(summary_table)


if __name__ == "__main__":
    main()
