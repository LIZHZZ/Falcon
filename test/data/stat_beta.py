#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute beta (number of significant digits) for each dataset.
- Number of data points
- Average beta
- Maximum beta

Beta definition (number of significant digits), for example:
- 3.14 -> beta = 3
- 0.0214 -> beta = 3 (first non-zero digit is 2, so 3 significant digits)
"""

import re
import csv
from pathlib import Path
from typing import List, Tuple, Optional
from statistics import mean

NUMBER_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
TRIM_CHARS = ' \t\r\n,;:[](){}"\'`'

def is_number_token(tok: str) -> bool:
    """Check whether a string is a valid number token."""
    s = tok.strip(TRIM_CHARS)
    if not s:
        return False
    low = s.lower()
    if low in ('nan', '+nan', '-nan', 'inf', '+inf', '-inf', 'infinity', '-infinity'):
        return False
    return bool(NUMBER_RE.match(s))

def count_beta(num_str: str) -> int:
    """
    Compute beta (number of significant digits) for a numeric string.
    Rules:
    - Strip leading sign
    - If there is a decimal point, count from first non-zero digit to end (including trailing zeros)
    - If there is no decimal point (integer), trailing zeros are not significant
    - Scientific notation: only consider the mantissa
    - All zeros ("0", "0.0", etc.) are treated as 1 significant digit
    """
    s = num_str.strip(TRIM_CHARS).lower()
    if s.startswith(('+', '-')):
        s = s[1:]
    
    # Exclude inf/NaN etc.
    if s in ('nan', 'inf'):
        return 0
    
    # Handle scientific notation
    if 'e' in s:
        mantissa = s.split('e', 1)[0]
    else:
        mantissa = s
    
    has_decimal = '.' in mantissa
    digits = [ch for ch in mantissa if ch.isdigit()]
    
    if not digits:
        return 0
    
    # Check if all digits are zero
    if all(d == '0' for d in digits):
        return 1
    
    # Remove leading zeros
    first_non_zero = 0
    while first_non_zero < len(digits) and digits[first_non_zero] == '0':
        first_non_zero += 1
    
    if first_non_zero >= len(digits):
        return 1
    
    if has_decimal:
        # With decimal point: from first non-zero to end, including trailing zeros
        return len(digits) - first_non_zero
    else:
        # Without decimal point: trailing zeros are not significant
        end = len(digits)
        while end > first_non_zero and digits[end - 1] == '0':
            end -= 1
        return max(1, end - first_non_zero)

def iter_number_tokens_from_file(path: Path) -> List[str]:
    """
    Extract all numeric tokens (as strings) from a CSV file.
    """
    tokens = []
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            # Try to read as CSV
            try:
                reader = csv.reader(f)
                for row in reader:
                    for cell in row:
                        cell = cell.strip()
                        if cell and is_number_token(cell):
                            tokens.append(cell)
            except:
                # Fallback: split by lines if not standard CSV
                f.seek(0)
                for line in f:
                    for raw in re.split(r'[,\s;]+', line.strip()):
                        tok = raw.strip(TRIM_CHARS)
                        if tok and is_number_token(tok):
                            tokens.append(tok)
    except Exception as e:
        print(f"[WARN] Failed to parse: {path} -> {e}")
    
    return tokens

def analyze_dataset(path: Path) -> Tuple[int, Optional[float], Optional[int]]:
    """
    Analyze a dataset file.
    Returns: (count, average beta, maximum beta)
    """
    tokens = iter_number_tokens_from_file(path)
    
    if not tokens:
        return 0, None, None
    
    betas = []
    for tok in tokens:
        beta = count_beta(tok)
        if beta > 0:  # Filter out invalid values
            betas.append(beta)
    
    if not betas:
        return len(tokens), None, None
    
    return (
        len(betas),
        round(mean(betas), 4),
        max(betas)
    )

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute beta statistics (significant digits) for each dataset."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories"
    )
    parser.add_argument(
        "--output",
        default="beta_stats.csv",
        help="Output CSV file name (default: beta_stats.csv)"
    )
    
    args = parser.parse_args()
    
    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: path does not exist or is not a directory: {root}")
        return
    
    # Find all CSV files
    files = []
    if args.recursive:
        files = list(root.rglob("*.csv"))
    else:
        files = list(root.glob("*.csv"))
    
    files.sort()
    
    if not files:
        print(f"No CSV files found in {root}")
        return
    
    print(f"Scanning path: {root}")
    print(f"Found {len(files)} CSV files\n")
    
    results = []
    for fp in files:
        count, avg_beta, max_beta = analyze_dataset(fp)
        dataset_name = fp.stem
        results.append({
            "dataset": dataset_name,
            "file_path": str(fp.relative_to(root)),
            "count": count,
            "avg_beta": avg_beta if avg_beta is not None else "",
            "max_beta": max_beta if max_beta is not None else ""
        })
        
        # Print progress
        print(f"Processing: {dataset_name}")
        print(f"  count: {count}")
        if avg_beta is not None:
            print(f"  avg_beta: {avg_beta:.4f}")
            print(f"  max_beta: {max_beta}")
        print()
    
    # Save as CSV
    output_path = Path(args.output)
    with output_path.open('w', encoding='utf-8-sig', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=["dataset", "file_path", "count", "avg_beta", "max_beta"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
    
    print(f"Statistics saved to: {output_path}")
    
    # Print summary table
    print("\nSummary:")
    print(f"{'dataset':<30} {'count':<12} {'avg_beta':<12} {'max_beta':<10}")
    print("-" * 70)
    for r in results:
        dataset = r["dataset"][:28]  # Truncate long names
        count = r["count"]
        avg_beta = f"{r['avg_beta']:.4f}" if r['avg_beta'] != "" else "N/A"
        max_beta = r["max_beta"] if r["max_beta"] != "" else "N/A"
        print(f"{dataset:<30} {count:<12} {avg_beta:<12} {max_beta:<10}")

if __name__ == "__main__":
    main()


