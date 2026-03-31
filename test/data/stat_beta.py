#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计每个数据集的beta值（有效位数）
- 数据个数
- 平均beta值
- 最大beta值

beta值定义：有效位数，例如：
- 3.14 的 beta = 3
- 0.0214 的 beta = 3（第一个非零数字是2，所以是3位有效数字）
"""

import re
import csv
from pathlib import Path
from typing import List, Tuple, Optional
from statistics import mean

NUMBER_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
TRIM_CHARS = ' \t\r\n,;:[](){}"\'`'

def is_number_token(tok: str) -> bool:
    """检查字符串是否为有效数字"""
    s = tok.strip(TRIM_CHARS)
    if not s:
        return False
    low = s.lower()
    if low in ('nan', '+nan', '-nan', 'inf', '+inf', '-inf', 'infinity', '-infinity'):
        return False
    return bool(NUMBER_RE.match(s))

def count_beta(num_str: str) -> int:
    """
    计算beta值（有效位数）
    规则：
    - 去掉正负号
    - 若字符串包含小数点，则从第一个非零数字开始到结尾全部算有效（包括尾随0）
    - 若不含小数点（纯整数），则尾随0不算有效
    - 科学计数法仅根据"尾数部分"判断
    - 全零（如 "0"、"0.0"）视为 1 位有效数字
    """
    s = num_str.strip(TRIM_CHARS).lower()
    if s.startswith(('+', '-')):
        s = s[1:]
    
    # 排除无穷/NaN等
    if s in ('nan', 'inf'):
        return 0
    
    # 处理科学计数法
    if 'e' in s:
        mantissa = s.split('e', 1)[0]
    else:
        mantissa = s
    
    has_decimal = '.' in mantissa
    digits = [ch for ch in mantissa if ch.isdigit()]
    
    if not digits:
        return 0
    
    # 检测是否全为0
    if all(d == '0' for d in digits):
        return 1
    
    # 去掉前导0
    first_non_zero = 0
    while first_non_zero < len(digits) and digits[first_non_zero] == '0':
        first_non_zero += 1
    
    if first_non_zero >= len(digits):
        return 1
    
    if has_decimal:
        # 有小数点：从第一个非零到结尾，全部算有效（包括尾随0）
        return len(digits) - first_non_zero
    else:
        # 无小数点：尾随0不算有效
        end = len(digits)
        while end > first_non_zero and digits[end - 1] == '0':
            end -= 1
        return max(1, end - first_non_zero)

def iter_number_tokens_from_file(path: Path) -> List[str]:
    """
    从CSV文件中提取所有数字token（字符串形式）
    """
    tokens = []
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            # 尝试作为CSV读取
            try:
                reader = csv.reader(f)
                for row in reader:
                    for cell in row:
                        cell = cell.strip()
                        if cell and is_number_token(cell):
                            tokens.append(cell)
            except:
                # 如果不是标准CSV，按行分割
                f.seek(0)
                for line in f:
                    for raw in re.split(r'[,\s;]+', line.strip()):
                        tok = raw.strip(TRIM_CHARS)
                        if tok and is_number_token(tok):
                            tokens.append(tok)
    except Exception as e:
        print(f"[WARN] 解析失败: {path} -> {e}")
    
    return tokens

def analyze_dataset(path: Path) -> Tuple[int, Optional[float], Optional[int]]:
    """
    分析数据集文件
    返回: (数据个数, 平均beta, 最大beta)
    """
    tokens = iter_number_tokens_from_file(path)
    
    if not tokens:
        return 0, None, None
    
    betas = []
    for tok in tokens:
        beta = count_beta(tok)
        if beta > 0:  # 排除无效值
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
        description="统计每个数据集的beta值（有效位数）"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="数据集文件夹路径"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录"
    )
    parser.add_argument(
        "--output",
        default="beta_stats.csv",
        help="输出CSV文件名（默认：beta_stats.csv）"
    )
    
    args = parser.parse_args()
    
    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"错误：路径不存在或不是文件夹: {root}")
        return
    
    # 查找所有CSV文件
    files = []
    if args.recursive:
        files = list(root.rglob("*.csv"))
    else:
        files = list(root.glob("*.csv"))
    
    files.sort()
    
    if not files:
        print(f"在 {root} 中未找到CSV文件")
        return
    
    print(f"扫描路径: {root}")
    print(f"找到 {len(files)} 个CSV文件\n")
    
    results = []
    for fp in files:
        count, avg_beta, max_beta = analyze_dataset(fp)
        dataset_name = fp.stem
        results.append({
            "数据集": dataset_name,
            "文件路径": str(fp.relative_to(root)),
            "数据个数": count,
            "平均beta": avg_beta if avg_beta is not None else "",
            "最大beta": max_beta if max_beta is not None else ""
        })
        
        # 打印进度
        print(f"处理: {dataset_name}")
        print(f"  数据个数: {count}")
        if avg_beta is not None:
            print(f"  平均beta: {avg_beta:.4f}")
            print(f"  最大beta: {max_beta}")
        print()
    
    # 保存为CSV
    output_path = Path(args.output)
    with output_path.open('w', encoding='utf-8-sig', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=["数据集", "文件路径", "数据个数", "平均beta", "最大beta"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
    
    print(f"统计结果已保存到: {output_path}")
    
    # 打印汇总表格
    print("\n汇总结果:")
    print(f"{'数据集':<30} {'数据个数':<12} {'平均beta':<12} {'最大beta':<10}")
    print("-" * 70)
    for r in results:
        dataset = r["数据集"][:28]  # 截断过长的名称
        count = r["数据个数"]
        avg_beta = f"{r['平均beta']:.4f}" if r['平均beta'] != "" else "N/A"
        max_beta = r["最大beta"] if r["最大beta"] != "" else "N/A"
        print(f"{dataset:<30} {count:<12} {avg_beta:<12} {max_beta:<10}")

if __name__ == "__main__":
    main()


