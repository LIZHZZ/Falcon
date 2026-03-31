import re
import os
import csv
from pathlib import Path
from collections import defaultdict

import pandas as pd


DATA_PATTERNS = [
    # 匹配: [Gorilla] 正在处理文件: "SM(Sim-Memory).csv" 或 [Chimp] 正在处理文件: "SM(Sim-Memory).csv"
    re.compile(r'正在处理文件[:： ]+\s*"([^"]+\.csv[^\"]*)"'),
    # 匹配: 文件: ../test/data/use/AP(Air-pressure).csv (不带引号)
    re.compile(r'文件[:：]\s*([^\s]+\.csv[^\s]*)'),
    # 匹配: 文件: "path/to/file.csv" (带引号)
    re.compile(r'文件[:：]?\s*"([^"]+\.csv[^\"]*)"'),
    # 匹配: Processing file: path/to/file.csv
    re.compile(r'Processing file[:： ]+\s*(.+?\.csv[^\s]*)'),
]

METRIC_PATTERNS = {
    'compression_ratio': re.compile(r'压缩率:\s*([-\d\.]+)'),
    'kernel_compress_time': re.compile(r'压缩核函数时间:\s*([-\d\.]+)\s*ms'),
    'total_compress_time': re.compile(r'压缩总时间:\s*([-\d\.]+)\s*ms'),
    'kernel_decompress_time': re.compile(r'解压核函数时间:\s*([-\d\.]+)\s*ms'),
    'total_decompress_time': re.compile(r'解压总时间:\s*([-\d\.]+)\s*ms'),
    'compress_throughput': re.compile(r'压缩吞吐量:\s*([-\d\.]+)\s*GB/s'),
    'decompress_throughput': re.compile(r'解压吞吐量:\s*([-\d\.]+)\s*GB/s'),
}


def normalize_dataset(raw_path: str) -> str:
    dataset = raw_path.strip().strip('"')
    return os.path.basename(dataset)


def detect_dataset(line: str):
    for pattern in DATA_PATTERNS:
        match = pattern.search(line)
        if match:
            return normalize_dataset(match.group(1))
    return None


def parse_metrics(block: str):
    result = {}
    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(block)
        if not match:
            return None
        result[key] = float(match.group(1))
    return result


def parse_log_file(file_path: Path, display_method: str, data_store):
    with file_path.open('r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    current_dataset = None
    i = 0
    matched_blocks = 0

    while i < len(lines):
        line = lines[i]
        dataset = detect_dataset(line)
        if dataset:
            current_dataset = dataset
            i += 1
            continue

        if '压缩信息' in line and current_dataset:
            block_lines = [line]
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                # 检测新文件开始
                if detect_dataset(next_line):
                    break
                # 检测新文件开始的其他格式
                if 'Processing file' in next_line or '正在处理文件' in next_line or next_line.strip().startswith('文件:'):
                    break
                # 检测分隔线（可能表示新块开始）
                if next_line.strip().startswith('---') or next_line.strip().startswith('==='):
                    # 检查分隔线后是否还有数据，如果没有则停止
                    if j + 1 < len(lines) and (detect_dataset(lines[j + 1]) or '压缩信息' not in lines[j + 1]):
                        block_lines.append(next_line)
                        j += 1
                        continue
                    else:
                        break
                block_lines.append(next_line)
                j += 1

            block_text = ''.join(block_lines)
            metrics = parse_metrics(block_text)
            if metrics:
                matched_blocks += 1
                for metric_key, metric_value in metrics.items():
                    data_store[metric_key][display_method][current_dataset] = metric_value
            i = j
            continue

        i += 1

    return matched_blocks


def parse_log_files(log_dir: Path):
    data = {
        'compression_ratio': defaultdict(dict),
        'kernel_compress_time': defaultdict(dict),
        'total_compress_time': defaultdict(dict),
        'kernel_decompress_time': defaultdict(dict),
        'total_decompress_time': defaultdict(dict),
        'compress_throughput': defaultdict(dict),
        'decompress_throughput': defaultdict(dict),
    }

    method_map = {
        "gpu": "FALCON-g",
        "gpu_nopack": "FALCON-g-nopack",
        "gpu_br": "FALCON-g-br",
        "gpu_spare": "FALCON-g-spare",
        "muti_3step_block": "三阶段阻塞",
        "muti_3step_noblock": "三阶段非阻塞",
        "muti_stream": "四级流水线",
        "Falcon_cpu": "FALCON-cpu",

        "ALP": "CPU:ALP",
        "ALP_GPU": "GPU:ALP",
        "ALP_GPU_v2": "GPU:ALP_v2",
        "ALP_g": "GPU:ALP_g",

        "elf": "CPU:elf",
        "elf_star": "CPU:elf_star",
        "elf_star_g": "GPU:elf_star_g",
        "ndzip": "GPU:ndzip",
        "bitcomp": "GPU:bitcomp",
        "LZ4": "GPU:LZ4",
        "gdeflate": "GPU:gdeflate",
        "Snappy": "GPU:Snappy",

        "chimp":"CPU::chimp",
        "patas":"CPU::patas",
        "gorilla":"CPU::gorilla",

    }

    if not log_dir.exists():
        raise FileNotFoundError(f"日志目录不存在: {log_dir}")

    parsed_any = False

    for log_file in sorted(log_dir.glob('output_*.log')):
        method = log_file.name.replace('output_', '').replace('.log', '')
        display_method = method_map.get(method)
        if not display_method:
            continue

        matched = parse_log_file(log_file, display_method, data)
        if matched == 0:
            print(f"[WARN] 日志 {log_file.name} 未解析到任何数据块，可能格式不匹配。")
        else:
            parsed_any = True

    if not parsed_any:
        raise ValueError("未能从任何日志中解析到数据，请检查日志格式或路径配置。")

    compression_ratios = data['compression_ratio']
    avg_ratio_per_method = {}
    for method_name, dataset_dict in compression_ratios.items():
        values = list(dataset_dict.values())
        if values:
            avg_ratio_per_method[method_name] = sum(values) / len(values)
        else:
            avg_ratio_per_method[method_name] = 0

    print("不同方法的平均压缩率:")
    for method, avg_ratio in avg_ratio_per_method.items():
        print(f"方法 '{method}': {avg_ratio:.4f}")
    return data
    
    # 2. 计算每种方法下压缩率的平均值
    # 创建一个空字典来存储结果
    avg_ratio_per_method = {}

    # 定位到存储压缩率的字典
    compression_ratios = data['compression_ratio']

    # 遍历每种方法 (method_name) 及其对应的数据集字典 (dataset_dict)
    for method_name, dataset_dict in compression_ratios.items():
        
        # 获取该方法下所有数据集的压缩率值
        # dataset_dict.values() 会返回一个包含所有压缩率值的视图
        values = list(dataset_dict.values())
        
        # 安全检查：确保列表不为空
        if values:
            # 计算平均值
            average_ratio = sum(values) / len(values)
            # 存储结果
            avg_ratio_per_method[method_name] = average_ratio
        else:
            avg_ratio_per_method[method_name] = 0

    # 3. 打印最终结果
    print("不同方法的平均压缩率:")
    for method, avg_ratio in avg_ratio_per_method.items():
        print(f"方法 '{method}': {avg_ratio:.4f}")
    return data

# def create_combined_csv_report(data, output_file):
#     # 定义数据集顺序（从实际数据中获取所有唯一数据集）
#     all_datasets = set()
#     for metric in data.values():
#         for method_data in metric.values():
#             all_datasets.update(method_data.keys())
#     datasets = sorted(all_datasets)
    
#     # 定义方法顺序
#     methods = ['CPU:ALP', 'nvcomp::bitcomp', 'elf', 'nvcomp::gdf', 
#                'nvcomp::LZ4', 'ndzip', 'nvcomp::Snappy']
    
#     # 定义指标顺序和显示名称
#     metrics = [
#         ('compression_ratio', '压缩率实验'),
#         ('kernel_compress_time', '压缩时间（核函数）'),
#         ('kernel_decompress_time', '核函数解压时间'),
#         ('total_compress_time', '压缩时间'),
#         ('total_decompress_time', '解压时间'),
#         ('compress_throughput', '压缩吞吐量'),
#         ('decompress_throughput', '解压吞吐量')
#     ]
    
#     # 创建DataFrame列表
#     dfs = []
    
#     # 为每种指标创建表格
#     for metric_key, metric_name in metrics:
#         # 创建DataFrame
#         df = pd.DataFrame(index=methods, columns=datasets)
        
#         # 填充数据
#         for method in methods:
#             for dataset in datasets:
#                 if method in data[metric_key] and dataset in data[metric_key][method]:
#                     df.at[method, dataset] = data[metric_key][method][dataset]
        
#         # 添加标题行
#         df.columns.name = '数据集名称'
#         df.index.name = metric_name
        
#         # 添加到列表
#         dfs.append(df)
    
#     # 合并所有DataFrame
#     combined_df = pd.concat(dfs)
    
#     # 保存为CSV
#     combined_df.to_csv(output_file, encoding='utf-8-sig')
#     print(f"整合报告已生成: {output_file}")

def create_combined_csv_report(data, output_file):
    # 定义数据集顺序（从实际数据中获取所有唯一数据集）
    all_datasets = set()
    for metric in data.values():
        for method_data in metric.values():
            all_datasets.update(method_data.keys())
    datasets = sorted(all_datasets)
    
    # 定义方法顺序
    methods = [
        "FALCON-g",
        "FALCON-g-nopack",
        "FALCON-g-spare",
        "FALCON-g-br",
        "三阶段阻塞",
        "三阶段非阻塞",
        "四级流水线",
        "FALCON-cpu",

        "CPU:ALP",
        "GPU:ALP",
        "GPU:ALP_v2",
        "GPU:ALP_g",

        "CPU:elf",
        "CPU:elf_star",
        "GPU:elf_star_g",
        
        "CPU::chimp",
        "CPU::patas",
        "CPU::gorilla",

        "GPU:ndzip",
        "GPU:LZ4",
        "GPU:Snappy",
        "GPU:gdeflate",
        "GPU:bitcomp",
    ]
    
    # 定义指标顺序和显示名称
    metrics = [
        ('compression_ratio', '压缩率实验'),
        ('kernel_compress_time', '核函数压缩时间'),
        ('kernel_decompress_time', '核函数解压时间'),
        ('total_compress_time', '压缩时间'),
        ('total_decompress_time', '解压时间'),
        ('compress_throughput', '压缩吞吐量'),
        ('decompress_throughput', '解压吞吐量')
    ]
    
    # 准备写入CSV的行数据
    csv_rows = []
    
    # 为每种指标创建表格
    for metric_key, metric_name in metrics:
        # 添加指标标题行
        csv_rows.append([f"=== {metric_name} ==="])
        csv_rows.append([""] + datasets)  # 列标题行
        
        # 添加数据行
        for method in methods:
            row = [method]
            for dataset in datasets:
                value = ""
                if method in data[metric_key] and dataset in data[metric_key][method]:
                    value = data[metric_key][method][dataset]
                row.append(str(value) if value != "" else "")
            csv_rows.append(row)
        
        # 添加空行分隔
        csv_rows.append([])
    
    # 写入CSV文件
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    print(f"整合报告已生成: {output_file}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    log_directory = script_dir  # 默认使用脚本所在目录
    output_csv = script_dir / 'compression_combined_report_all_old.csv'

    parsed_data = parse_log_files(log_directory)
    create_combined_csv_report(parsed_data, str(output_csv))