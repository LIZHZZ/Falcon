import re
import os
import csv
from pathlib import Path
from collections import defaultdict

import pandas as pd


DATA_PATTERNS = [
    #: [Gorilla] : "SM(Sim-Memory).csv" [Chimp] : "SM(Sim-Memory).csv"
    re.compile(r'正在处理文件[:： ]+\s*"([^"]+\.csv[^\"]*)"'),
    #: : ../test/data/use/AP(Air-pressure).csv ( )
    re.compile(r'文件[:：]\s*([^\s]+\.csv[^\s]*)'),
    #: : "path/to/file.csv" ( )
    re.compile(r'文件[:：]?\s*"([^"]+\.csv[^\"]*)"'),
    #: Processing file: path/to/file.csv
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
                #translated comment
                if detect_dataset(next_line):
                    break
                #translated comment
                if 'Processing file' in next_line or '正在处理文件' in next_line or next_line.strip().startswith('文件:'):
                    break
                #translated comment
                if next_line.strip().startswith('---') or next_line.strip().startswith('==='):
                    #translated comment
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
    
    #translated comment
    #translated comment
    avg_ratio_per_method = {}

    #translated comment
    compression_ratios = data['compression_ratio']

    #(method_name) (dataset_dict)
    for method_name, dataset_dict in compression_ratios.items():
        
        #translated comment
        #dataset_dict.values()
        values = list(dataset_dict.values())
        
        #translated comment
        if values:
            #translated comment
            average_ratio = sum(values) / len(values)
            #translated comment
            avg_ratio_per_method[method_name] = average_ratio
        else:
            avg_ratio_per_method[method_name] = 0

    #translated comment
    print("不同方法的平均压缩率:")
    for method, avg_ratio in avg_ratio_per_method.items():
        print(f"方法 '{method}': {avg_ratio:.4f}")
    return data

# def create_combined_csv_report(data, output_file):
#translated comment
#     all_datasets = set()
#     for metric in data.values():
#         for method_data in metric.values():
#             all_datasets.update(method_data.keys())
#     datasets = sorted(all_datasets)
    
#translated comment
#     methods = ['CPU:ALP', 'nvcomp::bitcomp', 'elf', 'nvcomp::gdf', 
#                'nvcomp::LZ4', 'ndzip', 'nvcomp::Snappy']
    
#translated comment
#     metrics = [
#('compression_ratio', ' '),
#('kernel_compress_time', ' （ ）'),
#('kernel_decompress_time', ' '),
#('total_compress_time', ' '),
#('total_decompress_time', ' '),
#('compress_throughput', ' '),
#('decompress_throughput', ' ')
#     ]
    
## DataFrame
#     dfs = []
    
#translated comment
#     for metric_key, metric_name in metrics:
## DataFrame
#         df = pd.DataFrame(index=methods, columns=datasets)
        
#translated comment
#         for method in methods:
#             for dataset in datasets:
#                 if method in data[metric_key] and dataset in data[metric_key][method]:
#                     df.at[method, dataset] = data[metric_key][method][dataset]
        
#translated comment
#df.columns.name = ' '
#         df.index.name = metric_name
        
#translated comment
#         dfs.append(df)
    
## DataFrame
#     combined_df = pd.concat(dfs)
    
## CSV
#     combined_df.to_csv(output_file, encoding='utf-8-sig')
#print(f" : {output_file}")

def create_combined_csv_report(data, output_file):
    #translated comment
    all_datasets = set()
    for metric in data.values():
        for method_data in metric.values():
            all_datasets.update(method_data.keys())
    datasets = sorted(all_datasets)
    
    #translated comment
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
    
    #translated comment
    metrics = [
        ('compression_ratio', '压缩率实验'),
        ('kernel_compress_time', '核函数压缩时间'),
        ('kernel_decompress_time', '核函数解压时间'),
        ('total_compress_time', '压缩时间'),
        ('total_decompress_time', '解压时间'),
        ('compress_throughput', '压缩吞吐量'),
        ('decompress_throughput', '解压吞吐量')
    ]
    
    #CSV
    csv_rows = []
    
    #translated comment
    for metric_key, metric_name in metrics:
        #translated comment
        csv_rows.append([f"=== {metric_name} ==="])
        csv_rows.append([""] + datasets)  #translated comment
        
        #translated comment
        for method in methods:
            row = [method]
            for dataset in datasets:
                value = ""
                if method in data[metric_key] and dataset in data[metric_key][method]:
                    value = data[metric_key][method][dataset]
                row.append(str(value) if value != "" else "")
            csv_rows.append(row)
        
        #translated comment
        csv_rows.append([])
    
    #CSV
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    print(f"整合报告已生成: {output_file}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    log_directory = script_dir  #translated comment
    output_csv = script_dir / 'compression_combined_report_all_old.csv'

    parsed_data = parse_log_files(log_directory)
    create_combined_csv_report(parsed_data, str(output_csv))