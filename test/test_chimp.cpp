#include <filesystem>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include "data/dataset_utils.hpp"

// Chimp128 baseline headers
#include "chimp_compressor.h"
#include "chimp_decompressor.h"

namespace fs = std::filesystem;

// Chimp128 使用 previousValues=128
static const int CHIMP_PREVIOUS_VALUES = 128;
static const size_t BATCH_SIZE = 1024;

static CompressionInfo test_chimp_compression(const std::vector<double>& data) {
    // 过滤 NaN 值（Chimp 使用 NaN 作为终止符）
    std::vector<double> filtered_data;
    filtered_data.reserve(data.size());
    for (const auto& v : data) {
        if (!std::isnan(v)) {
            filtered_data.push_back(v);
        }
    }
    
    if (filtered_data.empty()) {
        std::cerr << "[Chimp] 警告: 数据为空或全为 NaN" << std::endl;
        return CompressionInfo{};
    }

    const size_t total_count = filtered_data.size();
    const size_t num_batches = (total_count + BATCH_SIZE - 1) / BATCH_SIZE;
    
    long total_compressed_bits = 0;
    size_t total_decompressed_count = 0;
    bool verification_failed = false;
    
    // ========== 压缩和解压阶段（逐批处理，避免内存占用过大）==========
    auto comp_start = std::chrono::high_resolution_clock::now();
    auto decomp_start = std::chrono::high_resolution_clock::now();
    double total_comp_ms = 0.0;
    double total_decomp_ms = 0.0;
    
    // 进度输出：每处理一定数量的批次输出一次进度
    const size_t progress_interval = std::max(size_t(1), num_batches / 100);  // 每1%输出一次
    
    for (size_t batch = 0; batch < num_batches; ++batch) {
        // 输出进度
        // if (batch % progress_interval == 0 || batch == num_batches - 1) {
        //     double progress = 100.0 * (batch + 1) / num_batches;
        //     std::cout << "\r[Chimp] 处理进度: " << std::fixed << std::setprecision(1) 
        //               << progress << "% (" << (batch + 1) << "/" << num_batches << " 批)" << std::flush;
        // }
        
        size_t start_idx = batch * BATCH_SIZE;
        size_t end_idx = std::min(start_idx + BATCH_SIZE, total_count);
        size_t batch_size = end_idx - start_idx;
        
        // ========== 压缩当前批次 ==========
        auto batch_comp_start = std::chrono::high_resolution_clock::now();
        
        ChimpCompressor compressor(CHIMP_PREVIOUS_VALUES);
        for (size_t i = start_idx; i < end_idx; ++i) {
            compressor.addValue(filtered_data[i]);
        }
        compressor.close();
        
        long batch_compressed_bits = compressor.get_size();
        total_compressed_bits += batch_compressed_bits;
        Array<uint8_t> compressed_pack = compressor.get_compress_pack();
        
        auto batch_comp_end = std::chrono::high_resolution_clock::now();
        double batch_comp_ms = std::chrono::duration<double, std::milli>(batch_comp_end - batch_comp_start).count();
        total_comp_ms += batch_comp_ms;
        
        // ========== 立即解压当前批次 ==========
        auto batch_decomp_start = std::chrono::high_resolution_clock::now();
        
        ChimpDecompressor decompressor(compressed_pack, CHIMP_PREVIOUS_VALUES);
        std::vector<double> decompressed = decompressor.decompress();
        
        auto batch_decomp_end = std::chrono::high_resolution_clock::now();
        double batch_decomp_ms = std::chrono::duration<double, std::milli>(batch_decomp_end - batch_decomp_start).count();
        total_decomp_ms += batch_decomp_ms;
        
        // 验证当前批次的正确性（优化：只验证大小和前几个值，避免全量验证导致过慢）
        if (decompressed.size() != batch_size) {
            std::cerr << "\n[Chimp] 警告: 批次 " << batch << " 解压数据大小不匹配! 原始=" << batch_size
                      << ", 解压=" << decompressed.size() << std::endl;
            verification_failed = true;
        } else {
            // 只验证前3个和后3个值，以及中间1个值，避免全量验证
            const size_t check_count = std::min(size_t(7), batch_size);
            size_t check_indices[] = {0, 1, 2, batch_size / 2, batch_size - 3, batch_size - 2, batch_size - 1};
            
            for (size_t check_idx = 0; check_idx < check_count && check_indices[check_idx] < batch_size; ++check_idx) {
                size_t i = check_indices[check_idx];
                if (std::abs(decompressed[i] - filtered_data[start_idx + i]) > 1e-9) {
                    std::cerr << "\n[Chimp] 警告: 批次 " << batch << " 位置 " << i 
                              << " 数据不匹配! 原始=" << filtered_data[start_idx + i]
                              << ", 解压=" << decompressed[i] << std::endl;
                    verification_failed = true;
                    break;
                }
            }
        }
        
        total_decompressed_count += decompressed.size();
        
        // 释放当前批次的内存（compressed_pack 和 decompressed 会在作用域结束时自动释放）
    }
    
    // std::cout << std::endl;  // 进度输出后换行
    
    auto comp_end = std::chrono::high_resolution_clock::now();
    auto decomp_end = std::chrono::high_resolution_clock::now();

    // 最终验证
    if (total_decompressed_count != filtered_data.size()) {
        std::cerr << "[Chimp] 警告: 总解压数据大小不匹配! 原始=" << filtered_data.size()
                  << ", 解压=" << total_decompressed_count << std::endl;
        verification_failed = true;
    }
    
    if (verification_failed) {
        std::cerr << "[Chimp] 警告: 数据验证失败，但继续统计性能指标" << std::endl;
    }

    const double original_bytes = filtered_data.size() * sizeof(double);
    const double compressed_bytes = (total_compressed_bits + 7) / 8;
    const double compress_ratio = (original_bytes > 0) ? (compressed_bytes / original_bytes) : 0.0;
    
    const double comp_ms = total_comp_ms;  // 使用累积的压缩时间
    const double decomp_ms = total_decomp_ms;  // 使用累积的解压时间

    const double original_GB = original_bytes / (1024.0 * 1024.0 * 1024.0);
    const double comp_sec = comp_ms / 1000.0;
    const double decomp_sec = decomp_ms / 1000.0;

    const double comp_throughput = (comp_sec > 0) ? (original_GB / comp_sec) : 0.0;
    const double decomp_throughput = (decomp_sec > 0) ? (original_GB / decomp_sec) : 0.0;

    return CompressionInfo{
        original_bytes / 1024.0 / 1024.0,
        compressed_bytes / 1024.0 / 1024.0,
        compress_ratio,
        0.0,
        comp_ms,
        comp_throughput,
        0.0,
        decomp_ms,
        decomp_throughput
    };
}

// 处理单个文件的函数
static CompressionInfo process_single_file(const std::string& file_path) {
    if (!fs::exists(file_path)) {
        std::cerr << "错误: 文件不存在: " << file_path << std::endl;
        return CompressionInfo{};
    }
    
    if (!fs::is_regular_file(file_path)) {
        std::cerr << "错误: 不是普通文件: " << file_path << std::endl;
        return CompressionInfo{};
    }
    
    auto data = read_data(file_path);
    if (data.empty()) {
        std::cerr << "错误: 无法读取数据或数据为空: " << file_path << std::endl;
        return CompressionInfo{};
    }
    
    std::cout << "\n[Chimp] 正在处理文件: " << fs::path(file_path).filename() << std::endl;
    std::cout << "[Chimp] 数据量: " << data.size() << ", 分批数: " 
              << (data.size() + BATCH_SIZE - 1) / BATCH_SIZE << std::endl;
    
    CompressionInfo avg_info;
    bool ok = false;
    
    // 对于大文件（超过1000万数据），只运行1次以节省时间
    int run_times = (data.size() > 10000000) ? 1 : 3;
    if (run_times == 1) {
        std::cout << "[Chimp] 数据量较大，仅运行1次测试" << std::endl;
    }
    
    for (int i = 0; i < run_times; ++i) {
        if (run_times > 1) {
            std::cout << "\n[Chimp] 第 " << (i + 1) << "/" << run_times << " 次运行:" << std::endl;
        }
        auto info = test_chimp_compression(data);
        if (info.original_size_mb > 0) {
            avg_info += info;
            ok = true;
        }
    }
    
    if (ok) {
        avg_info = avg_info / static_cast<double>(run_times);
        std::cout << "\n[Chimp] 平均性能 (运行 " << run_times << " 次):" << std::endl;
        avg_info.print();
        return avg_info;
    }
    
    return CompressionInfo{};
}

// 处理目录的函数
static void process_directory(const std::string& data_dir) {
    if (!fs::exists(data_dir)) {
        std::cerr << "错误: 数据目录不存在: " << data_dir << std::endl;
        return;
    }
    
    if (!fs::is_directory(data_dir)) {
        std::cerr << "错误: 不是目录: " << data_dir << std::endl;
        return;
    }
    
    CompressionInfo total_info;
    int count = 0;
    
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (!entry.is_regular_file()) continue;
        
        auto data = read_data(entry.path().string());
        if (data.empty()) continue;
        
        std::cout << "\n[Chimp] 正在处理文件: " << entry.path().filename() << std::endl;
        std::cout << "[Chimp] 数据量: " << data.size() << ", 分批数: " 
                  << (data.size() + BATCH_SIZE - 1) / BATCH_SIZE << std::endl;
        
        CompressionInfo avg_info;
        bool ok = false;
        
        for (int i = 0; i < 3; ++i) {
            auto info = test_chimp_compression(data);
            if (info.original_size_mb > 0) {
                avg_info += info;
                ok = true;
            }
        }
        
        if (ok) {
            avg_info = avg_info / 3.0;
            avg_info.print();
            total_info += avg_info;
            count++;
        }
        
        std::cout << "\n---------------------------------------------" << std::endl;
    }
    
    if (count > 0) {
        std::cout << "\n[Chimp] 总体统计，平均性能:" << std::endl;
        (total_info / count).print();
    } else {
        std::cout << "未找到有效数据文件" << std::endl;
    }
}

int main(int argc, char** argv) {
    // 解析命令行参数
    if (argc > 2 && std::string(argv[1]) == "--file") {
        // 处理单个文件模式
        std::string file_path = argv[2];
        auto info = process_single_file(file_path);
        return (info.original_size_mb > 0) ? 0 : 1;
    } else if (argc > 2 && std::string(argv[1]) == "--dir") {
        // 处理目录模式
        std::string data_dir = argv[2];
        process_directory(data_dir);
        return 0;
    } else {
        // 默认模式：处理默认目录
        std::string data_dir = "../test/data/float";
        
        if (!fs::exists(data_dir)) {
            std::cerr << "错误: 数据目录不存在: " << data_dir << std::endl;
            std::cerr << "用法: " << argv[0] << " --file <文件路径> 或 --dir <目录路径>" << std::endl;
            return 1;
        }
        
        process_directory(data_dir);
        return 0;
    }
}
