#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>  //translated comment
#include "MPC_12.h"

//translated comment
constexpr size_t N = 1024 * 1024;  //translated comment
constexpr int DIM = 1;            //translated comment
constexpr int TEST_RUNS = 5;      //translated comment


//translated comment
template<typename T>
struct MPC_Traits;

template<>
struct MPC_Traits<float> {
    using CompressedType = int;
    static constexpr auto DecompressedSize = MPC_float_decompressedSize;
    static constexpr auto Compress = MPC_float_compressMemory;
    static constexpr auto Decompress = MPC_float_decompressMemory;
    static constexpr auto CompressBound = MPC_float_compressBound;
};

template<>
struct MPC_Traits<double> {
    using CompressedType = long;
    static constexpr auto DecompressedSize = MPC_double_decompressedSize;
    static constexpr auto Compress = MPC_double_compressMemory;
    static constexpr auto Decompress = MPC_double_decompressMemory;
    static constexpr auto CompressBound = MPC_double_compressBound;
};


//translated comment
template<typename T>
std::vector<T> generate_data(size_t size) {
    std::vector<T> data(size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist(-100.0, 100.0);
    
    //translated comment
    for (size_t i = 0; i < size; ++i) {
        if (i % 2 == 0) {
            data[i] = dist(gen);
        } else {
            data[i] = 50.0 * std::sin(i * 0.1);
        }
    }
    return data;
}

//translated comment
template<typename T>
void test_mpc_compression() {
    using Traits = MPC_Traits<T>;
    
    //translated comment
    double total_compress_time = 0.0;
    double total_decompress_time = 0.0;
    size_t final_compressed_size = 0;

    auto original = generate_data<T>(N);
    
    //translated comment
    const int input_bytes = N * sizeof(T);
    const int max_compressed_bytes = Traits::CompressBound(input_bytes);
    std::vector<typename Traits::CompressedType> compressed(max_compressed_bytes / sizeof(typename Traits::CompressedType));

    //translated comment
    uint64_t kernel_time = 0;
    Traits::Compress(compressed.data(), 
                    reinterpret_cast<const typename Traits::CompressedType*>(original.data()),
                    input_bytes, DIM, &kernel_time);

    //translated comment
    for (int i = 0; i < TEST_RUNS; ++i) {
        //translated comment
        auto start = std::chrono::high_resolution_clock::now();
        int cmp_size = Traits::Compress(  //Traits::Compress
            compressed.data(),
            reinterpret_cast<const typename Traits::CompressedType*>(original.data()),
            input_bytes,
            DIM,
            &kernel_time
        );
        auto end = std::chrono::high_resolution_clock::now();
        total_compress_time += std::chrono::duration<double>(end - start).count();
        
        //translated comment
        std::vector<T> decompressed(N);
        int dec_size = Traits::DecompressedSize(compressed.data(), cmp_size);
        
        start = std::chrono::high_resolution_clock::now();
        Traits::Decompress(
            reinterpret_cast<typename Traits::CompressedType*>(decompressed.data()),
            compressed.data(),
            cmp_size,
            &kernel_time
        );
        end = std::chrono::high_resolution_clock::now();
        total_decompress_time += std::chrono::duration<double>(end - start).count();

        //translated comment
        ASSERT_EQ(dec_size, input_bytes) << "解压大小不匹配";
        for (size_t j = 0; j < N; ++j) {
            ASSERT_NEAR(original[j], decompressed[j], 1e-6) 
                << "数据不一致 @ 位置 " << j;
        }

        final_compressed_size = cmp_size;
    }

    //translated comment
    const double avg_compress_time = total_compress_time / TEST_RUNS;
    const double avg_decompress_time = total_decompress_time / TEST_RUNS;
    const double compression_ratio = (final_compressed_size * 1.0) / input_bytes;

    std::cout << "\n[MPC 性能报告 (" << typeid(T).name() << ")]"
              << "\n压缩率: " << std::fixed << std::setprecision(2) << compression_ratio * 100 << "%"
              << "\n平均压缩时间: " << avg_compress_time * 1000 << " ms"
              << "\n平均解压时间: " << avg_decompress_time * 1000 << " ms"
              << "\n压缩吞吐量: " << (input_bytes / (1024.0 * 1024.0 * 1024.0)) / avg_compress_time << " GB/s"
              << "\n解压吞吐量: " << (input_bytes / (1024.0 * 1024.0 * 1024.0)) / avg_decompress_time << " GB/s"
              << "\nGPU内核时间: " << kernel_time << " μs\n";
}

//translated comment
TEST(MPCCompressionTest, FloatCompression) {
    test_mpc_compression<float>();
}

TEST(MPCCompressionTest, DoubleCompression) {
    test_mpc_compression<double>();
}
//translated comment
TEST(MPCCompressionTest, ExtremeValues) {
    std::vector<double> data = {
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::epsilon()
    };

    //translated comment
    long compressed[1024];
    uint64_t kernel_time;
    int cmp_size = MPC_double_compressMemory(
        compressed, 
        reinterpret_cast<const long*>(data.data()),
        data.size() * sizeof(double),
        1,
        &kernel_time
    );

    //translated comment
    std::vector<double> decompressed(data.size());
    MPC_double_decompressMemory(
        reinterpret_cast<long*>(decompressed.data()),
        compressed,
        cmp_size,
        &kernel_time
    );

    //translated comment
    for (size_t i = 0; i < data.size(); ++i) {
        ASSERT_DOUBLE_EQ(data[i], decompressed[i]);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}