#!/bin/bash

# 启用调试模式（可选，显示每条执行的命令）
set -x
cd ..


mkdir -p build
cd build || exit 1

# 编译项目
cmake .. || { echo "CMake failed!"; exit 1; }
make -j || { echo "Make failed!"; exit 1; }

# 运行测试并同时输出到屏幕和日志文件
run_test() {
    local test_name=$1
    local log_file="../script/output_${test_name}.log"
    echo "===== Running ${test_name} ====="
    # ./test/test_${test_name} --dir ../../../../dataset/public_bi_benchmark/stable_datasets_cleaned/ 2>&1 | tee "$log_file"
    # ./test/test_${test_name} --dir ../test/data/wrong/ 2>&1 | tee "$log_file"
    ./test/test_${test_name} --dir ../test/data/use/ 2>&1 | tee "$log_file"
}
# 执行所有测试

# run_test "Falcon_cpu"
# run_test "gpu"
# run_test "gpu_nopack"
# run_test "gpu_br"
# run_test "gpu_spare"
# run_test "muti_3step_block"
# run_test "muti_3step_noblock"
run_test "muti_stream"


# run_test "ALP"
# run_test "ALP_GPU"
# run_test "ALP_GPU_v2"
# run_test "ALP_g"

# run_test "elf"
# run_test "elf_star"
# run_test "elf_star_g"


# run_test "ndzip"
# run_test "bitcomp"
# run_test "LZ4"
# run_test "gdeflate"-
# run_test "Snappy"


echo "All tests completed! Logs saved to output_*.log files."
cd ../script/
python3 get.py