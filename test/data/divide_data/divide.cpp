#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

class EnhancedCSVSplitter {
private:
    std::string inputFilename;
    std::vector<double> targetSizesInMB;
    std::string headerLine;
    std::string outputDirectory;
    char delimiter;
    size_t totalColumns;
    size_t totalRows;
    
public:
    /**
     *translated comment
     *@param filename
     *@param startSizeMB （MB）
     *@param endSizeMB （MB）
     *@param stepSizeMB （MB）
     *@param delim ，
     */
    EnhancedCSVSplitter(const std::string& filename, 
                       double startSizeMB = 30.0, 
                       double endSizeMB = 300.0, 
                       double stepSizeMB = 30.0,
                       char delim = ',') 
        : inputFilename(filename), delimiter(delim), totalColumns(0), totalRows(0) {
        
        //translated comment
        generateTargetSizes(startSizeMB, endSizeMB, stepSizeMB);
        
        //translated comment
        createOutputDirectory();
    }
    
    /**
     *translated comment
     *@param filename
     *@param customSizes （MB）
     *@param delim ，
     */
    EnhancedCSVSplitter(const std::string& filename, 
                       const std::vector<double>& customSizes,
                       char delim = ',') 
        : inputFilename(filename), targetSizesInMB(customSizes), delimiter(delim), totalColumns(0), totalRows(0) {
        
        //translated comment
        createOutputDirectory();
    }
    
private:
    /**
     *translated comment
     */
    void generateTargetSizes(double startSizeMB, double endSizeMB, double stepSizeMB) {
        if (stepSizeMB <= 0) {
            std::cerr << "错误: 步长必须大于0" << std::endl;
            return;
        }
        
        if (startSizeMB > endSizeMB) {
            std::cerr << "错误: 起始大小不能大于结束大小" << std::endl;
            return;
        }
        
        targetSizesInMB.clear();
        for (double size = startSizeMB; size <= endSizeMB + 1e-9; size += stepSizeMB) {
            targetSizesInMB.push_back(size);
        }
        
        std::cout << "生成目标大小列表: ";
        for (size_t i = 0; i < targetSizesInMB.size(); ++i) {
            std::cout << std::fixed << std::setprecision(1) << targetSizesInMB[i] << "MB";
            if (i < targetSizesInMB.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    /**
     *translated comment
     */
    void createOutputDirectory() {
        //translated comment
        size_t lastDot = inputFilename.find_last_of('.');
        size_t lastSlash = inputFilename.find_last_of("/\\");
        
        std::string baseName;
        if (lastSlash != std::string::npos) {
            baseName = inputFilename.substr(lastSlash + 1);
        } else {
            baseName = inputFilename;
        }
        
        if (lastDot != std::string::npos && lastDot > lastSlash) {
            baseName = baseName.substr(0, lastDot - (lastSlash != std::string::npos ? lastSlash + 1 : 0));
        }
        
        outputDirectory = baseName + "_split_output";
        
        //translated comment
        try {
            if (!fs::exists(outputDirectory)) {
                fs::create_directory(outputDirectory);
                std::cout << "创建输出目录: " << outputDirectory << std::endl;
            } else {
                std::cout << "输出目录已存在: " << outputDirectory << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "创建目录失败: " << e.what() << std::endl;
            outputDirectory = "."; //translated comment
        }
    }
    
    /**
     *CSV ，
     */
    bool analyzeCSVStructure() {
        std::ifstream file(inputFilename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << inputFilename << std::endl;
            return false;
        }
        
        std::cout << "正在分析CSV文件结构..." << std::endl;
        
        //translated comment
        if (!std::getline(file, headerLine)) {
            std::cerr << "无法读取文件头部" << std::endl;
            file.close();
            return false;
        }
        
        //translated comment
        std::stringstream ss(headerLine);
        std::string cell;
        totalColumns = 0;
        while (std::getline(ss, cell, delimiter)) {
            totalColumns++;
        }
        
        //translated comment
        totalRows = 0;
        std::string line;
        while (std::getline(file, line)) {
            totalRows++;
        }
        
        file.close();
        
        std::cout << "文件分析完成:" << std::endl;
        std::cout << "  - 列数: " << totalColumns << std::endl;
        std::cout << "  - 数据行数: " << totalRows << std::endl;
        std::cout << "  - 总数据点数: " << totalColumns * totalRows << std::endl;
        std::cout << "  - 估计总数据大小: " << std::fixed << std::setprecision(2) 
                  << (double)(totalColumns * totalRows * sizeof(double)) / (1024.0 * 1024.0) << " MB" << std::endl;
        
        return totalColumns > 0 && totalRows > 0;
    }
    
    /**
     *translated comment
     */
    size_t calculateRowsForTargetSize(double targetSizeMB) {
        double targetSizeBytes = targetSizeMB * 1024.0 * 1024.0;
        double bytesPerRow = totalColumns * sizeof(double);
        size_t targetRows = static_cast<size_t>(std::round(targetSizeBytes / bytesPerRow));
        
        //translated comment
        return std::min(targetRows, totalRows);
    }
    
    /**
     *translated comment
     */
    std::string generateOutputFilename(double sizeInMB) {
        //translated comment
        size_t lastDot = inputFilename.find_last_of('.');
        size_t lastSlash = inputFilename.find_last_of("/\\");
        
        std::string baseName;
        if (lastSlash != std::string::npos) {
            baseName = inputFilename.substr(lastSlash + 1);
        } else {
            baseName = inputFilename;
        }
        
        if (lastDot != std::string::npos && lastDot > lastSlash) {
            baseName = baseName.substr(0, lastDot - (lastSlash != std::string::npos ? lastSlash + 1 : 0));
        }
        
        std::ostringstream oss;
        oss << outputDirectory << "/" << baseName << "_" 
            << std::fixed << std::setprecision(1) << sizeInMB << "MB.csv";
        return oss.str();
    }
    
    /**
     *translated comment
     */
    bool createSplitFile(double targetSizeMB) {
        size_t targetRows = calculateRowsForTargetSize(targetSizeMB);
        
        if (targetRows == 0) {
            std::cout << "跳过 " << targetSizeMB << "MB - 目标大小太小" << std::endl;
            return true;
        }
        
        if (targetRows > totalRows) {
            std::cout << "跳过 " << targetSizeMB << "MB - 目标大小超过文件总大小" << std::endl;
            return true;
        }
        
        std::ifstream inputFile(inputFilename);
        if (!inputFile.is_open()) {
            return false;
        }
        
        std::string outputFilename = generateOutputFilename(targetSizeMB);
        std::ofstream outputFile(outputFilename);
        if (!outputFile.is_open()) {
            inputFile.close();
            return false;
        }
        
        //translated comment
        outputFile << headerLine << "\n";
        
        //translated comment
        std::string line;
        std::getline(inputFile, line);
        
        size_t linesWritten = 0;
        size_t validDataPoints = 0;
        
        std::cout << "正在创建 " << targetSizeMB << "MB 文件..." << std::endl;
        
        //translated comment
        while (std::getline(inputFile, line) && linesWritten < targetRows) {
            //translated comment
            size_t validPointsInLine = countValidDataPoints(line);
            
            outputFile << line << "\n";
            linesWritten++;
            validDataPoints += validPointsInLine;
            
            //translated comment
            if (linesWritten % 10000 == 0 || linesWritten == targetRows) {
                double currentSizeMB = (double)(validDataPoints * sizeof(double)) / (1024.0 * 1024.0);
                double progress = (double)linesWritten / targetRows * 100.0;
                
                std::cout << "  进度: " << std::fixed << std::setprecision(1) << progress << "% "
                          << "(" << linesWritten << "/" << targetRows << " 行) "
                          << "当前数据大小: " << std::setprecision(3) << currentSizeMB << " MB" << std::endl;
            }
        }
        
        inputFile.close();
        outputFile.close();
        
        //translated comment
        double actualSizeMB = (double)(validDataPoints * sizeof(double)) / (1024.0 * 1024.0);
        double accuracy = (actualSizeMB / targetSizeMB) * 100.0;
        
        std::cout << "✓ 创建文件: " << fs::path(outputFilename).filename().string() << std::endl;
        std::cout << "  - 目标大小: " << std::fixed << std::setprecision(3) << targetSizeMB << " MB" << std::endl;
        std::cout << "  - 实际大小: " << std::setprecision(3) << actualSizeMB << " MB" << std::endl;
        std::cout << "  - 精度: " << std::setprecision(2) << accuracy << "%" << std::endl;
        std::cout << "  - 行数: " << linesWritten + 1 << " (含头部)" << std::endl;
        std::cout << "  - 有效数据点: " << validDataPoints << std::endl;
        std::cout << std::endl;
        
        return true;
    }
    
    /**
     *translated comment
     */
    size_t countValidDataPoints(const std::string& line) {
        std::stringstream ss(line);
        std::string cell;
        size_t validCount = 0;
        
        while (std::getline(ss, cell, delimiter)) {
            if (!cell.empty()) {
                char* endptr = nullptr;
                double val = std::strtod(cell.c_str(), &endptr);
                
                //translated comment
                if (endptr != cell.c_str() && !std::isnan(val) && std::isfinite(val)) {
                    validCount++;
                }
            }
        }
        
        return validCount;
    }
    
public:
    /**
     *CSV
     */
    bool splitCSV() {
        if (!analyzeCSVStructure()) {
            return false;
        }
        
        if (targetSizesInMB.empty()) {
            std::cerr << "没有指定目标大小" << std::endl;
            return false;
        }
        
        std::cout << "========================================" << std::endl;
        std::cout << "开始分割文件..." << std::endl;
        std::cout << "========================================" << std::endl;
        
        //translated comment
        bool success = true;
        for (double targetSize : targetSizesInMB) {
            if (!createSplitFile(targetSize)) {
                std::cerr << "创建 " << targetSize << "MB 文件失败" << std::endl;
                success = false;
                continue;
            }
        }
        
        return success;
    }
    
    /**
     *translated comment
     */
    void showConfiguration() {
        std::cout << "CSV分割器配置:" << std::endl;
        std::cout << "  - 输入文件: " << inputFilename << std::endl;
        std::cout << "  - 分隔符: '" << delimiter << "'" << std::endl;
        std::cout << "  - 输出目录: " << outputDirectory << std::endl;
        std::cout << "  - 目标大小数量: " << targetSizesInMB.size() << std::endl;
        std::cout << "  - 大小范围: " << std::fixed << std::setprecision(1) 
                  << *std::min_element(targetSizesInMB.begin(), targetSizesInMB.end()) << "MB - " 
                  << *std::max_element(targetSizesInMB.begin(), targetSizesInMB.end()) << "MB" << std::endl;
    }
};

/**
 *translated comment
 */
void showUsage(const char* programName) {
    std::cout << "增强版CSV分割器 - 基于数据大小精确分割" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "用法 1 (自定义范围): " << programName << " <CSV文件> <起始大小MB> <结束大小MB> <步长MB> [分隔符]" << std::endl;
    std::cout << "用法 2 (默认范围): " << programName << " <CSV文件>" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << programName << " data.csv                    # 默认: 30-300MB, 步长30MB" << std::endl;
    std::cout << "  " << programName << " data.csv 10 100 10          # 10-100MB, 步长10MB" << std::endl;
    std::cout << "  " << programName << " data.csv 5 50 5 ';'         # 5-50MB, 步长5MB, 分号分隔" << std::endl;
    std::cout << std::endl;
    std::cout << "特性:" << std::endl;
    std::cout << "  - 基于实际double数据量进行精确分割" << std::endl;
    std::cout << "  - 支持自定义大小范围和步长" << std::endl;
    std::cout << "  - 高精度大小控制（通常误差<1%）" << std::endl;
    std::cout << "  - 自动验证数据有效性" << std::endl;
    std::cout << "  - 详细的进度和统计信息" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 6) {
        showUsage(argv[0]);
        return 1;
    }
    
    std::string inputFile = argv[1];
    
    //translated comment
    std::ifstream testFile(inputFile);
    if (!testFile.is_open()) {
        std::cerr << "错误: 无法打开文件 " << inputFile << std::endl;
        return 1;
    }
    testFile.close();
    
    EnhancedCSVSplitter* splitter = nullptr;
    
    if (argc == 2) {
        //translated comment
        splitter = new EnhancedCSVSplitter(inputFile);
    } else if (argc >= 5) {
        //translated comment
        double startSize = std::stod(argv[2]);
        double endSize = std::stod(argv[3]);
        double stepSize = std::stod(argv[4]);
        char delimiter = (argc == 6) ? argv[5][0] : ',';
        
        if (startSize <= 0 || endSize <= 0 || stepSize <= 0) {
            std::cerr << "错误: 所有大小参数必须大于0" << std::endl;
            return 1;
        }
        
        splitter = new EnhancedCSVSplitter(inputFile, startSize, endSize, stepSize, delimiter);
    } else {
        showUsage(argv[0]);
        return 1;
    }
    
    std::cout << "增强版CSV分割器启动" << std::endl;
    std::cout << "========================================" << std::endl;
    
    splitter->showConfiguration();
    std::cout << "========================================" << std::endl;
    
    if (splitter->splitCSV()) {
        std::cout << "========================================" << std::endl;
        std::cout << "所有文件处理完成!" << std::endl;
        std::cout << "输出文件保存在: " << std::endl;
        std::cout << "  " << fs::current_path() << "/" << fs::path(inputFile).stem().string() << "_split_output/" << std::endl;
    } else {
        std::cerr << "处理过程中出现错误!" << std::endl;
        delete splitter;
        return 1;
    }
    
    delete splitter;
    return 0;
}