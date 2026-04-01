import csv

def extract_float_values(input_file):
    float_values = []  #translated comment
    
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  #translated comment
        
        for row in reader:
            for value in row:
                #translated comment
                if value.strip():  #translated comment
                    try:
                        num = float(value)
                        float_values.append(num)
                    except ValueError:
                        #translated comment
                        continue
    return float_values

if __name__ == "__main__":
    input_filename = "new_tsbs//float_data_complete.csv" # CSV
    output_filename = "new_tsbs//chunk.csv"
    
    #translated comment
    extracted_data = extract_float_values(input_filename)
    
    #CSV
    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["floats"])  #translated comment
        for value in extracted_data:
            writer.writerow([value])
    
    print(f"完成! 共提取 {len(extracted_data)} 个浮点数")
    print(f"结果已保存到: {output_filename}")