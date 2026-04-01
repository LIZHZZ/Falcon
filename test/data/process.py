import csv
import pandas
import os

from itertools import combinations

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

def remove_highly_similar_columns(df, threshold=0.9):
    """
    检查 DataFrame 中所有列的相似性，并移除相似度超过阈值的冗余列。

    参数:
    df (pd.DataFrame): 输入的 DataFrame。
    threshold (float): 相似度阈值，默认为 0.9 (90%)。

    返回:
    pd.DataFrame: 移除了高度相似列之后的新 DataFrame。
    """
    print("--- 开始检测高度相似的列 ---")
    #translated comment
    cols_to_drop = set()
    
    #translated comment
    all_cols = df.columns.tolist()
    
    #itertools.combinations
    #， [A, B, C]， (A, B), (A, C), (B, C)
    for col1, col2 in combinations(all_cols, 2):
        
        #translated comment
        if col1 in cols_to_drop or col2 in cols_to_drop:
            continue
            
        #translated comment
        matches = (df[col1] == df[col2]).sum()
        total_rows = len(df)
        
        #translated comment
        if total_rows > 0:
            similarity = matches / total_rows
        else:
            similarity = 0
            
        #translated comment
        if similarity > threshold:
            #translated comment
            cols_to_drop.add(col2)
            print(f"- 列 '{col1}' 和 '{col2}' 的相似度为 {similarity:.2%}，超过阈值 {threshold:.0%}")
            print(f"  > 标记列 '{col2}' 为待删除。")

    #translated comment
    if cols_to_drop:
        print(f"\n--- 检测完成，共标记 {len(cols_to_drop)} 列待删除: {list(cols_to_drop)} ---")
        #， DataFrame
        df_processed = df.drop(columns=list(cols_to_drop))
        print("已成功移除冗余列。")
    else:
        print("\n--- 检测完成，未发现需要删除的高度相似列。---")
        df_processed = df.copy() #translated comment

    return df_processed

def process_and_filter_csv(input_file,output_file):
    df = pandas.read_csv(input_file)
    # print(df)

    valid_columns = []
    for column_name in df.columns:
        column = df[column_name]

        numeric_col = pandas.to_numeric(column, errors='coerce')
        # print(numeric_col)
        # print(numeric_col.notna().sum())

        is_integer = (numeric_col % 1 == 0)
        # print(is_integer.sum())
        if is_integer.sum()<numeric_col.notna().sum():
            print(column_name,is_integer.sum(),numeric_col.notna().sum())
        
        is_not_integer_column = is_integer.sum()<numeric_col.notna().sum()

        is_not_index = numeric_col.nunique()>10
        if not is_not_index:
            print(column_name, numeric_col.nunique())
        if is_not_integer_column and is_not_index:
            valid_columns.append(column_name)

    new_df = df[valid_columns].copy()

    
    cleaned_df = new_df.dropna(how='all', axis=0)

    cleaned_df = remove_highly_similar_columns(cleaned_df,0.9)
    print(cleaned_df)
    print(cleaned_df.nunique(), cleaned_df.count())

    cleaned_df.to_csv(output_file, index=False, encoding='utf-8-sig')






def batch_process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        print(f"输出文件夹 '{output_folder}' 不存在，正在创建...")
        os.makedirs(output_folder)
    
    print(f"\n--- 开始批量处理文件夹 '{input_folder}' ---")
    
    #translated comment
    found_csv_files = False
    for filename in os.listdir(input_folder):
        #3. CSV
        if filename.lower().endswith('.csv'):
            found_csv_files = True
            
            #translated comment
            input_path = os.path.join(input_folder, filename)
            
            #， data_set_A.csv -> processed_data_set_A.csv
            output_filename = f"processed_{filename}"
            output_path = os.path.join(output_folder, output_filename)
            
            print(f"\n--- 正在处理文件: {filename} ---")
            
            #translated comment
            process_and_filter_csv(input_path, output_path)
            
    if not found_csv_files:
        print("在输入文件夹中没有找到任何CSV文件。")

    print("\n--- 批量处理完成 ---")




if __name__ == "__main__":
    input_folder = "source"  #CSV
    output_folder = "cleaned2"
    
    #translated comment
    extracted_data = batch_process_folder(input_folder,output_folder)
    
    #CSV
    # with open(output_filename, 'w', newline='') as outfile:
    #     writer = csv.writer(outfile)
    #writer.writerow(["floats"]) #
    #     for value in extracted_data:
    #         writer.writerow([value])
    
    #print(f" ! {len(extracted_data)} ")
    #print(f" : {output_filename}")