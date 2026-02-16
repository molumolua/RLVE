import os
import json
import pandas as pd

def extract_user_prompt(prompt_data):
    """
    从 prompt 字段中提取 user_prompt。
    
    参数:
    prompt_data: 可能是 list (包含 dict 或 str) 或 str
    
    返回:
    str: 提取出的 user prompt 字符串
    """
    for item in reversed(prompt_data):
        if isinstance(item, dict) and item.get('role') == 'user':
            return item.get('content')
    
    return str(prompt_data)

def convert_parquet_to_json(raw_data_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 获取所有 .parquet 文件
    parquet_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.parquet')]
    
    print(f"Found {len(parquet_files)} parquet files in {raw_data_dir}")

    for file_name in parquet_files:
        file_path = os.path.join(raw_data_dir, file_name)
        
        try:
            # 读取 Parquet 文件
            df = pd.read_parquet(file_path)
            
            output_data = []
            
            # 遍历每一行
            for index, row in df.iterrows():
                entry = {}
                
                # 处理 prompt -> user_prompt
                if 'prompt' in row:
                    entry['user_prompt'] = extract_user_prompt(row['prompt'])
                else:
                    # 如果没有 prompt 字段，尝试查找其他可能的字段名，或跳过
                    print(f"Warning: 'prompt' field missing in row {index} of {file_name}")
                    continue
                
                # 处理 ground_truth -> ground_truth (原封不动)
                if 'reward_model' in row:
                    entry['ground_truth'] = row['reward_model']['ground_truth']
                elif 'ground_truth' in row:
                    entry['ground_truth'] = row['ground_truth']
                else:
                    entry['ground_truth'] = "" # 或者根据需求设置为 None
                
                output_data.append(entry)
            
            # 构建输出文件路径
            json_file_name = file_name.replace('.parquet', '.json')
            output_path = os.path.join(output_dir, json_file_name)
            
            # 写入 JSON 文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
                
            print(f"Converted {file_name} -> {output_path} ({len(output_data)} records)")
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    # 定义输入和输出目录
    RAW_DATA_DIR = "./raw_data"
    OUTPUT_DIR = "./data/EVAL"
    
    convert_parquet_to_json(RAW_DATA_DIR, OUTPUT_DIR)
