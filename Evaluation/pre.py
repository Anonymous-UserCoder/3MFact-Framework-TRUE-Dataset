






import os
import json

def extract_evidence(subfolder_path):
    # 初始化一个空字典，用于存储最终的Evidence数据
    evidence_summary = {}

    # 获取IR_result.json的路径
    ir_result_path = os.path.join(subfolder_path, 'IR_result.json')

    # 检查IR_result.json文件是否存在
    if os.path.exists(ir_result_path):
        with open(ir_result_path, 'r', encoding='utf-8') as f:
            ir_result = json.load(f)
            
            # 提取RelevantEvidence部分
            relevant_evidence = ir_result.get('RelevantEvidence', {})

            # 将子文件夹名作为键，将Evidence数据存储在evidence_summary中
            subfolder_name = os.path.basename(subfolder_path)
            evidence_summary[subfolder_name] = relevant_evidence
    
    return evidence_summary

def save_to_cv_result(json_data, output_file):
    # 检查目标文件是否存在
    if os.path.exists(output_file):
        # 如果文件存在，加载现有数据并追加
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        # 如果文件不存在，初始化为空字典
        existing_data = {}

    # 更新现有数据
    existing_data.update(json_data)

    # 将更新后的数据保存回文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

def process_subfolders(folder_path):
    evidence_summary = {}
    
    # 遍历当前文件夹的所有子文件夹
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        
        # 检查是否为子文件夹
        if os.path.isdir(subfolder_path):
            # 提取Evidence数据
            subfolder_evidence = extract_evidence(subfolder_path)
            evidence_summary.update(subfolder_evidence)
    
    return {"Evidences": evidence_summary}

def process_all_folders(base_path):
    # 遍历主文件夹中的所有子文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # 检查是否为子文件夹
        if os.path.isdir(folder_path):
            # 构建当前子文件夹中CV_result.json的路径
            output_file = os.path.join(folder_path, f'{folder_name}_CV_result.json')

            # 对该子文件夹中的所有子文件夹进行处理
            evidence_data = process_subfolders(folder_path)

            # 保存到CV_result.json
            save_to_cv_result(evidence_data, output_file)

            print(f"Evidence数据已成功保存到 {output_file}")

# if __name__ == "__main__":
#     base_path = r"E:\aim\AAAI\AAAI25\Experiment\Result\MM\all_videos_description_minicpm_with_evidence"

#     # 对主文件夹中的所有子文件夹进行处理
#     process_all_folders(base_path)










import os
import shutil

def extract_cv_result_files(base_path, destination_folder):
    # 如果目标文件夹不存在，创建它
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历主文件夹中的所有子文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # 检查是否为子文件夹
        if os.path.isdir(folder_path):
            # 构建当前子文件夹中CV_result.json的路径
            cv_result_file = os.path.join(folder_path, f'{folder_name}_CV_result.json')

            # 检查CV_result.json文件是否存在
            if os.path.exists(cv_result_file):
                # 构建目标文件路径
                destination_file_path = os.path.join(destination_folder, f'{folder_name}_CV_result.json')

                # 复制CV_result.json文件到目标文件夹
                shutil.copy(cv_result_file, destination_file_path)

                print(f"{cv_result_file} 已复制到 {destination_file_path}")

if __name__ == "__main__":
    base_path = r"E:\aim\AAAI\AAAI25\Experiment\Result\MM\all_videos_description_minicpm_with_evidence"
    destination_folder = r"E:\aim\AAAI\AAAI25\Experiment\Result\MM\only_cv_result_MM"

    # 提取所有CV_result.json文件到单独的文件夹
    extract_cv_result_files(base_path, destination_folder)
