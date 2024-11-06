

import logging

# 清空日志文件并配置日志系统
# open('test.log', 'w').close()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='test.log', filemode='a')


import time

import json
import os

# 接下来的8哥函数都是用来计算G-Eval类似的指标的
from Comprehensiveness import evaluate_comprehensiveness
from conciseness import evaluate_conciseness
from currency import evaluate_currency
from Fact_Hallucination import evaluate_fact_hallucination
from faithfulness import evaluate_faithfulness
from logical_consistency import evaluate_logical_consistency
from strength_of_evidence import evaluate_strength_of_evidence
from Reasons_Evidence_Omission import evaluate_Omission_of_Reasons_and_Evidence


# 用于计算ROUGE和BLEU分数
from bleu_rouge import compute_rouge_bleu
















start_time = time.time()

json_path = r'E:\aim\AAAI\AAAI25\Experiment\evaluate_metrics\evaluation\test_example\1929908\1929908_CV_result-4omini.json'

# Load the JSON data from the file
# 使用UTF-8编码读取JSON文件
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取需要的内容
document_content = {
    "Claim": data['Claim'],
    "Video_information": {
        "video_date": data['Video_information']['video_date'],
        "platform": data['Video_information']['platform'],
        "video_headline": data['Video_information']['video_headline'],
        "video_transcript": data['Video_information']['video_transcript']
    },
    "Final_Judgement": {
        "Answer": data['Final_Judgement']['Answer'],
        "Reasons": data['Final_Judgement']['Reasons']
    }
}

# 提取Evidences
all_evidences_content = {
    "Evidences": data['Evidences']
}


logging.info(f"Document Content: \n{document_content}")
logging.info(f"All Evidences Content: \n{all_evidences_content}")




# 生成 content_credibility
content_credibility = {
    "comprehensiveness": evaluate_comprehensiveness(document_content, all_evidences_content),
    "conciseness": evaluate_conciseness(document_content, all_evidences_content),
    "currency": evaluate_currency(document_content,all_evidences_content),
    "fact_hallucination": evaluate_fact_hallucination(document_content, all_evidences_content),
    "faithfulness": evaluate_faithfulness(document_content, all_evidences_content),
    "logical_consistency": evaluate_logical_consistency(document_content, all_evidences_content),
    "strength_of_evidence": evaluate_strength_of_evidence(document_content, all_evidences_content)
}

# 将 content_credibility 添加到原始数据中
data['content_credibility'] = content_credibility

# 将更新后的数据写回到JSON文件中
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Content credibility added successfully.")






# 定义要查找的目录路径
target_gt_path = r'E:\aim\AAAI\AAAI25\Experiment\evaluate_metrics\evaluation\data'

# 目标 Claim，假设从已加载的数据中获取
target_claim = data['Claim']

# 初始化一个变量用于存储找到的结果
ground_truth_content = None

# 遍历目录中的所有 JSON 文件
for filename in os.listdir(target_gt_path):
    if filename.endswith('.json'):
        file_path = os.path.join(target_gt_path, filename)
        
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            
            # 检查 Claim 是否匹配
            if json_data.get('claim') == target_claim:
                # 提取所需的内容
                ground_truth_content = {
                    "original_rationales": json_data.get('original_rationales', {}),
                    "summary_rationales": json_data.get('summary_rationales', {}),
                    "evidences": json_data.get('evidences', {}),
                    "relationship_with_evidence": json_data.get('relationship_with_evidence', [])
                }

                logging.info(f"GroundTruth Content: \n{ground_truth_content}")
                # 如果找到匹配的文件，就跳出循环
                break





def flatten_dict_values(d):
    """
    将嵌套字典的所有值提取出来并合并成一个字符串。
    """
    values = []
    for value in d.values():
        if isinstance(value, dict):
            values.append(flatten_dict_values(value))
        else:
            values.append(str(value))
    return " ".join(values)




# 如果找到匹配的内容，则将其添加到原始数据的 'GroundTruth' 键下
if ground_truth_content:

    reasons_str = document_content["Final_Judgement"]["Reasons"]

    logging.info("-"*50)

    # 提取并拼接原始推理的值
    original_rationales_str = flatten_dict_values(ground_truth_content["original_rationales"])

    # 提取并拼接总结推理的值
    summary_rationales_str = flatten_dict_values(ground_truth_content["summary_rationales"])

    logging.info(f"Reasons: \n{reasons_str}")
    logging.info(f"Original Rationales: \n{original_rationales_str}")
    logging.info(f"Summary Rationales: \n{summary_rationales_str}")



    # 计算 Reasons 与 original_rationales 的 ROUGE 和 BLEU 分数
    original_scores = compute_rouge_bleu(original_rationales_str, reasons_str)

    # 计算 Reasons 与 summary_rationales 的 ROUGE 和 BLEU 分数
    summary_scores = compute_rouge_bleu(summary_rationales_str, reasons_str)

    # 将分数添加到数据中
    data['comparison_with_ground_truth'] = {
        "reasons_vs_original_rationales": original_scores,
        "reasons_vs_summary_rationales": summary_scores
    }


    comparison_with_gt_score = evaluate_Omission_of_Reasons_and_Evidence(document_content, all_evidences_content, ground_truth_content)

    # # 将 content_credibility 添加到原始数据中
    data['llm_comparison_with_gt_score'] = comparison_with_gt_score

    # 将更新后的数据写回到JSON文件中
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("GroundTruth content added successfully.")
else:
    print("No matching claim found in the specified directory.")




end_time = time.time()

logging.info(f"Time taken: {end_time - start_time:.2f} seconds")

