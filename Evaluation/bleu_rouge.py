from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import numpy as np
import json

def compute_rouge_bleu(reference_str, hypothesis_str):
    # 预处理，确保以空格分隔字符
    reference = " ".join(reference_str.strip().split())
    hypothesis = " ".join(hypothesis_str.strip().split())
    
    # 初始化 ROUGE 评估器
    rouge = Rouge()
    
    # 计算 ROUGE 分数
    rouge_scores = rouge.get_scores(hyps=hypothesis, refs=reference, avg=True)
    
    # 计算 BLEU 分数
    smoothing_fn = SmoothingFunction().method1
    
    bleu_1 = sentence_bleu(
        references=[reference.split(' ')],
        hypothesis=hypothesis.split(' '),
        weights=(1, 0, 0, 0),
        smoothing_function=smoothing_fn
    )
    
    bleu_2 = sentence_bleu(
        references=[reference.split(' ')],
        hypothesis=hypothesis.split(' '),
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing_fn
    )
    
    bleu_3 = sentence_bleu(
        references=[reference.split(' ')],
        hypothesis=hypothesis.split(' '),
        weights=(0.33, 0.33, 0.33, 0),
        smoothing_function=smoothing_fn
    )
    
    bleu_4 = sentence_bleu(
        references=[reference.split(' ')],
        hypothesis=hypothesis.split(' '),
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing_fn
    )
    
    # 构建结果字典
    result = {
        'ROUGE-1': rouge_scores['rouge-1']['f'],
        'ROUGE-2': rouge_scores['rouge-2']['f'],
        'ROUGE-L': rouge_scores['rouge-l']['f'],
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'BLEU-3': bleu_3,
        'BLEU-4': bleu_4
    }
    
    return result

# # 示例
# reference_str = "This is a test."
# hypothesis_str = "This is a test test."

# result = compute_rouge_bleu(reference_str, hypothesis_str)

# # 将结果转换为 JSON 格式
# result_json = json.dumps(result, indent=4)

# # 输出结果到控制台
# print(result_json)

# # # 保存结果到 JSON 文件
# # with open("metrics_result.json", "w") as json_file:
# #     json_file.write(result_json)
