
import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import json
import requests
import re

import logging
import os
import sys
import regex
import traceback

from CV_relate_code import *
# from InformationRetriever import information_retriever_complete
from IR1 import *

# from IR2 import *

import os
import json
import shutil
import logging
from datetime import datetime
import pytz
import os
import json
import shutil
import logging
import threading
import traceback
import time


# 定义一个自定义的北京时间格式化器
class BeijingFormatter(logging.Formatter):
    def converter(self, timestamp):
        # 将时间戳转换为UTC时间
        dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        # 将UTC时间转换为北京时间
        return dt.astimezone(pytz.timezone('Asia/Shanghai'))

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        # 使用自定义的日期格式
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    def format(self, record):
        record.asctime = self.formatTime(record)
        return super().format(record)

# 创建一个自定义的日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)













def update_cv_result_with_ir_data(key, cv_output_file_path, ir_output_file_path):
    try:
        # 从IR_output_file_path中提取内容
        with open(ir_output_file_path, 'r') as ir_file:
            ir_data = json.load(ir_file)

        # 只提取QA键值
        qa_data = ir_data.get('QA', {})

        # 更新CV_output_file_path中的json文件
        cv_json_file_path = os.path.join(cv_output_file_path)

        # 读取CV_result.json内容
        with open(cv_json_file_path, 'r') as cv_file:
            cv_data = json.load(cv_file)

        # 追加存储在指定键值部分
        if key in cv_data:
            if 'QA' in cv_data[key]:
                cv_data[key]['QA'].update(qa_data)
            else:
                cv_data[key]['QA'] = qa_data
        else:
            cv_data[key] = {'QA': qa_data}

        # 将更新后的数据写回CV_result.json
        with open(cv_json_file_path, 'w') as cv_file:
            json.dump(cv_data, cv_file, indent=4)

        logging.info("Updated CV_result.json with QA data from IR_result.json")

    except Exception as e:
        logging.error(f"Failed to update CV_result.json: {e}")





def extract_qa_contexts(cv_output_file_path):
    # 检查文件是否存在，如果存在则读取现有内容
    if os.path.exists(cv_output_file_path):
        with open(cv_output_file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                logging.error("Error reading JSON file.")
                return {}
    else:
        logging.error(f"File {cv_output_file_path} does not exist.")
        return {}

    # 提取Initial_Question_Generation和Follow_Up_Question_{counter}的内容
    new_QA_CONTEXTS = {}
    for key, value in data.items():
        if key == "Initial_Question_Generation":
            initial_question_generation = {
                "Question": value.get("Question", ""),
                "Answer": value.get("Answer", ""),
                "Confidence": value.get("Confidence", "")
            }
            new_QA_CONTEXTS[key] = initial_question_generation
        elif key.startswith("Follow_Up_Question_"):
            new_QA_CONTEXTS[key] = value

    return new_QA_CONTEXTS





# 提取video_id的方法
def extract_video_id(CV_output_file_path):
    # 获取文件名
    file_name = os.path.basename(CV_output_file_path)
    
    # 去掉后缀名
    file_base_name = os.path.splitext(file_name)[0]
    
    # 去掉"_CV_result"部分，得到video_id
    video_id = file_base_name.replace("_CV_result", "")
    
    return video_id








































def process_claim_verification(CV_output_file_path, Claim, Video_information, QA_CONTEXTS):
    # 检查文件是否存在
    if not os.path.exists(CV_output_file_path):
        # 如果文件不存在，创建文件并写入一个空的JSON对象
        with open(CV_output_file_path, 'w') as file:
            json.dump({}, file)

    with open(CV_output_file_path, 'r+') as file:
        data = json.load(file)
        data["Claim"] = Claim
        data["Video_information"] = Video_information
        file.seek(0)
        json.dump(data, file, indent=4)

    try:
        # 调用 process_claim_verifier 函数
        judgment, confidence = process_claim_verifier(Claim, Video_information, QA_CONTEXTS, CV_output_file_path)
        logging.info("process_claim_verifier result - Judgment: %s, Confidence: %s", judgment, confidence)

        # 判断是否需要生成问题
        if_generate_question = not (judgment and confidence >= 0.92)

        logging.warning("\n" * 5)
        logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info("!!!!!!!!!! Processing Initial Question !!!!!!!!!!")
        logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.warning("\n" * 5)

        if if_generate_question:
            max_attempts = 3
            attempts = 0
            is_now_QA_useful = False

            while not is_now_QA_useful and attempts < max_attempts:
                try:
                    # key, question = generate_initial_question(Claim, Video_information, CV_output_file_path)
                    key, primary_question, secondary_questions = generate_initial_question(Claim, Video_information, CV_output_file_path)


                    logging.info("Generated Initial Question: %s", primary_question)

                    key_folder_path = os.path.join(os.path.dirname(CV_output_file_path), key)
                    os.makedirs(key_folder_path, exist_ok=True)

                    IR_output_file_path = os.path.join(key_folder_path, "IR_result.json")


                    video_id = extract_video_id(CV_output_file_path)

                    information_retriever_complete(Claim, Video_information, QA_CONTEXTS, primary_question, IR_output_file_path, video_id)
                    logging.info("IR results saved to: %s", IR_output_file_path)


                    with open(IR_output_file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        newest_QA_Context = data['QA']
                    
                    is_now_QA_useful = get_validator_result(Claim, Video_information, newest_QA_Context)
                    
                    attempts += 1
                except Exception as e:
                    logging.error("Error in initial question generation attempt %d: %s", attempts, str(e))
                    logging.error(traceback.format_exc())

            if is_now_QA_useful:
                update_cv_result_with_ir_data(key, CV_output_file_path, IR_output_file_path)
            else:
                logging.warning("Max generate_initial_question attempts reached and QA context is still not useful.")

        new_question_count = 1
        while if_generate_question:
            new_QA_CONTEXTS = extract_qa_contexts(CV_output_file_path)

            new_judgment, new_confidence = process_claim_verifier(Claim, Video_information, new_QA_CONTEXTS, CV_output_file_path)
            logging.info("New process_claim_verifier result - Judgment: %s, Confidence: %s", new_judgment, new_confidence)

            if new_judgment and new_confidence >= 0.92:
                break

            if new_question_count >= 3:
                break

            logging.warning("\n" * 5)
            logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.info("!!!!!!!!!! Processing question #%d !!!!!!!!!!", new_question_count)
            logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.warning("\n" * 5)

            new_question_count += 1

            max_attempts = 3
            attempts = 0
            is_now_QA_useful = False

            while not is_now_QA_useful and attempts < max_attempts:
                try:
                    new_key, follow_up_question = generate_follow_up_question(Claim, Video_information, new_QA_CONTEXTS, secondary_questions, CV_output_file_path)
                    logging.info("%s Generated Question: %s", new_key, follow_up_question)

                    new_key_folder_path = os.path.join(os.path.dirname(CV_output_file_path), new_key)
                    os.makedirs(new_key_folder_path, exist_ok=True)

                    IR_output_file_path = os.path.join(new_key_folder_path, "IR_result.json")

                    # information_retriever_complete(Claim, Video_information, new_QA_CONTEXTS, follow_up_question, IR_output_file_path)

                    video_id = extract_video_id(CV_output_file_path)

                    information_retriever_complete(Claim, Video_information, QA_CONTEXTS, follow_up_question, IR_output_file_path, video_id)

                    
                    logging.info("IR results saved to: %s", IR_output_file_path)

                    with open(IR_output_file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        newest_QA_Context = data['QA']
                    
                    is_now_QA_useful = get_validator_result(Claim, Video_information, newest_QA_Context)
                    
                    attempts += 1
                except Exception as e:
                    logging.error("Error in follow-up question generation attempt %d: %s", attempts, str(e))
                    logging.error(traceback.format_exc())

            if is_now_QA_useful:
                update_cv_result_with_ir_data(new_key, CV_output_file_path, IR_output_file_path)
            else:
                logging.warning("Max generate_follow_up_question attempts reached and QA context is still not useful.")

        new_QA_CONTEXTS = extract_qa_contexts(CV_output_file_path)
        final_json_answer = process_claim_final(Claim, Video_information, new_QA_CONTEXTS, CV_output_file_path)
        logging.info("final_json_answer \n%s", final_json_answer)

    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        logging.error(traceback.format_exc())











# 定义一个包装函数来处理超时
def process_with_timeout(CV_output_file_path, Claim, Video_information, QA_CONTEXTS, timeout, event):
    thread = threading.Thread(target=process_claim_verification, args=(CV_output_file_path, Claim, Video_information, QA_CONTEXTS))
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logging.error("Timeout reached for %s", CV_output_file_path)
        event.set()  # 设置事件，通知超时










def main():


    target_folder = r"/home/public/FakeNews/nkp/LLMFND_minicpm/result/new"


    json_files = sorted([f for f in os.listdir(target_folder) if f.endswith('.json')], reverse=True)



    # 初始化计数器
    count = 0

    # 处理每个JSON文件
    for file_name in json_files:
        try:
            count += 1
            print(f"---------- Processing {count}/{len(json_files)}: {file_name} ----------")

            json_file_path = os.path.join(target_folder, file_name)
            file_base_name = os.path.splitext(file_name)[0]
            output_folder = os.path.join(target_folder, file_base_name)
            CV_output_file_path = os.path.join(output_folder, f"{file_base_name}_CV_result.json")

            if os.path.exists(CV_output_file_path):
                with open(CV_output_file_path, 'r', encoding='utf-8') as f:
                    cv_data = json.load(f)
                    if "Final_Judgement" in cv_data:
                        print(f"Skipping {file_name} as it has already been processed.")
                        continue

            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder, exist_ok=True)

            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            Claim = data["claim"]
            Video_information = data["Video_information"]

            log_file_name = f'{file_base_name}_claim_verifer.log'
            log_file_path = os.path.join(output_folder, log_file_name)

            if os.path.exists(log_file_path):
                os.remove(log_file_path)

            file_handler = logging.FileHandler(log_file_path)
            formatter = BeijingFormatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            timeout_event = threading.Event()
            timeout_minutes = 13

            process_with_timeout(CV_output_file_path, Claim, Video_information, {}, timeout_minutes * 60, timeout_event)

            if timeout_event.is_set():
                print(f"Timeout reached for {file_name}. Skipping to the next file.")
                logger.removeHandler(file_handler)
                continue

            shutil.move(log_file_path, os.path.join(output_folder, log_file_name))
            logger.removeHandler(file_handler)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            logging.error(f"Error processing file {file_name}: {e}")
            logging.error(traceback.format_exc())
        finally:
            if 'file_handler' in locals():
                logger.removeHandler(file_handler)

if __name__ == "__main__":
    for attempt in range(3):
        try:
            main()
            break
        except Exception as e:
            print("Terminated")
            logging.error("Terminated")
            logging.error(traceback.format_exc())
            print("Error occurred during execution. Pausing for 10 seconds...")
            time.sleep(10)
        else:
            print("Processed successfully.")