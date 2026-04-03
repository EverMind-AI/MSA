from openai import OpenAI
import os
from tqdm import tqdm
import sys
import json
import numpy as np
from glob import glob
from concurrent.futures import ThreadPoolExecutor

def parse_match_result(text):
    text_lower = text.strip().lower()
    if 'match' in text_lower and 'mismatch' not in text_lower:
        return 1
    return 0

def build_match_prompt(gold_answer, model_answer):
    return f"""You are a strict but fair evaluator.

Judge whether the Generated Answer correctly includes
the core meaning of the Reference Answer.

- The Generated Answer may add extra details.
- Do NOT penalize additional information.
- The core concept in the Reference Answer must be present
  or clearly implied.

  
Reference Answer:
{gold_answer}

Generated Answer:
{model_answer}

Output:
match or mismatch"""


def build_score_prompt(gold_answer, model_answer, query):
    return f""""Based on the accuracy, completeness, and relevance of the predicted answer to the real answer in the context of the **query**, assign an objective score from 0 to 5 (5 being the highest, 0 the lowest).

    The scoring must strictly adhere to the following criteria. The final output can only be a single number.

    Scoring Criteria:

    5: The predicted answer is exactly the same as the real answer and correctly answers the query. Differences in wording do not affect factual accuracy.

    4: The predicted answer contains all the core information of the real answer, with no errors, but includes a small amount of non-critical redundant content.

    3: The predicted answer captures the core information but differs from the real answer in some aspects. The predicted answer is slightly incomplete or imprecise, but contains no errors.

    2: The predicted answer is partially relevant to the real answer but omits a significant amount of information or deviates from the core topic of the query.

    1: The predicted answer attempts to address the query (maintains basic relevance to the topic) but provides factually incorrect information. It does not contradict the core claim of the real answer, but shows incomplete or inaccurate understanding of the topic.

    0. The predicted answer is completely unrelated to the query, consists of gibberish, or is a pure hallucination that shares no logical connection with the real answer.

    Query:

    {query}

    True Answer:

    {gold_answer}

    Predicted Answer:

    {model_answer}

    Output only a single number (0, 1, 2, 3, 4, or 5): """



def parse_score_result(text):
    """解析 LLM 返回的评分结果，返回 0-5 的分数"""
    text = text.strip()
    # 尝试直接解析数字
    for char in text:
        if char.isdigit() and int(char) <= 5:
            return int(char)
    # 如果没有找到有效数字，返回 0
    return 0

def get_eval_response(prompt):
    try:
        completion = client.chat.completions.create(
        extra_headers={
        },
        extra_body={},
        model='google/gemini-2.5-flash',
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text":  prompt
                },
            ]
            }
        ],
        temperature=0.0 
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return ''

if __name__ == "__main__":
    dirname = sys.argv[1]
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not openrouter_api_key:
        raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )
    JUDGE_MODEL = 'google/gemini-2.5-flash'
    if dirname.endswith(".json") and os.path.isfile(dirname):
        json_dir = os.path.split(dirname)[0]
        json_paths = [dirname]
    elif os.path.exists(dirname) and os.path.isdir(dirname):
        json_dir = dirname
        json_paths = glob(f'{json_dir}/*.json')
    else:
        json_dir = f"src/evaluation/outputs/{dirname}"
        json_paths = glob(f'{json_dir}/*.json')

    for json_path in tqdm(json_paths, total=len(json_paths)):
        if 'score' in json_path:
            continue
        json_name = os.path.split(json_path)[-1].split('.')[0]
        with open(json_path, 'r') as f:
            datas = json.load(f)["anonymous"]["precision"]
        print(datas["metrics"])

        final_dict = dict()
        final_dict['metrics'] = datas["metrics"]
        final_dict['record_list'] = []
        pred_answers = []
        true_answers = []
        questions    = []
        pred_contexts = []
        true_contexts = []
        pred_ids = []
        labels_ids = []

        for dic in datas['record_list']:
            question = dic['question']
            questions.append(question)
            true_answer = dic['true_answer']
            key_word = "response" if "response" in dic else "pred_answer"
            pred_answer = dic[key_word].replace("<|im_end|>",'').replace("<|endoftext|>",'')

            pred_ids.append(dic['pred_id'])
            labels_ids.append(dic['labels_id'])

            pred_con_li = [v for di in dic['predict_context'] for k,v in di.items()]
            true_con_li = [v for di in dic['gt_context'] for k,v in di.items()]
            pred_contexts.append(pred_con_li)
            true_contexts.append(true_con_li)

            true_answers.append(true_answer)
            qa = pred_answer.split('<answer>')[-1]
            try:
                q = qa.split("The answer to the question is: ")[-2].split("The user's question is: ")[-1].replace('\n<|object_ref_end|>','')
            except:
                q = ''
            a = qa.split("The answer to the question is: ")[-1].replace('</answer>','').replace("The user's question is:",'')
            a = a.split('</think>')[-1]
            if "Answer:" in a:
                a = a.split("Answer:")[1].strip()
            pred_answers.append(a)
        
        match_prompts = []
        score_prompts = []
        for idx in tqdm(range(len(pred_answers))):
            true_answer = true_answers[idx]
            pred_answer = pred_answers[idx]
            question = questions[idx]
            match_prompt = build_match_prompt(true_answer, pred_answer)
            score_prompt = build_score_prompt(true_answer, pred_answer, question)
            match_prompts.append(match_prompt)
            score_prompts.append(score_prompt)

        
        with ThreadPoolExecutor(max_workers=32) as executor:
            score_results = list(tqdm(executor.map(get_eval_response, score_prompts), total=len(score_prompts), desc="Tokenizing texts"))

        match_final_scores = []
        score_final_scores = []
        for idx in range(len(score_results)):
            score_result = score_results[idx]
            score_score = parse_score_result(score_result)
            score_final_scores.append(score_score)

        
        match_final_scores_mean = np.mean(match_final_scores)
        score_final_scores_mean = np.mean(score_final_scores)
        print(" ======================================= ")
        print(f"{json_name} LLM Score: {score_final_scores_mean:.4f}")
        print(" ======================================= ")

        old_file_path = json_path
        new_file_path = json_path.replace(".json", f"_score{str(score_final_scores_mean).replace('.','point')[:9]}.json")
        os.rename(old_file_path, new_file_path)

        # final_dict
        for idx in range(len(questions)):
            question    =  questions[idx]
            answer      =  true_answers[idx]
            pred_answer =  pred_answers[idx]
            score       =  score_final_scores[idx]
            final_dict['record_list'].append({'question':question, 'true_answer':answer, 'pred_answer':pred_answer, 'pred_score':score})
        final_dict["score_final_scores_mean"] = score_final_scores_mean
        new_file_path = f'{json_dir}/{json_name}_llmscore.json'

        with open(new_file_path, 'w',encoding='utf-8') as f:
            json.dump(final_dict, f, indent=4,ensure_ascii=False)
