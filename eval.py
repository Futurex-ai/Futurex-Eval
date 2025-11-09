from typing import List, Set, Dict, Any
from tenacity import retry, stop_after_attempt, wait_random_exponential
from smolagents import AzureOpenAIServerModel
from concurrent.futures import ThreadPoolExecutor
import traceback
import concurrent.futures
import re
from llm_judge_level_34 import judge_level_34_score
from utils import to_float
import os
from openai import OpenAI

def log_before_retry(retry_state):
    """Log before each retry"""
    print(
        f"Attempt {retry_state.attempt_number} failed, retrying in {retry_state.next_action.sleep} seconds..."
    )

def extract_by_gpt(questions: List[str], draft_answers: List[str]):
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_API_MODEL")
    base_url = os.getenv("OPENAI_API_BASE")  # Optional, for custom API addresses (e.g., locally deployed models)
    # Create a general OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url  # If using the official OpenAI API, this parameter can be omitted (defaults to the official address)
    )
    
    prompt = "Given the question and a draft answer, please extract the answer from the draft answer. If the answer is wrapped in \\boxed{{}}, \\text{{}} or other format, please extract the content inside the braces. Finally, identify if the answer is a number. Please only output the extracted answer and whether the answer is an number, without outputing other any content. For example:\n$115\nyes\nAnother example:\nBeijing\nNo \n\nQeustion: {question}\n\nDraft Answer: {answer}"
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10), before_sleep=log_before_retry)
    def process_single(question, draft_answer):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt.format(question=question, answer=draft_answer)}
                ]
            ).choices[0].message.content
            return response
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            raise  # Must re-raise the exception for tenacity to catch and retry
    
    # Use ThreadPoolExecutor to execute multithreaded tasks
    with ThreadPoolExecutor(max_workers=4) as executor:  # Number of threads can be adjusted
        responses = list(executor.map(process_single, questions, draft_answers))
    
    return responses

def extract_answer(title, event):
    if 'answer' in event and 'prediction' not in event:
        event['prediction'] = event['answer']
    if event['prediction'] is None:
        return None
    if isinstance(event['prediction'], list):
        event['prediction'] = event['prediction'][0]
    
    result = extract_by_gpt([title + '\n',], [event['prediction']])[0]
    event['prediction'] = '\n'.join(result.split('\n')[:-1])
    is_number = result.split('\n')[-1].strip()

    if is_number.lower() == 'yes':
        event['prediction'] = [to_float(event['prediction'])]
    else:
        if ',' in event['prediction']:
            event['prediction'] = [x.strip() for x in event['prediction'].split(',')]
        else:
            event['prediction'] = [event['prediction']] # unify the format
    
    return event['prediction']

def estimate_score_level_1_2(
    questions: List[str],
    y_true_raw: List[Set[str]], 
    y_pred_raw: List[Set[str]]
) -> Dict[str, Any]:
    """
    Evaluates multi-label classification performance, safely handling empty inputs.
    """
    assert len(y_true_raw) == len(y_pred_raw), "The lengths of the true labels and predicted labels lists must be the same."
    
    f1_scores = []
    for y_true, y_pred in zip(y_true_raw, y_pred_raw):
        if len(y_true) == 1:
            f1_scores.append(1.0 if y_true == y_pred else 0.0)
            continue
        # Ensure data is of set type for easy intersection and difference calculations
        y_true = set(y_true)
        y_pred = set(y_pred)
        
        # --- Calculate F1-score for a single question ---
        
        # If both true and predicted are empty sets
        if not y_true and not y_pred:
            f1_scores.append(0.)
            continue
            
        # Calculate TP (True Positives): The correctly predicted part, i.e., the intersection
        tp = len(y_true.intersection(y_pred))
        
        # Calculate Precision (Precision = TP / |Predicted|)
        # |Predicted| = TP + FP (False Positives)
        if len(y_pred) == 0:
            precision = 0.0
        else:
            precision = tp / len(y_pred)
            
        # Calculate Recall (Recall = TP / |True|)
        # |True| = TP + FN (False Negatives)
        if len(y_true) == 0:
            recall = 0.0
        else:
            recall = tp / len(y_true)
            
        # Calculate F1-score (F1 = 2 * Precision * Recall / (Precision + Recall))
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        f1_scores.append(f1)

    # --- Average the F1-scores for all questions ---
    if not f1_scores:
        return 0.0
    return f1_scores, sum(f1_scores) / len(f1_scores)

def estimate_score_level_3_4(questions, y_true_raw, y_pred_raw, stds, max_workers=8):
    """
    Estimates the score for Type B questions using a thread pool and caching.
    """
    # 2. Add lru_cache decorator to the worker function
    # maxsize=None means the cache can grow indefinitely, you can also set a specific value, e.g., maxsize=1024
    def worker(question, y_true, y_pred, std):
        """
        A wrapper function to call the judging API. Results are now cached.
        """
        try:
            if std is None:
                std = 1.0
            # judge_level_34_score will only be called the first time the (question, y_true, y_pred) combination appears
            return judge_level_34_score(
                originaL_question=question,
                model_prediction=y_pred,
                real_answer=y_true,
                std=std
            )
        except Exception as e:
            print(f"Error processing question at [estimate_type_b_score]: {question}, pred: {y_pred}, true: {y_true}. Error: {e}")
            traceback.print_exc()
            return 0.0
    
    if not all([questions, y_true_raw, y_pred_raw, stds]):
        return 0.0

    all_scores = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # The map function will map identical tasks (worker calls with the same parameters) to the cached results
       results = executor.map(worker, questions, y_true_raw, y_pred_raw, stds)
       all_scores = list(results)

    avg_score = sum(all_scores) / len(questions) if questions else 0.0
    
    return all_scores, avg_score

if __name__ == "__main__":
    questions = [
        "What is the answer?",
        "What is the answer?",
    ]
    y_true_1_2 = [
        set(["A"]), 
        set(["A", "B"]),
    ]
    y_pred_1_2 = [
        set(["\\boxed{C}"]), 
        set(["\\boxed{D}", "\\boxed{B}"]), 
    ]
    y_pred_1_2 = [extract_answer(questions[i], {"prediction": pred}) for i, pred in enumerate(y_pred_1_2)]
    scores_1_2, avg_score_1_2 = estimate_score_level_1_2(questions, y_true_1_2, y_pred_1_2)
