import os
import re
from collections import Counter

def wrap_prediction_to_list(prediction, gt_len: int):
    """
    把 prediction 包成 judge_level_34_score 期望的 list 格式。

    当模型把多项答案输出成"item1, item2, item3"这种逗号串、而 gt 是 list-of-N 时,
    用 ", " 严格 split:仅当 prediction.count(", ") 恰好 == gt_len-1 才拆,
    避免把 "Smith, Inc." 这种实体内逗号误拆。其它情况一律包成单元素 list。
    """
    if isinstance(prediction, str) and gt_len > 1 and prediction.count(", ") == gt_len - 1:
        return [s.strip() for s in prediction.split(", ")]
    return [prediction]

def to_float(s: str) -> float:
    """
    Converts a string to a float, supporting negative numbers.
    - First, checks if the format is valid using is_number().
    - If valid, extracts the numeric part of the string (including negative sign, commas, and periods).
    - Removes thousand separators (commas).
    - Converts the cleaned string to a float.
    - Returns None if any step fails.
    """
    if not is_number(s):
        return None

    # [Core Change] Modify regex to include optional negative sign during extraction
    # -?[\d,.]+ -> Matches optional negative sign (-?) + one or more digits/commas/dots
    match = re.search(r'-?[\d,.]+', s)
    if not match:
        return None

    numeric_string = match.group(0)
    clean_string = numeric_string.replace(',', '')

    try:
        return float(clean_string)
    except ValueError:
        return None

def is_number(s: str) -> bool:
    """
    Judges a string based on the rule: returns True only if the string starts or ends with a number.
    Supports negative sign at the beginning, but not internal spaces.
    The 'numeric part' here can include digits, decimal points, and commas.
    """
    s = s.strip()
    
    # New rule: Do not allow internal spaces or newlines
    if ' ' in s or '\n' in s:
        return False
        
    if not s:
        return False
    if not re.search(r'\d', s):
        return False
        
    # [Core Change] Modify regex to support optional negative sign at the start
    # ^-?[\d,.] -> Matches a string starting with an optional negative sign (-?) + digit/comma/dot
    if re.search(r'^-?[\d,.]|[\d,.]$', s):
        return True
        
    return False