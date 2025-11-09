import re

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