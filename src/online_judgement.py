import re
def str_cmp(sample):
    gt = sample["gt"]
    if gt is None:
        raise ValueError("Ground truth is not provided in the sample.")
    response = sample["response"]

    # directly match
    matches = re.findall(f"{gt}", response)
    if len(matches) >= 1:
        return 1.0
    else:
        return 0.0
    # First find some key words
    # key_words = ["final answer", "Final answer", "correct option", "correct answer", ]
    matches = re.findall(r"\b([A-D])\b", response)
