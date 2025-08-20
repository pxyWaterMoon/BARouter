import re
def str_cmp(sample):
    gt = sample["gt"]
    if gt is None:
        raise ValueError("Ground truth is not provided in the sample.")
    response = sample["response"]
    sentences = re.split(r"[.!?\n]", response)
    # directly match
    # matches = re.findall(f"{gt}", response)
    # if len(matches) >= 1:
    #     return 1.0
    # else:
    #     return 0.0
    
    # First find some key words
    key_words = ["final answer", "Final answer", "correct option", "correct answer", "boxed", "Therefore", "Thus", "So", "correct choice"]
    for sentence in sentences:
        if any(word in sentence for word in key_words):
            # Now check if the ground truth is in the sentence
            if gt in sentence:
                return 1.0
        if sentence == gt or sentence == (gt + ")"):
            return 1.0
    # If no key words found, return 0.0
    return 0.0

