import re

def remove_special_characters(batch, chars_to_remove_regex):
    # remove special characters
    batch["sentence"] = re.sub(
        chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}
