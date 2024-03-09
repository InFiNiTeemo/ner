def tokenize_row(tokenizer, example):
    text = []
    token_map = []
    labels = []
    targets = []
    idx = 0
    for t, l, ws in zip(example["tokens"], example["labels"], example["trailing_whitespace"]):
        text.append(t)
        labels.extend([l]*len(t))
        token_map.extend([idx]*len(t))

        if l in config.target_cols:  
            targets.append(1)
        else:
            targets.append(0)
        
        if ws:
            text.append(" ")
            labels.append("O")
            token_map.append(-1)
        idx += 1

    if config.valid_stride:
        tokenized = tokenizer("".join(text), return_offsets_mapping=True, padding='longest', truncation=True, max_length=2048)  # Adjust max_length if needed
    else:
        tokenized = tokenizer("".join(text), return_offsets_mapping=True, padding='longest', truncation=True, max_length=config.max_length)  # Adjust max_length if needed
        
    target_num = sum(targets)
    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        if start_idx == 0 and end_idx == 0: 
            token_labels.append(label2id["O"])
            continue
        
        if text[start_idx].isspace():
            start_idx += 1
        try:
            token_labels.append(label2id[labels[start_idx]])
        except:
            continue
    length = len(tokenized.input_ids)
    
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "offset_mapping": tokenized.offset_mapping,
        "labels": token_labels,
        "length": length,
        "target_num": target_num,
        "group": 1 if target_num > 0 else 0,
        "token_map": token_map,
    }
