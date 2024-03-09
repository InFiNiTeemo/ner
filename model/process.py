import pandas as pd

def create_val_df(df, fold):
    val_df = df[df['fold']==fold].reset_index(drop=True).copy()
    
    val_df = val_df[['document', 'tokens', 'labels']].copy()
    val_df = val_df.explode(['tokens', 'labels']).reset_index(drop=True).rename(columns={'tokens': 'token', 'labels': 'label'})
    val_df['token'] = val_df.groupby('document').cumcount()
    
    label_list = val_df['label'].unique().tolist()
    
    reference_df = val_df[val_df['label'] != 'O'].copy()
    reference_df = reference_df.reset_index().rename(columns={'index': 'row_id'})
    reference_df = reference_df[['row_id', 'document', 'token', 'label']].copy()
    return reference_df


def downsample_df(train_df, percent):

    train_df['is_labels'] = train_df['labels'].apply(lambda labels: any(label != 'O' for label in labels))
    
    true_samples = train_df[train_df['is_labels'] == True]
    false_samples = train_df[train_df['is_labels'] == False]
    
    n_false_samples = int(len(false_samples) * percent)
    downsampled_false_samples = false_samples.sample(n=n_false_samples, random_state=42)
    
    downsampled_df = pd.concat([true_samples, downsampled_false_samples])    
    return downsampled_df


def add_token_indices(doc_tokens):
    token_indices = list(range(len(doc_tokens)))
    return token_indices


def backwards_map_preds(sub_predictions, max_len):
    if max_len != 1: # nothing to map backwards if sequence is too short to be split in the first place
        if i == 0:
            # First sequence needs no SEP token (used to end a sequence)
            sub_predictions = sub_predictions[:,:-1,:]
        elif i == max_len-1:
            # End sequence needs to CLS token + Stride tokens 
            sub_predictions = sub_predictions[:,1+STRIDE:,:] # CLS tokens + Stride tokens
        else:
            # Middle sequence needs to CLS token + Stride tokens + SEP token
            sub_predictions = sub_predictions[:,1+STRIDE:-1,:]
    return sub_predictions


def backwards_map_(row_attribute, max_len):
    # Same logics as for backwards_map_preds - except lists instead of 3darray
    if max_len != 1:
        if i == 0:
            row_attribute = row_attribute[:-1]
        elif i == max_len-1:
            row_attribute = row_attribute[1+STRIDE:]
        else:
            row_attribute = row_attribute[1+STRIDE:-1]
    return row_attribute


def predictions_to_df(preds, ds, id2label=id2label):
    triplets = []
    pairs = set()
    document, token, label, token_str = [], [], [], []
    for p, token_map, offsets, tokens, doc in zip(preds, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]):
        # p = p.argmax(-1).cpu().detach().numpy()
        p = p.cpu().detach().numpy()
        
        for token_pred, (start_idx, end_idx) in zip(p, offsets):
            label_pred = id2label[(token_pred)]

            if start_idx + end_idx == 0: continue

            if token_map[start_idx] == -1:
                start_idx += 1

            # ignore "\n\n"
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1

            if start_idx >= len(token_map): 
                break

            
            token_id = token_map[start_idx]

            if label_pred == "O" or token_id == -1:
                continue
            
            pair = (doc, token_id)
    
            if pair in pairs:
                continue

            document.append(doc)
            token.append(token_id)
            label.append(label_pred)
            token_str.append(tokens[token_id])
            pairs.add(pair)
                
    df = pd.DataFrame({
        "document": document,
        "token": token,
        "label": label,
        "token_str": token_str
    })
    df["row_id"] = list(range(len(df)))
    return df