#!/usr/bin/env python
# coding: utf-8

# ### Notebook for finetuning H2o-danube-1.8b-base Model using Pytorch Lightning.  
# 
# You can find more details about the model: 
# 
# **Research paper:** https://arxiv.org/abs/2401.16818
# 
# **Model Huggingface card:** https://huggingface.co/h2oai/h2o-danube-1.8b-base

# #### Inference Notebook: https://www.kaggle.com/code/nischaydnk/h2o-danube-1-8b-llm-submission
# 
# #### Settings to get 0.962+ CV:
# - Training Sequence Length - 1400
# - Downsample competition Data with samples having only 'O' labels with 0.75 ratio
# - Use MPware dataset shared here: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/477989
# 

# # 🚚 Imports

# In[1]:


import os

# 设置no_proxy环境变量
os.environ['no_proxy'] = 'localhost,127.0.0.1'

# 设置http_proxy和https_proxy环境变量
os.environ['http_proxy'] = 'http://10.200.3.253:12798'
os.environ['https_proxy'] = 'http://10.200.3.253:12798'

# 设置REQUESTS_CA_BUNDLE和SSL_CERT_FILE环境变量
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
os.environ['SSL_CERT_FILE'] = '/etc/ssl/certs/ca-certificates.crt'

print('设置成功')


# In[2]:


import os
import gc
from tqdm.auto import tqdm
import json

import numpy as np 
import pandas as pd 
from itertools import chain

from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs
from datasets import concatenate_datasets,load_dataset,load_from_disk

from sklearn.metrics import log_loss

from transformers import AutoModel, AutoTokenizer, AdamW, DataCollatorWithPadding

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from transformers import get_polynomial_decay_schedule_with_warmup,get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers import DataCollatorWithPadding,DataCollatorForTokenClassification


# In[3]:


from datasets import Dataset, load_from_disk
import pickle
import re
from transformers import TrainingArguments, AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification


# # ⚙️ Config

# Notebook was ran on my local Instance, you will need to change the paths for Kaggle accordingly. 

# In[4]:


from dataclasses import dataclass, asdict, field

@dataclass
class CFG:
    # Devices & Seeds
    device: str = 'cuda'
    seed: int = 69
    # Dataset Paths
    train_dataset_path: str = "data/train.json"
    test_dataset_path: str = "data/test.json"
    sample_submission_path: str = "data/sample_submission.csv"
    save_dir: str = "exp/exp"
    # Tokenizer parameters
    downsample: float = 0.75
    truncation: bool = True 
    padding: bool = False  # If False acts as None, for 'max_length' use str 'max_length'
    max_length: int = 1024
    freeze_layers: int = 0
    # Model Parameters
    model_name: str = "h2oai/h2o-danube-1.8b-base"
    target_cols: list = field(default_factory=lambda: ['B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 
                                                       'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 
                                                       'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O'])
    load_from_disk: str = None
    # Training Parameters
    learning_rate: float = 2e-4
    batch_size: int = 1
    epochs: int = 3
    NFOLDS: int = 4
    trn_fold: int = 0
    
    n_lstm_layers: int = 2
    
    def post_init(self):
        if self.device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')



config = CFG()
seed_everything(config.seed)


# In[5]:


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    from pathlib import Path
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


# In[6]:


config.save_dir = increment_path(config.save_dir, mkdir=True)
print(config.save_dir)


# In[7]:


if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)


# # 📊 Preprocessing

# In[8]:


data = json.load(open(config.train_dataset_path))
test_data = json.load(open(config.test_dataset_path))

print('num_samples:', len(data))
print(data[0].keys())


# In[9]:


all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i,l in enumerate(all_labels)}
id2label = {v:k for k,v in label2id.items()}

print(id2label)


# In[10]:


tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(f'{config.save_dir}')


# In[11]:


df_train = pd.DataFrame(data)
df_train.head(5)


# In[12]:


df_mpware = json.load(open('data/mixtral-8x7b-v1.json'))
df_mpware = pd.DataFrame(df_mpware)
df_mpware['document'] =  [i+30000 for i in range(len(df_mpware))]
df_mpware.columns = df_train.columns
df_mpware['fold'] = -1
df_mpware.head(3)


# In[13]:


df_train['fold'] = df_train['document'] % 4
df_train.head(3)


# In[14]:


df_train.shape


# In[ ]:





# In[15]:


def tokenize_row(example):
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


# In[16]:


import pandas as pd

def downsample_df(train_df, percent):

    train_df['is_labels'] = train_df['labels'].apply(lambda labels: any(label != 'O' for label in labels))
    
    true_samples = train_df[train_df['is_labels'] == True]
    false_samples = train_df[train_df['is_labels'] == False]
    
    n_false_samples = int(len(false_samples) * percent)
    downsampled_false_samples = false_samples.sample(n=n_false_samples, random_state=42)
    
    downsampled_df = pd.concat([true_samples, downsampled_false_samples])    
    return downsampled_df


# In[17]:


def add_token_indices(doc_tokens):
    token_indices = list(range(len(doc_tokens)))
    return token_indices

df_train['token_indices'] = df_train['tokens'].apply(add_token_indices)


# In[18]:


df_train.describe()


# In[19]:


get_ipython().run_cell_magic('time', '', 'if config.load_from_disk is None:\n    for i in range(-1, config.NFOLDS):\n        \n        train_df = df_train[df_train[\'fold\']==i].reset_index(drop=True)\n\n        if i==config.trn_fold:\n            config.valid_stride = True\n        if i!=config.trn_fold and config.downsample > 0:\n            train_df = downsample_df(train_df, config.downsample)\n            config.valid_stride = False\n\n        print(len(train_df))\n        ds = Dataset.from_pandas(train_df)\n\n        ds = ds.map(\n          tokenize_row,\n          batched=False,\n          num_proc=2,\n          desc="Tokenizing",\n        )\n\n        ds.save_to_disk(f"{config.save_dir}/fold_{i}.dataset")\n        with open(f"{config.save_dir}/train_pkl", "wb") as fp:\n            pickle.dump(train_df, fp)\n        print("Saving dataset to disk:", config.save_dir)\n\n      \n        \n')


# In[20]:


ds[0].keys()


# # 🔝 Competition Metrics

# In[21]:


def freeze(module):
    """
    Freezes module's parameters.
    """

    
    for parameter in module.parameters():
        parameter.requires_grad = False


# In[22]:


get_ipython().system('pip install comp_metric')


# In[23]:


#import sys
#sys.path.append('/kaggle/input/piimetric')
from comp_metric import compute_metrics


# In[ ]:





# In[24]:


import pandas as pd

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


# # Record

# In[25]:


from pydantic import BaseModel
from typing import List, Dict, Any, Union
from pathlib import Path
import json
import os
import os.path as osp

class Record(BaseModel):
    records: List[Dict[str, Any]] = []
    config: Dict[str, Any]
    save_path: Union[str, Path]

    def add_record(self, metrics: Dict[str, Any]):
        self.records.append(metrics)

    def save(self):
        if "device" in self.config:
            self.config.pop("device")
        if "save_dir" in self.config:
            self.config["save_dir"] = str(self.config["save_dir"])
        print(osp.join(self.save_path, "record.json"))
        with open(osp.join(self.save_path, "record.json"), "w") as f:
            # Since CFG can now be directly converted to dict, we use it here
            json.dump({"records": self.records, "config": self.config}, f)

    @classmethod
    def load(cls, file_path: str):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                return cls(**data, save_path=file_path)
        else:
            raise FileNotFoundError(f"No record found at {file_path}")


            
record = Record(config=asdict(config), save_path=config.save_dir)


# # 🧠 Model

# In[26]:


def process_predictions(flattened_preds, threshold=0.9):
    
    preds_final = []
    for predictions in flattened_preds:
        
        predictions_softmax = torch.softmax(predictions, dim=-1)        
        predictions_argmax = predictions.argmax(-1)
        predictions_without_O = predictions_softmax[ :, :12].argmax(-1)
        
        O_predictions = predictions_softmax[ :, 12]
        pred_final = torch.where(O_predictions < threshold, predictions_without_O, predictions_argmax)        
        preds_final.append(pred_final)
    
    return preds_final



# In[27]:


from peft import get_peft_model, LoraConfig, TaskType
from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import TokenClassifierOutput

class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        return sequence_output


# In[29]:


import random

class LSTMHead(nn.Module):
    def __init__(self, in_features, hidden_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(in_features,
                            hidden_dim,
                            n_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.1)
        self.out_features = hidden_dim

    def forward(self, x):
        self.lstm.flatten_parameters()
        hidden, (_, _) = self.lstm(x)
        out = hidden
        return out

        self.true_val_df = true_val_df
        self.model_config = AutoConfig.from_pret
        

        
CNT = 0
    
class PIIModel(pl.LightningModule):
    def __init__(self,config, val_ds,true_val_df, record=None):
        super().__init__()
        self.cfg = config
        self.val_ds = val_ds
        self.true_val_df = true_val_df
        self.model_config = AutoConfig.from_pretrained(
            config.model_name,
        )

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7
        self.model_config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )

        self.transformers_model = LlamaForTokenClassification.from_pretrained(
            config.model_name, num_labels=len(self.cfg.target_cols), id2label=id2label, label2id=label2id, 
        )
        peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.0)
        self.transformers_model = get_peft_model(self.transformers_model, peft_config)
        self.transformers_model.gradient_checkpointing_enable()  
        self.transformers_model.print_trainable_parameters()
        self.head = LSTMHead(in_features=self.model_config.hidden_size, hidden_dim=self.model_config.hidden_size//2, n_layers=self.cfg.n_lstm_layers)
        self.output = nn.Linear(self.model_config.hidden_size, len(self.cfg.target_cols))
        
        self.record = record
        self.loss_function = nn.CrossEntropyLoss(reduction='mean',ignore_index=-100) 
        self.validation_step_outputs = []


    def forward(self, input_ids, attention_mask,train):
        
        transformer_out = self.transformers_model(input_ids,attention_mask = attention_mask)#[0]
        sequence_output = self.head(transformer_out)
        logits = self.output(sequence_output)
        
        return (logits, _)
    

    def training_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['labels'] 

        outputs = self(input_ids,attention_mask,train=True)
        output = outputs[0]
        loss = self.loss_function(output.view(-1,len(self.cfg.target_cols)), target.view(-1))
        
        self.log('train_loss', loss , prog_bar=True)
        return {'loss': loss}
    
    def train_epoch_end(self,outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print(f'epoch {trainer.current_epoch} training loss {avg_loss}')
        return {'train_loss': avg_loss} 
    
    def validation_step(self,batch,batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['labels'] 

        outputs = self(input_ids,attention_mask,train=False)
        output = outputs[0]

        loss = self.loss_function(output.view(-1,len(self.cfg.target_cols)), target.view(-1))
        
        self.log('val_loss', loss , prog_bar=True)
        self.validation_step_outputs.append({"val_loss": loss, "logits": output, "targets": target})
        return {'val_loss': loss, 'logits': output,'targets':target}        

    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        flattened_preds = [logit for batch in outputs for logit in batch['logits']]

        flattened_preds = process_predictions(flattened_preds)
        # print(flattened_preds.shape)
        pred_df = predictions_to_df(flattened_preds, self.val_ds)
        
        #print(pred_df.shape)
        #print(pred_df)
        
        self.validation_step_outputs = []

        # print(output_val.shape)
        metrics = compute_metrics(pred_df,self.true_val_df)
        self.record.add_record(metrics)
        self.record.save()
        f5_score = metrics['ents_f5']
        print(f'epoch {trainer.current_epoch} validation loss {avg_loss}')
        print(f'epoch {trainer.current_epoch} validation scores {metrics}')
        
        return {'val_loss': avg_loss,'val_f5':f5_score}
    
    def on_save_checkpoint(self, checkpoint):
        global CNT
        # not work
        # trainable_weights = {
        #     k: v for k, v 
        #     in checkpoint["state_dict"].items()
        #     if self.get_parameter(k).requires_grad
        # }
        # torch.save(trainable_weights, f"{CNT}.pth")
        # print([k for k, v in trainable_weights.items()])
        # CNT += 1
        # state = trainable_weights
        
        state = checkpoint["state_dict"]
        for name in list(state.keys()):
            if not self.get_parameter(name).requires_grad:
            #if "lora" not in name:  # <-- adapt the condition to your use case
                state.pop(name)
        print(state.keys())
        return state
    
        
    def train_dataloader(self):
        return self._train_dataloader 
    
    def validation_dataloader(self):
        return self._validation_dataloader

    def get_optimizer_params(self, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in self.transformers_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in self.transformers_model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in self.named_parameters() if "transformers_model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr = config.learning_rate)

        epoch_steps = self.cfg.data_length
        batch_size = self.cfg.batch_size

        warmup_steps = 0.0 * epoch_steps // batch_size
        training_steps = self.cfg.epochs * epoch_steps // batch_size
        # scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,training_steps,-1)
        # scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end=1e-6, power=3.0)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps, num_cycles=1)
        
        lr_scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}


# In[30]:


collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=512)


# In[31]:


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
    


# In[32]:


from torch.utils.data import DataLoader, Subset

print(config.save_dir)

for fold in range(-1, config.NFOLDS):
    if fold != config.trn_fold:
        continue
    train_ds_list = []


    print(f"====== FOLD RUNNING {fold}======")

    
    for i in range(-1, config.NFOLDS):
        if i == fold:
            continue
        if len(train_ds_list) >= 0:
            print(len(train_ds_list))
            train_ds_list.append(load_from_disk(f'{config.save_dir}/fold_{i}.dataset'))

    keep_cols = {"input_ids", "attention_mask", "labels"}
    train_ds = concatenate_datasets(train_ds_list).sort("length") #.select([i for i in range(30)])

    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    valid_ds = load_from_disk(f'{config.save_dir}/fold_{fold}.dataset').sort("length")
    valid_ds = valid_ds.remove_columns([c for c in valid_ds.column_names if c not in keep_cols])
    val_ds = load_from_disk(f'{config.save_dir}/fold_{fold}.dataset').sort("length")

    true_val_df = create_val_df(df_train, fold)
    
    config.data_length = len(train_ds)
    config.len_token = len(tokenizer)
    # swa_callback = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=None, 
                                                              # annealing_epochs=1, annealing_strategy='cos', 
                                                              # avg_fn=None, device="cuda")
    print('Dataset Loaded....')
    print((train_ds[0].keys()))
    print((valid_ds)[0].keys())
    
    # subset
    train_ds = Subset(train_ds, list(range(50)))
    valid_ds = Subset(valid_ds, list(range(10)))

    
    print("Generating Train DataLoader")
    train_dataloader = DataLoader(train_ds, batch_size = config.batch_size, shuffle = True, num_workers= 4, pin_memory=False,collate_fn = collator)
    
    print("Generating Validation DataLoader")
    validation_dataloader = DataLoader(valid_ds, batch_size = config.batch_size, shuffle = False, num_workers= 4, pin_memory=False,collate_fn = collator)

    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=8, verbose= True, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath= config.save_dir,
                                      save_top_k=1,
                                      save_last= False,
                                      save_weights_only=True,
                                      filename= f'checkpoint_{fold}',
                                      verbose= True,
                                      mode='min')
    
    print("Model Creation")
    

    model = PIIModel(config, val_ds,true_val_df, record)
    # model.load_state_dict(torch.load('/home/nischay/PID/nbs/outputs2/exp12_baseline_debv3base_1024_extv1/ckeckpoint_0-v2.ckpt','cpu')['state_dict'])
    trainer = Trainer(max_epochs= config.epochs,
                      deterministic=True,
                      val_check_interval=0.5,
                      accumulate_grad_batches=2, 
                      devices=[0],
                      precision=16, 
                      accelerator="gpu" ,
                      callbacks=[checkpoint_callback,early_stop_callback])    
    # print("Trainer Starting")
    trainer.fit(model , train_dataloader , validation_dataloader)  

    print("prediction on validation data")
    #print("best_score:", early_stop_callback.state_dict["best_model_score"])

    
    del model,train_dataloader,validation_dataloader,train_ds,valid_ds
    gc.collect()
    torch.cuda.empty_cache()
    


