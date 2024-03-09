import os
import fire
from dataclasses import dataclass, asdict, field

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


from datasets import Dataset, load_from_disk
import pickle
import re
from transformers import TrainingArguments, AutoConfig, AutoModelForTokenClassification, DataCollatorForTokenClassification
from torch.utils.data import DataLoader, Subset

import sys
sys.path.append("/root/autodl-tmp/PII")

from .utils.utils import increment_path
from .model.process import *
from .model.model import PIIModel
from .utils.record import Record



def prepare_proxy():
    os.environ['no_proxy'] = 'localhost,127.0.0.1'
    os.environ['https_proxy'] = 'http://10.200.3.253:12798'
    os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'
    os.environ['SSL_CERT_FILE'] = '/etc/ssl/certs/ca-certificates.crt'
    print("Prepare proxy completed.")
    
    
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


def prepare(config):
    data = json.load(open(config.train_dataset_path))
    test_data = json.load(open(config.test_dataset_path))

    print('num_samples:', len(data))
    print(data[0].keys())



    all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
    label2id = {l: i for i,l in enumerate(all_labels)}
    id2label = {v:k for k,v in label2id.items()}
    print(id2label)


    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f'{config.save_dir}')


    df_train = pd.DataFrame(data)
    df_train.head(5)

    df_mpware = json.load(open('data/mixtral-8x7b-v1.json'))
    df_mpware = pd.DataFrame(df_mpware)
    df_mpware['document'] =  [i+30000 for i in range(len(df_mpware))]
    df_mpware.columns = df_train.columns
    df_mpware['fold'] = -1
    df_mpware.head(3)


    df_train['fold'] = df_train['document'] % 4
    df_train.head(3)
    
    return data, test_data, tokenizer, df_mpware, df_train



def train(config, record):
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



def run(config=None, **kwargs):
    # 如果配置和命令行参数均提供，则更新配置对象
    if config:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        config = CFG(**kwargs)
        
    prepare_proxy()
    seed_everything(config.seed)
    config.save_dir = increment_path(config.save_dir, mkdir=True)
    print("save_dir:", config.save_dir)
    
    
    data, test_data, tokenizer, df_mpware, df_train = prepare(config)
    df_train['token_indices'] = df_train['tokens'].apply(add_token_indices)
    
    record = Record(config=asdict(config), save_path=config.save_dir)
    
    train(config, record)

    

if __name__ == '__main__':
    fire.Fire(run)
