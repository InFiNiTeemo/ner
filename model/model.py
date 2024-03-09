from peft import get_peft_model, LoraConfig, TaskType
from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import TokenClassifierOutput

from comp_metric import compute_metrics


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