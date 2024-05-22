from transformers import BertModel, AlbertModel, DistilBertModel
from torch.utils.data import TensorDataset
import os.path
import numpy as np
import torch
import torch.nn as nn
import nltk
nltk.download('punkt')


class BertModel_FineTuned(torch.nn.Module):
    def __init__(self, config):
        super(BertModel_FineTuned, self).__init__()
        self.bert_version = config.bert_version
        if 'albert' in config.bert_version:
            self.bert = AlbertModel.from_pretrained('cache/' + self.bert_version, output_hidden_states=True,
                                                    local_files_only=True)
        elif 'distilbert' in config.bert_version:
            self.bert = DistilBertModel.from_pretrained('cache/' + self.bert_version, output_hidden_states=True,
                                                        local_files_only=True)
        else:
            self.bert = BertModel.from_pretrained('cache/' + self.bert_version, output_hidden_states=True,
                                                  local_files_only=True)

        self.in_dim = self.bert.config.hidden_size  # in_dim = 768
        self.bert_pooling = config.bert_pooling
        self.emb_pooling = config.emb_pooling
        self.out_dim = config.out_dim  # out_dim = 1
        self.layer_out = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        self.relu = nn.ReLU()
        self.batch_size = config.batch_size
        self.register_buffer('model', None)

    def forward(self, token_ids, input_mask, labels):
        emb_in1 = self.get_bert_emb(token_ids, input_mask, self.bert_pooling)
        loss, preds = self.en_loss(emb_in1, labels)
        #  print('loss: ', loss)
        return loss, preds

    def get_bert_emb(self, token_ids, input_masks, pooling):
        # get the embedding from the last layer
        outputs = self.bert(input_ids=token_ids)
        last_hidden_states = outputs.hidden_states[-1]

        if pooling is None:
            pooling = 'cls'
        if pooling == 'cls':
            pooled = last_hidden_states[:, 0, :]
        elif pooling == 'mean':
            pooled = last_hidden_states.sum(axis=1) / input_masks.sum(axis=-1).unsqueeze(-1)
        elif pooling == 'max':
            pooled = torch.max((last_hidden_states * input_masks.unsqueeze(-1)), axis=1)
            pooled = pooled.values

        del token_ids
        del input_masks
        return pooled

    def en_loss(self, trans_in, en_labels):
        batch_size = trans_in.shape[0]
        t_labels = en_labels.float()
        pred_all = torch.tensor([]).to(trans_in.device)

        for i in range(0, batch_size):
            emb_in = trans_in[i]
            x = self.layer_out(emb_in)
            # concatenation
            pred_all = torch.cat([pred_all, x], dim=0)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(pred_all, t_labels)

        del t_labels
        return loss, pred_all


def tokenize_mask_clauses(prem, query, max_seq_len, tokenizer):
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    tokens = [cls_token]
    tokens += tokenizer.tokenize(prem)
    tokens += [sep_token] + tokenizer.tokenize(query) + [sep_token]
    # print('tokens: ', tokens)
    # if tokens is longer than max_seq_len
    if len(tokens) >= max_seq_len:
        tokens = tokens[:max_seq_len - 2] + [sep_token]
    assert len(tokens) <= max_seq_len
    t_id = tokenizer.convert_tokens_to_ids(tokens)
    # add padding
    padding = [0] * (max_seq_len - len(t_id))
    i_mask = [1] * len(t_id) + padding
    t_id += padding
    return t_id, i_mask


def tokenize_mask(premise, query, max_seq_len, tokenizer):
    token_ids = []
    input_mask = []
    i = 0
    for p, q in zip(premise, query):
        t_id, i_mask = tokenize_mask_clauses(p, q, max_seq_len, tokenizer)
        token_ids.append(t_id)
        input_mask.append(i_mask)
        i += 1
    return token_ids, input_mask


def convert_tuple_to_tensor(input_tuple, use_gpu=False):
    token_ids, input_masks = input_tuple
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)
    if use_gpu:
        token_ids = token_ids.to("cuda")
        input_masks = input_masks.to("cuda")
    return token_ids, input_masks


def convert_dataframe_to_tensor(df, max_seq_length, bert_tokenizer, use_gpu):
    train_id = df.iloc[:, 0]
    train_labels = df.label.values
    train_facts = df.facts.values
    train_rules = df.rules.values
    train_query = df.queries.values

    train_seq = tokenize_mask(train_facts, train_rules, train_query, max_seq_length, bert_tokenizer)

    token_ids, input_masks = convert_tuple_to_tensor(train_seq, use_gpu)
    train_id = torch.tensor(train_id, dtype=torch.long)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    dataset = TensorDataset(train_id, token_ids, input_masks, train_labels)
    return dataset

'''
def dataloader(tensor, batch_size):
    sampler = SequentialSampler(tensor)
    loader = DataLoader(dataset=tensor, batch_size=batch_size, sampler=sampler)
    return loader
'''


def init_logging_path(dir_log):
    dir_log = os.path.join(dir_log, f"log/")
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f'log_{len(os.listdir(dir_log)) + 1}.log'
        with open(dir_log, 'w'):
            os.utime(dir_log, None)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
        dir_log += f'log_{len(os.listdir(dir_log)) + 1}.log'
        with open(dir_log, 'w'):
            os.utime(dir_log, None)
    return dir_log


class EarlyStopping:
    def __init__(self, patience=3, verbose=True, delta=1e-7, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # reset counter

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
