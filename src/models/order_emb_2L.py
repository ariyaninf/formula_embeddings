import gc

from transformers import AutoModel
import numpy as np
import os.path
import torch
import torch.nn as nn
import nltk
nltk.download('punkt')


class Bert_OrderEmb_HD_HL1 (torch.nn.Module):
    def __init__(self, config):
        super(Bert_OrderEmb_HD_HL1, self).__init__()
        self.bert = AutoModel.from_pretrained('cache/' + config.bert_version, output_hidden_states=True,
                                              local_files_only=True)
        self.in_dim = self.bert.config.hidden_size  # in_dim = 768
        self.bert_pooling = config.bert_pooling  # cls, mean
        self.sent_pooling = config.sent_pooling
        self.out_dim = config.out_dim
        self.hid_dim1 = 884  # intermediate layer dimension
        self.trans1 = nn.Linear(self.in_dim, self.hid_dim1, bias=True)
        self.layer_out = nn.Linear(self.hid_dim1, self.out_dim, bias=True)
        self.batch_size = config.batch_size
        self.m = config.error_margin
        self.threshold = config.threshold
        self.relu = nn.ReLU()
        self.register_buffer('model', None)

    def forward(self, token_ids1, input_masks1, token_ids2, input_masks2, labels):

        emb_in1 = self.get_sent_emb_premises(token_ids1, input_masks1)
        emb_in2 = self.get_sent_emb(token_ids2, input_masks2)

        if torch.cuda.is_available():
            labels = torch.tensor(labels).to("cuda")

        loss, preds = self.en_loss(emb_in1, emb_in2, labels)
        return loss, preds

    def get_sent_emb_premises(self, token_ids, input_masks):
        # get the embeddings from the BERT's last layer
        bert_pooling = self.bert_pooling
        sent_pooling = self.sent_pooling
        pm_emb_all = torch.tensor([]).to(token_ids.device)

        if bert_pooling is None:
            pooling = 'cls'

        if bert_pooling == 'cls':
            for tok_id, mask_id in zip(token_ids, input_masks):
                outputs = self.bert(input_ids=tok_id, attention_mask=mask_id)
                last_hidden_states = outputs.hidden_states[-1]
                pooled = last_hidden_states[:, 0, :]
                pooled = self.relu(self.trans1(pooled))
                pooled = self.layer_out(pooled)
                if sent_pooling == 'max':
                    p_max_emb, p_max_idx = pooled.max(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_max_emb.unsqueeze(0)], dim=0)
                elif sent_pooling == 'mean':
                    p_mean_emb = pooled.mean(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_mean_emb.unsqueeze(0)], dim=0)
                elif sent_pooling == 'min':
                    p_min_emb, p_min_idx = pooled.min(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_min_emb.unsqueeze(0)], dim=0)

        elif bert_pooling == 'mean':
            for tok_id, mask_id in zip(token_ids, input_masks):
                outputs = self.bert(input_ids=tok_id, attention_mask=mask_id)
                last_hidden_states = outputs.hidden_states[-1]
                pooled = last_hidden_states.sum(axis=1) / mask_id.sum(axis=1).unsqueeze(1)
                pooled = self.relu(self.trans1(pooled))
                pooled = self.layer_out(pooled)
                if sent_pooling == 'max':
                    p_max_emb, p_max_idx = pooled.max(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_max_emb.unsqueeze(0)], dim=0)
                elif sent_pooling == 'mean':
                    p_mean_emb = pooled.mean(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_mean_emb.unsqueeze(0)], dim=0)
                elif sent_pooling == 'min':
                    p_min_emb, p_min_idx = pooled.min(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_min_emb.unsqueeze(0)], dim=0)

        del token_ids, input_masks, pooled
        gc.collect()
        return pm_emb_all

    def get_sent_emb(self, token_ids, input_masks):
        # get the embedding from BERT's last layer
        bert_pooling = self.bert_pooling

        token_ids = token_ids[0::]
        token_ids = token_ids.squeeze(dim=1)

        input_masks = input_masks[0::]
        input_masks = input_masks.squeeze(dim=1)

        outputs = self.bert(input_ids=token_ids, attention_mask=input_masks)
        # print('outputs: ', outputs)
        last_hidden_states = outputs.hidden_states[-1]

        if bert_pooling is None:
            pooling = 'cls'

        if bert_pooling == 'cls':
            pooled = last_hidden_states[:, 0, :]
        elif bert_pooling == 'mean':
            pooled = last_hidden_states.sum(axis=1) / input_masks.sum(axis=1).unsqueeze(1)
        elif bert_pooling == 'max':
            pooled = torch.max((last_hidden_states * input_masks.unsqueeze(-1)), axis=1)
            pooled = pooled.values

        h1 = self.relu(self.trans1(pooled))
        h1 = self.layer_out(h1)

        del token_ids, last_hidden_states, pooled, input_masks
        gc.collect()
        return h1

    def en_loss(self, trans_in1, trans_in2, en_labels):
        batch_size = trans_in1.shape[0]
        t_labels = en_labels
        device = trans_in1.device
        outputs = torch.tensor([]).to(device)
        e = torch.tensor([]).to(device)
        x = torch.tensor([]).to(device)

        pos_labels = en_labels
        neg_labels = torch.abs(torch.sub(torch.tensor([1.]).to(device), en_labels, alpha=1))

        e = calc_penalty_pos(trans_in1, trans_in2)

        e_pos = torch.mul(e, pos_labels)
        err_pos = torch.sum(e_pos, dim=0)

        e_neg = torch.sub(torch.tensor([self.m]).to(device), e, alpha=1)
        e_neg = torch.max(e_neg, torch.tensor([0.]).to(device))
        e_neg = torch.mul(e_neg, neg_labels)
        err_neg = torch.sum(e_neg, dim=0)

        err_loss = err_pos + err_neg
        # print('err_loss: ', err_pos, ' + ', err_neg, ' = ', err_loss)

        for i in range(0, batch_size):
            penalty = e[i]
            # print('ei: ', penalty)

            # Predicts the entailments
            if penalty >= self.threshold:
                x = torch.tensor([0]).to(device)
            else:
                x = torch.tensor([1]).to(device)

            outputs = torch.cat([outputs, x], dim=0)

        del e_pos, e_neg, pos_labels, neg_labels, x
        return err_loss, outputs


def calc_penalty_pos(vec_p, vec_h):
    device = vec_p.device
    pen = torch.tensor([]).to(device)
    pen = torch.sub(vec_p, vec_h, alpha=1)
    pen = torch.max(pen, torch.tensor([0]).to(device))
    pen = torch.square(pen)
    pen = torch.sum(pen, dim=1)
    return pen


def tokenize_mask_clauses(clauses, max_seq_len, tokenizer, max_len_sen, clauses_type):
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    arr_t_id = []
    arr_i_mask = []

    sentences = nltk.tokenize.sent_tokenize(clauses)
    for sent in sentences:
        tokens = [cls_token]
        tokens += tokenizer.tokenize(sent)
        tokens += [sep_token]

        # if tokens is longer than max_seq_len
        if len(tokens) >= max_seq_len:
            tokens = tokens[:max_seq_len - 2] + [sep_token]
        assert len(tokens) <= max_seq_len

        t_id = tokenizer.convert_tokens_to_ids(tokens)
        # add padding
        padding = [0] * (max_seq_len - len(t_id))
        i_mask = [1] * len(t_id) + padding
        t_id += padding

        arr_t_id.append(t_id)
        arr_i_mask.append(i_mask)

    if clauses_type == 'premises':
        if max_len_sen is not None:
            max_len_sen = int(max_len_sen)
            for i in range(max_len_sen - len(arr_t_id)):
                arr_t_id.append([0] * max_seq_len)
                arr_i_mask.append([0] * max_seq_len)

    return arr_t_id, arr_i_mask


def tokenize_mask(sentences, max_seq_len, tokenizer, max_len_sen, clauses_type):
    token_ids = []
    input_mask = []
    for sent in sentences:  # for each row in batch
        t_id, i_mask = tokenize_mask_clauses(sent, max_seq_len, tokenizer, max_len_sen, clauses_type)
        token_ids.append(t_id)
        input_mask.append(i_mask)
    return token_ids, input_mask


def convert_tuple_to_tensor(input_tuple, use_gpu=False):
    token_ids, input_masks = input_tuple
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)
    if use_gpu:
        token_ids = token_ids.to("cuda")
        input_masks = input_masks.to("cuda")
    return token_ids, input_masks


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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
