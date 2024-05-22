from transformers import BertModel
# from tqdm import tqdm
# from src.loader import *
# import gc
import numpy as np
import torch
# import torch.nn as nn
import nltk

nltk.download('punkt')


class BiEncoder_OrderEmb(torch.nn.Module):
    def __init__(self, config):
        super(BiEncoder_OrderEmb, self).__init__()
        self.bert_prem = BertModel.from_pretrained('cache/' + config.bert_version, output_hidden_states=True,
                                                   local_files_only=True)
        self.bert_hypo = BertModel.from_pretrained('cache/' + config.bert_version, output_hidden_states=True,
                                                   local_files_only=True)
        self.bert_pooling = config.bert_pooling  # cls, mean
        self.sent_pooling = config.sent_pooling  # max, mean, min
        self.batch_size = config.batch_size
        self.m = config.error_margin
        self.threshold = config.threshold
        self.register_buffer('model', None)

    def forward(self, token_ids1, input_masks1, token_ids2, input_masks2, labels):

        emb_premise = self.get_embeddings(token_ids1, input_masks1, is_premises=True)
        emb_hypothesis = self.get_embeddings(token_ids2, input_masks2, is_premises=False)

        if torch.cuda.is_available():
            labels = torch.tensor(labels).to("cuda")

        loss, preds = self.en_loss(emb_premise, emb_hypothesis, labels)
        return loss, preds

    def get_embeddings(self, token_ids, input_masks, is_premises):
        # get the embedding from BERT's last layer
        bert_pooling = self.bert_pooling

        token_ids = token_ids[0::]
        token_ids = token_ids.squeeze(dim=1)

        input_masks = input_masks[0::]
        input_masks = input_masks.squeeze(dim=1)

        if is_premises:
            outputs = self.bert_prem(input_ids=token_ids, attention_mask=input_masks)
        else:
            outputs = self.bert_hypo(input_ids=token_ids, attention_mask=input_masks)
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

        del token_ids, last_hidden_states
        del input_masks
        return pooled

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

            # Predicts the entailments
            if penalty <= self.threshold:
                x = torch.tensor([1]).to(device)
            else:
                x = torch.tensor([0]).to(device)

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


def tokenize_mask(sentences, max_seq_len, tokenizer):
    token_ids = []
    input_mask = []
    i = 0
    for sent in sentences:  # for each row in batch
        t_id, i_mask = tokenize_mask_clauses(sent, max_seq_len, tokenizer)
        token_ids.append(t_id)
        input_mask.append(i_mask)
    return token_ids, input_mask


def tokenize_mask_clauses(clauses, max_seq_len, tokenizer):
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    arr_t_id = []
    arr_i_mask = []

    sentences = nltk.tokenize.sent_tokenize(clauses)
    tokens = [cls_token]
    for sent in sentences:
        tokens += tokenizer.tokenize(sent)
        tokens += [sep_token]

    # if tokens is longer than max_seq_len
    if len(tokens) >= max_seq_len:
        tokens = tokens[:max_seq_len - 2] + [sep_token]
        print('halooooooooooooooo')
    assert len(tokens) <= max_seq_len

    t_id = tokenizer.convert_tokens_to_ids(tokens)

    # add padding
    padding = [0] * (max_seq_len - len(t_id))
    i_mask = [1] * len(t_id) + padding
    t_id += padding

    arr_t_id.append(t_id)
    arr_i_mask.append(i_mask)

    return arr_t_id, arr_i_mask


def convert_tuple_to_tensor(input_tuple, use_gpu=False):
    token_ids, input_masks = input_tuple
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)
    if use_gpu:
        token_ids = token_ids.to('cuda')
        input_masks = input_masks.to('cuda')
    return token_ids, input_masks


class EarlyStopping:
    def __init__(self, patience=3, verbose=True, delta=1e-7, path='checkpoint.pt', trace_func=print):
        self.patience = int(patience)
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
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class EarlyStopping_by_losses:
    def __init__(self, patience=3, verbose=True, delta=1e-2, path='checkpoint.pt', trace_func=print):
        self.patience = int(patience)
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, train_loss, val_loss, model):
        if (val_loss - train_loss) < self.delta:
            self.save_checkpoint(train_loss, val_loss, model)
            self.early_stop = True
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, train_loss, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'Validation loss and train loss < delta ({train_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
