from transformers import BertModel, AlbertModel, DistilBertModel
from src.loader import *
import numpy as np
import torch
import gc
import nltk

nltk.download('punkt')


class Bert_OrderEmb_Clause_Loss(torch.nn.Module):
    def __init__(self, config):
        super(Bert_OrderEmb_Clause_Loss, self).__init__()
        if 'albert' in config.bert_version:
            self.bert = AlbertModel.from_pretrained('cache/' + config.bert_version, output_hidden_states=True,
                                                    local_files_only=True)
        elif 'distilbert' in config.bert_version:
            self.bert = DistilBertModel.from_pretrained('cache/' + config.bert_version, output_hidden_states=True,
                                                        local_files_only=True)
        else:
            self.bert = BertModel.from_pretrained('cache/' + config.bert_version, output_hidden_states=True,
                                                  local_files_only=True)

        self.bert_pooling = config.bert_pooling  # cls, mean
        self.sent_pooling = config.sent_pooling  # max, mean, min
        self.batch_size = config.batch_size
        self.m = config.error_margin
        self.threshold = config.threshold
        self.register_buffer('model', None)

    def forward(self, seq_equ1, seq_equ2, labels, use_gpu):
        token_ids_equ1, input_mask_equ1 = convert_tuple_to_tensor(seq_equ1, use_gpu)
        token_ids_equ2, input_mask_equ2 = convert_tuple_to_tensor(seq_equ2, use_gpu)

        emb_equ_in1, emb_equ_clauses_in1 = self.get_sent_emb_equ_premises(token_ids_equ1, input_mask_equ1)
        emb_equ_in2 = self.get_sent_emb_equ_hypos(token_ids_equ2, input_mask_equ2)

        """
        print('emb_in1 len: ', len(emb_in1))
        print('emb_clauses_in1: ', len(emb_clauses_in1))
        print('emb_equ_in1: ', len(emb_equ_in1))
        print('emb_equ_clauses_in1: ', len(emb_equ_clauses_in1))
        """

        if torch.cuda.is_available():
            labels = torch.tensor(labels).to("cuda")

        loss, preds = self.en_loss(emb_equ_in1, emb_equ_in2, labels)
        return loss, preds

    def get_sent_emb_premises(self, token_ids, input_masks):
        # get the embeddings from the BERT's last layer
        bert_pooling = self.bert_pooling
        sent_pooling = self.sent_pooling
        pm_emb_all = torch.tensor([]).to(token_ids.device)
        pm_emb_clauses_all = torch.tensor([]).to(token_ids.device)

        if bert_pooling is None:
            pooling = 'cls'

        if bert_pooling == 'cls':
            for tok_id, mask_id in zip(token_ids, input_masks):
                outputs = self.bert(input_ids=tok_id, attention_mask=mask_id)
                last_hidden_states = outputs.hidden_states[-1]
                pooled = last_hidden_states[:, 0, :]
                pm_emb_clauses_all = torch.cat([pm_emb_clauses_all, pooled], dim=0)
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
                pm_emb_clauses_all = torch.cat([pm_emb_clauses_all, pooled], dim=0)
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
        return pm_emb_all, pm_emb_clauses_all

    def get_sent_emb_hypos(self, token_ids, input_masks):
        # get the embedding from BERT's last layer
        bert_pooling = self.bert_pooling

        token_ids = token_ids[0::]
        token_ids = token_ids.squeeze(dim=1)

        input_masks = input_masks[0::]
        input_masks = input_masks.squeeze(dim=1)

        outputs = self.bert(input_ids=token_ids, attention_mask=input_masks)
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

        del token_ids, last_hidden_states, input_masks
        gc.collect()
        return pooled

    def get_sent_emb_equ_premises(self, token_ids, input_masks):
        # get the embeddings from the BERT's last layer
        bert_pooling = self.bert_pooling
        sent_pooling = self.sent_pooling
        pm_emb_clauses_all = torch.tensor([]).to(token_ids.device)
        pm_emb_all = torch.tensor([]).to(token_ids.device)

        if bert_pooling is None:
            pooling = 'cls'

        if bert_pooling == 'cls':
            for tok_id, mask_id in zip(token_ids, input_masks):  # batch_size
                pm_emb_clauses_batch = torch.tensor([]).to(token_ids.device)
                for cl_tok_id, cl_mask_id in zip(tok_id, mask_id):  # num clauses
                    outputs = self.bert(input_ids=cl_tok_id, attention_mask=cl_mask_id)
                    last_hidden_states = outputs.hidden_states[-1]
                    pooled = last_hidden_states[:, 0, :]
                    pm_lit_emb, pm_lit_emb_idx = pooled.max(axis=0)  # maximum coordinates in literals vectors
                    pm_emb_clauses_all = torch.cat([pm_emb_clauses_all, pm_lit_emb.unsqueeze(0)], dim=0)
                    pm_emb_clauses_batch = torch.cat([pm_emb_clauses_batch, pm_lit_emb.unsqueeze(0)], dim=0)
                if sent_pooling == 'min':
                    p_min_emb, p_min_idx = pm_emb_clauses_batch.min(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_min_emb.unsqueeze(0)], dim=0)

        elif bert_pooling == 'mean':
            for tok_id, mask_id in zip(token_ids, input_masks):  # batch_size
                pm_emb_clauses_batch = torch.tensor([]).to(token_ids.device)
                for cl_tok_id, cl_mask_id in zip(tok_id, mask_id):  # num clauses
                    outputs = self.bert(input_ids=tok_id, attention_mask=mask_id)
                    last_hidden_states = outputs.hidden_states[-1]
                    pooled = last_hidden_states.sum(axis=1) / mask_id.sum(axis=1).unsqueeze(1)
                    pm_lit_emb, pm_lit_emb_idx = pooled.max(axis=0)  # maximum coordinates in literals vectors
                    pm_emb_clauses_all = torch.cat([pm_emb_clauses_all, pm_lit_emb.unsqueeze(0)], dim=0)
                    pm_emb_clauses_batch = torch.cat([pm_emb_clauses_batch, pm_lit_emb.unsqueeze(0)], dim=0)
                if sent_pooling == 'min':
                    p_min_emb, p_min_idx = pm_emb_clauses_batch.min(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_min_emb.unsqueeze(0)], dim=0)

        del token_ids, input_masks, pooled
        gc.collect()
        return pm_emb_all, pm_emb_clauses_all

    def get_sent_emb_equ_hypos(self, token_ids, input_masks):
        # get the embedding from BERT's last layer
        bert_pooling = self.bert_pooling
        pm_clause_emb = torch.tensor([]).to(token_ids.device)

        if bert_pooling is None:
            bert_pooling = 'cls'

        if bert_pooling == 'cls':
            for tok_id, in_mask in zip(token_ids, input_masks):
                outputs = self.bert(input_ids=tok_id, attention_mask=in_mask)
                last_hidden_states = outputs.hidden_states[-1]
                pooled = last_hidden_states[:, 0, :]
                pm_lit_emb, pm_lit_emb_idx = pooled.max(axis=0)  # maximum coordinates in literals vectors
                pm_clause_emb = torch.cat([pm_clause_emb, pm_lit_emb.unsqueeze(0)], dim=0)
        elif bert_pooling == 'mean':
            for tok_id, in_mask in zip(token_ids, input_masks):
                outputs = self.bert(input_ids=tok_id, attention_mask=in_mask)
                last_hidden_states = outputs.hidden_states[-1]
                pooled = last_hidden_states.sum(axis=1) / in_mask.sum(axis=1).unsqueeze(1)
                pm_lit_emb, pm_lit_emb_idx = pooled.max(axis=0)  # maximum coordinates in literals vectors
                pm_clause_emb = torch.cat([pm_clause_emb, pm_lit_emb.unsqueeze(0)], dim=0)

        del last_hidden_states, pooled
        gc.collect()
        return pm_clause_emb

    def en_loss(self, emb_equ_in1, emb_equ_in2, en_labels):
        batch_size = emb_equ_in1.shape[0]
        t_labels = en_labels
        device = emb_equ_in1.device
        outputs = torch.tensor([]).to(device)
        e = torch.tensor([]).to(device)
        x = torch.tensor([]).to(device)

        pos_labels = en_labels
        neg_labels = torch.abs(torch.sub(torch.tensor([1.]).to(device), en_labels, alpha=1))

        e = calc_penalty_pos(emb_equ_in1, emb_equ_in2)

        e_pos = torch.mul(e, pos_labels)
        err_pos = torch.sum(e_pos, dim=0)

        e_neg = torch.sub(torch.tensor([self.m]).to(device), e, alpha=1)
        e_neg = torch.max(e_neg, torch.tensor([0.]).to(device))
        e_neg = torch.mul(e_neg, neg_labels)
        err_neg = torch.sum(e_neg, dim=0)

        err_loss = err_pos + err_neg
        # err_oe = err_pos + err_neg
        # print('err_oe: ', err_oe)
        # print('e_clauses: ', e_clauses)

        for i in range(0, batch_size):
            penalty = e[i]
            # Predicts the entailments
            if penalty <= self.threshold:
                x = torch.tensor([1]).to(device)
            else:
                x = torch.tensor([0]).to(device)

            outputs = torch.cat([outputs, x], dim=0)

        del e_pos, e_neg, pos_labels, neg_labels, x
        gc.collect()
        return err_loss, outputs


def compute_error(y, y_pred, w_i):
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int))) / sum(w_i)


def compute_alpha(error):
    return np.log((1 - error) / error) * 1/2


def update_weights(w_i, alpha, y, y_pred):
    # v_i = w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
    v_i = w_i * np.exp(-alpha * (np.multiply(y, y_pred)).astype(int))
    return v_i / sum(v_i)


def calc_penalty_pos(vec_p, vec_h):
    device = vec_p.device
    pen = torch.tensor([]).to(device)
    pen = torch.sub(vec_p, vec_h, alpha=1)
    pen = torch.max(pen, torch.tensor([0]).to(device))
    pen = torch.square(pen)
    pen = torch.sum(pen, dim=1)
    return pen


def calc_binary(vec_p, vec_h):
    device = vec_p.device
    x = torch.cat((vec_p, vec_h), dim=0)
    bin_x = torch.tensor([]).to(device)
    bin_x = torch.sub(torch.tensor([1]).to(device), x)
    bin_x = torch.mul(x, bin_x)
    bin_x = torch.sum(bin_x, dim=1)
    bin_x = torch.sum(bin_x, dim=0)
    return bin_x


def calc_clauses_loss(vc_p, vc_equ_p, vc_h, vc_equ_h):
    device = vc_p.device
    a = torch.cat((vc_p, vc_h), dim=0)
    b = torch.cat((vc_equ_p, vc_equ_h), dim=0)
    b_min_a = torch.tensor([]).to(device)
    b_min_a = torch.sub(b, a)
    b_min_a = torch.square(b_min_a)
    # print('b min a: ', torch.sum(b_min_a, dim=1))
    return torch.sum(b_min_a)


def tokenize_mask(sentences, max_seq_len, tokenizer, max_len_sen, clauses_type):
    token_ids = []
    input_mask = []

    for sent in sentences:  # for each row in batch
        t_id, i_mask = tokenize_mask_clauses(sent, max_seq_len, tokenizer, max_len_sen, clauses_type)
        token_ids.append(t_id)
        input_mask.append(i_mask)
    return token_ids, input_mask


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


def tokenize_equ(sentences, max_seq_len, tokenizer, max_len_sen, clauses_type):
    token_ids = []
    input_mask = []

    for sent in sentences:  # for each row in batch
        t_id, i_mask = tokenize_equ_clauses(sent, max_seq_len, tokenizer, max_len_sen, clauses_type)
        token_ids.append(t_id)
        input_mask.append(i_mask)
    return token_ids, input_mask


def tokenize_equ_clauses(clauses, max_seq_len, tokenizer, max_len_sen, clauses_type):
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    arr_t_id = []
    arr_i_mask = []
    arr_lit_id = []
    arr_lit_mask = []

    sentences = nltk.tokenize.sent_tokenize(clauses)
    for sent in sentences:
        literals = sent.split(" or ")
        arr_lit_mask = []
        arr_lit_id = []

        for lit in literals:
            tokens = [cls_token]
            tokens += tokenizer.tokenize(lit)
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
            arr_lit_id.append(t_id)
            arr_lit_mask.append(i_mask)

        if len(arr_lit_id) < 4:
            for i in range(4 - len(arr_lit_id)):
                arr_lit_id.append([0] * max_seq_len)
                arr_lit_mask.append([0] * max_seq_len)

        if clauses_type == 'hypos':
            arr_t_id = arr_lit_id
            arr_i_mask = arr_lit_mask
        else:
            arr_t_id.append(arr_lit_id)
            arr_i_mask.append(arr_lit_mask)

    if clauses_type == 'premises':
        if max_len_sen is not None:
            max_len_sen = int(max_len_sen)
            for i in range(max_len_sen - len(arr_t_id)):
                arr_lit_id = []
                arr_lit_mask = []
                for j in range(4):
                    arr_lit_id.append([0] * max_seq_len)
                    arr_lit_mask.append([0] * max_seq_len)
                arr_t_id.append(arr_lit_id)
                arr_i_mask.append(arr_lit_mask)

    return arr_t_id, arr_i_mask


def convert_tuple_to_tensor(input_tuple, use_gpu=False):
    token_ids, input_masks = input_tuple
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)
    if use_gpu:
        token_ids = token_ids.to('cuda')
        input_masks = input_masks.to('cuda')
    return token_ids, input_masks


def get_dataframe_arrays(df):
    arr_premises = df.sentence1.values
    arr_hypos = df.sentence2.values
    arr_labels = df.label.values
    arr_num_clauses = df.num_facts.values + df.num_rules.values
    return arr_premises, arr_hypos, arr_labels, arr_num_clauses


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
