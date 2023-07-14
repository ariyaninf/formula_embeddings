import math

from transformers import BertModel, AlbertModel, DistilBertModel
from tqdm import tqdm
from src.loader import *
import gc
import numpy as np
import torch
import torch.nn as nn
import nltk

nltk.download('punkt')


class Bert_OrderEmb(torch.nn.Module):
    def __init__(self, config):
        super(Bert_OrderEmb, self).__init__()
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

        if bert_pooling == 'cls':
            for tok_id, mask_id in zip(token_ids, input_masks):
                outputs = self.bert(input_ids=tok_id, attention_mask=mask_id)
                last_hidden_states = outputs.hidden_states[-1]
                pooled = last_hidden_states[:, 0, :]
                # print('pooled: ', pooled)
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
                # pooled = (last_hidden_states.sum(axis=1) * mask_id.unsqueeze(-1))/mask_id.sum(axis=1).unsqueeze(1)

                mask_id_expanded = mask_id.unsqueeze(-1).expand(last_hidden_states.size())
                sum_embeddings = torch.sum(last_hidden_states * mask_id_expanded, 1)
                sum_mask = mask_id_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                pooled = sum_embeddings / sum_mask

                if sent_pooling == 'max':
                    p_max_emb, p_max_idx = pooled.max(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_max_emb.unsqueeze(0)], dim=0)
                elif sent_pooling == 'mean':
                    p_mean_emb = pooled.mean(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_mean_emb.unsqueeze(0)], dim=0)
                elif sent_pooling == 'min':
                    p_min_emb, p_min_idx = pooled.min(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_min_emb.unsqueeze(0)], dim=0)

        elif bert_pooling == 'max':
            for tok_id, mask_id in zip(token_ids, input_masks):
                outputs = self.bert(input_ids=tok_id, attention_mask=mask_id)
                last_hidden_states = outputs.hidden_states[-1]

                input_mask_expanded = mask_id.unsqueeze(-1).expand(last_hidden_states.size())
                embeddings = last_hidden_states.clone()
                embeddings[input_mask_expanded == 0] = -1e4
                pooled, _ = torch.max(embeddings, dim=1)

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
            # print('ei: ', penalty)

            # Predicts the entailments
            if penalty <= self.threshold:
                x = torch.tensor([1]).to(device)
            else:
                x = torch.tensor([0]).to(device)

            outputs = torch.cat([outputs, x], dim=0)

        del e_pos, e_neg, pos_labels, neg_labels, x
        return err_loss, outputs


class Bert_OrderEmb_Boosting(torch.nn.Module):
    def __init__(self, config):
        super(Bert_OrderEmb_Boosting, self).__init__()
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
        self.max_seq_length = config.max_seq_length
        self.m = config.error_margin
        self.threshold = config.threshold
        self.register_buffer('model', None)

    def forward(self, token_ids1, input_masks1, token_ids2, input_masks2, labels, weights):

        emb_in1 = self.get_sent_emb_premises(token_ids1, input_masks1)
        emb_in2 = self.get_sent_emb(token_ids2, input_masks2)

        if torch.cuda.is_available():
            labels = torch.tensor(labels).to("cuda")

        loss, preds = self.en_loss(emb_in1, emb_in2, labels, weights)
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

        del token_ids, last_hidden_states
        del input_masks
        return pooled

    def en_loss(self, trans_in1, trans_in2, en_labels, weights):
        batch_size = trans_in1.shape[0]
        t_labels = en_labels
        device = trans_in1.device
        outputs = torch.tensor([]).to(device)
        e = torch.tensor([]).to(device)
        x = torch.tensor([]).to(device)

        pos_labels = torch.max(en_labels, torch.tensor([0.]).to(device))
        neg_labels = torch.abs(torch.sub(torch.tensor([1.]).to(device), pos_labels, alpha=1))

        e = calc_penalty_pos(trans_in1, trans_in2)

        e_pos = torch.mul(e, pos_labels)
        e_pos = torch.mul(e_pos, weights)
        err_pos = torch.sum(e_pos, dim=0)

        e_neg = torch.sub(torch.tensor([self.m]).to(device), e, alpha=1)
        e_neg = torch.max(e_neg, torch.tensor([0.]).to(device))
        e_neg = torch.mul(e_neg, neg_labels)
        e_neg = torch.mul(e_neg, weights)
        err_neg = torch.sum(e_neg, dim=0)

        err_loss = err_pos + err_neg
        # print('err_loss: ', err_pos, ' + ', err_neg, ' = ', err_loss)

        for i in range(0, batch_size):
            penalty = e[i]
            # print('ei: ', penalty)

            # Predicts the entailments
            if penalty <= self.threshold:
                x = torch.tensor([1]).to(device)
            else:
                x = torch.tensor([-1]).to(device)

            outputs = torch.cat([outputs, x], dim=0)

        del e_pos, e_neg, pos_labels, neg_labels, x
        return err_loss, outputs

    def get_final_prediction(self, alphas, model_names, df_samples, tokenizer, use_gpu):
        """
        Get predictions based on calculations of len(alphas) based models
        """
        f_result = "v6_order_embeddings/preds_model.csv"
        w_i = np.ones(len(df_samples)) * 1 / len(df_samples)
        L = len(alphas)
        # Initialise dataframe with weak predictions for each observation
        weak_preds = pd.DataFrame(index=range(len(df_samples)), columns=range(L))

        device = torch.device('cpu')
        if use_gpu:
            device = torch.device('cuda')

        arr_premises, arr_hypos, arr_labels, arr_num_clauses = get_dataframe_arrays(df_samples)
        df_samples = df_samples.drop(columns=['sentence1', 'sentence2'])

        loader = dataloader(df_samples, self.batch_size)

        for l in range(L):
            print(model_names[l])
            self.load_state_dict(torch.load(model_names[l]))
            self.eval()

            arr_preds = np.zeros(len(df_samples))

            for step, batch in enumerate(tqdm(loader, desc="Predictions")):
                indices = batch[0][:, 0].to(device).int()
                indices = indices.cpu().detach().numpy()
                labels = batch[0][:, 1].to(device).float()

                sent_premises = np.array(arr_premises)[indices - 1]
                sent_hypos = np.array(arr_hypos)[indices - 1]
                num_cl = np.array(arr_num_clauses)[indices - 1]
                b_w_i = np.array(w_i)[indices - 1]
                b_w_i = torch.from_numpy(b_w_i)
                b_w_i = b_w_i.to(device).float()
                max_num_cl = max(num_cl)

                seq_prm = tokenize_mask(sent_premises, self.max_seq_length, tokenizer, max_num_cl, 'premises')
                seq_hyp = tokenize_mask(sent_hypos, self.max_seq_length, tokenizer, max_num_cl, 'hypos')

                token_ids_prm, input_masks_prm = convert_tuple_to_tensor(seq_prm, use_gpu)
                token_ids_hyp, input_masks_hyp = convert_tuple_to_tensor(seq_hyp, use_gpu)

                with torch.no_grad():
                    loss, preds = self.forward(token_ids_prm, input_masks_prm, token_ids_hyp, input_masks_hyp, labels,
                                               b_w_i)

                preds = preds.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()

                for idx, y_pred in zip(indices, preds):
                    arr_preds[idx - 1] = y_pred

            y_pred_l = arr_preds * alphas[l]

            weak_preds.iloc[:, l] = y_pred_l

        # Calculate the final predictions
        y_preds = np.sign(weak_preds.sum(axis=1))

        weak_preds.to_csv(str(f_result), sep=';', encoding='utf-8', index=False)

        del df_samples
        gc.collect()

        return y_preds


class Bert_OrderEmb_Binary_LossFunc(torch.nn.Module):
    def __init__(self, config):
        super(Bert_OrderEmb_Binary_LossFunc, self).__init__()
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

        if bert_pooling == 'cls':
            for tok_id, mask_id in zip(token_ids, input_masks):
                outputs = self.bert(input_ids=tok_id, attention_mask=mask_id)
                last_hidden_states = outputs.hidden_states[-1]
                pooled = last_hidden_states[:, 0, :]
                # print('pooled: ', pooled)
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
                # pooled = (last_hidden_states.sum(axis=1) * mask_id.unsqueeze(-1))/mask_id.sum(axis=1).unsqueeze(1)

                mask_id_expanded = mask_id.unsqueeze(-1).expand(last_hidden_states.size())
                sum_embeddings = torch.sum(last_hidden_states * mask_id_expanded, 1)
                sum_mask = mask_id_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                pooled = sum_embeddings / sum_mask

                if sent_pooling == 'max':
                    p_max_emb, p_max_idx = pooled.max(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_max_emb.unsqueeze(0)], dim=0)
                elif sent_pooling == 'mean':
                    p_mean_emb = pooled.mean(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_mean_emb.unsqueeze(0)], dim=0)
                elif sent_pooling == 'min':
                    p_min_emb, p_min_idx = pooled.min(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_min_emb.unsqueeze(0)], dim=0)

        elif bert_pooling == 'max':
            for tok_id, mask_id in zip(token_ids, input_masks):
                outputs = self.bert(input_ids=tok_id, attention_mask=mask_id)
                last_hidden_states = outputs.hidden_states[-1]

                input_mask_expanded = mask_id.unsqueeze(-1).expand(last_hidden_states.size())
                embeddings = last_hidden_states.clone()
                embeddings[input_mask_expanded == 0] = -1e4
                pooled, _ = torch.max(embeddings, dim=1)

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

        bin_f = calc_binary(trans_in1, trans_in2)

        pos_labels = en_labels
        neg_labels = torch.abs(torch.sub(torch.tensor([1.]).to(device), en_labels, alpha=1))

        e = calc_penalty_pos(trans_in1, trans_in2)

        e_pos = torch.mul(e, pos_labels)
        err_pos = torch.sum(e_pos, dim=0)

        e_neg = torch.sub(torch.tensor([self.m]).to(device), e, alpha=1)
        e_neg = torch.max(e_neg, torch.tensor([0.]).to(device))
        e_neg = torch.mul(e_neg, neg_labels)
        err_neg = torch.sum(e_neg, dim=0)

        err_loss = err_pos + err_neg + bin_f
        # err_loss = err_pos + err_neg
        # print('err_loss: ', err_pos, ' + ', err_neg, ' = ', err_loss)

        for i in range(0, batch_size):
            penalty = e[i]
            # print('ei: ', penalty)

            # Predicts the entailments
            if penalty <= self.threshold:
                x = torch.tensor([1]).to(device)
            else:
                x = torch.tensor([0]).to(device)

            outputs = torch.cat([outputs, x], dim=0)

        del e_pos, e_neg, pos_labels, neg_labels, x
        return err_loss, outputs


class Bert_OrderEmb_HL(torch.nn.Module):
    def __init__(self, config):
        super(Bert_OrderEmb_HL, self).__init__()
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
        self.hidden_layer = config.hidden_layer
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
                if sent_pooling == 'max':
                    p_max_emb, p_max_idx = pooled.max(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_max_emb.unsqueeze(0)], dim=0)
                elif sent_pooling == 'mean':
                    p_mean_emb = pooled.mean(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_mean_emb.unsqueeze(0)], dim=0)
                elif sent_pooling == 'min':
                    p_min_emb, p_min_idx = pooled.min(axis=0)
                    pm_emb_all = torch.cat([pm_emb_all, p_min_emb.unsqueeze(0)], dim=0)
        if self.hidden_layer == 2:  # for model V13
            h1 = self.relu(pm_emb_all)
            h1 = self.relu(h1)
        else:
            h1 = self.relu(pm_emb_all)
        del token_ids, input_masks, pooled, pm_emb_all
        gc.collect()
        return h1

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

        del token_ids, last_hidden_states
        del input_masks
        gc.collect()
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
        # print('e: ', e)
        # print('t_labels: ', t_labels)

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
    # print('bin_x: ', bin_x)
    return bin_x


def tokenize_mask_clauses(clauses, max_seq_len, tokenizer, max_len_sen, clauses_type):
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    arr_t_id = []
    arr_i_mask = []

    sentences = nltk.tokenize.sent_tokenize(clauses)
    token_new = tokenizer(sentences, padding='max_length', max_length=max_seq_len)
    arr_t_id = token_new['input_ids']
    arr_i_mask = token_new['attention_mask']

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
    # print('sentences len: ', len(sentences))
    i = 0
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
