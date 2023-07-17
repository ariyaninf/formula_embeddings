from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from src.model_v2_cross_encoder import *
from src.loader_v2 import *
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import os.path
import argparse
import logging
import gc
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def init():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--dataset", default='V2_Sent_rp-140k_15preds_balanced', type=str)
    args_parser.add_argument("--dataset_path", default='dataset/Sent_test_ok/', type=str)
    args_parser.add_argument("--bert_version", default='albert_large_v2_uncased', type=str)
    args_parser.add_argument("--batch_size", default=8, type=int)
    args_parser.add_argument("--max_seq_length", default=512, type=int)
    args_parser.add_argument("--lr", default=2e-5, type=float)
    args_parser.add_argument("--weight_decay", default=1e-3, type=float)
    args_parser.add_argument("--lr_schedule", default='linear', type=str)
    args_parser.add_argument("--epochs", default=30, type=int)
    args_parser.add_argument("--bert_pooling", default='cls', type=str)
    args_parser.add_argument("--emb_pooling", default='concat', type=str)
    args_parser.add_argument("--out_dim", default=1, type=int)
    args_parser.add_argument("--model_path", default='output/v2_cross_encoder', type=str)
    args_parser.add_argument("--patience", default=3, type=int)
    args = args_parser.parse_args()
    return args


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Set parameters
    args = init()
    model_path = os.path.join(args.model_path, "v2_" + args.bert_version, str(args.batch_size),
                              args.dataset + args.bert_pooling + args.emb_pooling)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_params = [args.bert_version, str(args.lr), str(args.batch_size), str(args.weight_decay),
                    str(args.bert_pooling), str(args.emb_pooling), str(args.epochs)]

    logging_path = init_logging_path(model_path)
    print(logging_path)
    logging.basicConfig(filename=logging_path, encoding='utf-8', level=logging.INFO)
    logging.info(str(args))

    tokenizer = AutoTokenizer.from_pretrained('cache/' + args.bert_version, local_files_only=True)

    # 1. GPU setting
    logging.info("Setting GPU...")
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    f_name_train = os.path.join(args.dataset_path, args.dataset + "_train.csv")
    f_name_val = os.path.join(args.dataset_path, args.dataset + "_val.csv")

    # 2. Load the dataset and create dataloaders
    logging.info("load datasets...")

    df_train = pd.read_csv(f_name_train, sep=None, engine='python')
    print('df_train.shape: ', df_train.shape)
    df_val = pd.read_csv(f_name_val, sep=None, engine='python')
    print('df_val.shape: ', df_val.shape)

    # print('df_val columns: ', df_val.columns)
    logging.info("copy premises and hypothesis ...")
    # train_premises = df_train.facts.values + " " + df_train.rules.values
    train_premises = df_train.sentence1.values
    train_hypos = df_train.sentence2.values
    logging.info("total samples of train: " + str(len(train_premises)))

    # val_premises = df_val.facts.values + " " + df_val.rules.values
    val_premises = df_val.sentence1.values
    val_hypos = df_val.sentence2.values
    logging.info("total samples of val: " + str(len(val_premises)))

    df_train = df_train.drop(columns=['sentence1', 'sat1', 'sentence2', 'sat2', 'prompts'])
    df_val = df_val.drop(columns=['sentence1', 'sat1', 'sentence2', 'sat2', 'prompts'])

    train_loader = dataloader(df_train, args.batch_size)
    val_loader = dataloader(df_val, args.batch_size)

    del df_train, df_val

    # 4. Build model
    model = BertModel_FineTuned(args)
    tuned_parameters = [{'params': [param for name, param in model.named_parameters()]}]

    optimizer = torch.optim.AdamW(tuned_parameters, lr=args.lr)

    model_file = os.path.join(model_path, "_".join(model_params) + ".pt")
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=model_file, delta=1e-10)
    scheduler = get_linear_schedule_with_warmup(optimizer, len(train_loader) * 2,
                                                int(len(train_loader) * args.epochs))

    # 6. Start training
    device = torch.device("cpu")
    if torch.cuda.is_available():
        model.to('cuda')
        device = torch.device("cuda")

    model.train()
    logging.info("Start training...")
    logging.info("Epoch; t_loss; t_acc; t_precision; t_recall; t_f1score; v_loss; v_acc; v_precision; v_recall; "
                 "v_f1score")
    torch.set_printoptions(threshold=10)

    for epoch in trange(args.epochs, desc="Epoch"):
        train_loss = 0
        avg_train_acc = 0
        avg_train_f1score = 0
        avg_train_precision = 0
        avg_train_recall = 0

        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            tb_indices = batch[0][:, 0].to(device).int()
            tb_indices = tb_indices.cpu().detach().numpy()
            tb_labels = batch[0][:, 1].to(device).float()

            tb_premises = np.array(train_premises)[tb_indices - 1]
            tb_hypos = np.array(train_hypos)[tb_indices - 1]

            seq_in = tokenize_mask(tb_premises, tb_hypos, args.max_seq_length, tokenizer)
            tb_token_ids, tb_input_masks = convert_tuple_to_tensor(seq_in, use_gpu)

            optimizer.zero_grad()
            loss, preds = model(tb_token_ids, tb_input_masks, tb_labels)

            preds = torch.round(torch.sigmoid(preds))
            preds = preds.cpu().detach().numpy()
            tb_labels = tb_labels.cpu().detach().numpy()

            train_f1score = f1_score(tb_labels, preds, average='binary')
            train_acc = accuracy_score(tb_labels, preds)
            train_precision = precision_score(tb_labels, preds, average='binary')
            train_recall = recall_score(tb_labels, preds, average='binary')

            train_loss += loss.item()
            avg_train_f1score += train_f1score
            avg_train_acc += train_acc
            avg_train_precision += train_precision
            avg_train_recall += train_recall

            loss.backward()
            optimizer.step()
            scheduler.step()

            del tb_token_ids, tb_input_masks, tb_labels
            del loss
            gc.collect()

        print('train_loss: ', train_loss)
        print('num steps: ', str(step + 1))
        train_loss /= (step + 1)
        avg_train_acc /= (step + 1)
        avg_train_precision /= (step + 1)
        avg_train_recall /= (step + 1)
        avg_train_f1score /= (step + 1)

        print('epoch: ', epoch, ' train_loss: {:,}'.format(train_loss),
              ' train_acc: {:,}'.format(avg_train_acc),
              ' train_precision: {:,}'.format(avg_train_precision),
              ' train_recall: {:,}'.format(avg_train_recall),
              ' train_f1score: {:,}'.format(avg_train_f1score))
        torch.cuda.empty_cache()

        # validation
        model.eval()
        val_loss = 0
        avg_val_acc = 0
        avg_val_f1score = 0
        avg_val_precision = 0
        avg_val_recall = 0

        for step, batch in enumerate(tqdm(val_loader, desc="validation")):
            v_indices = batch[0][:, 0].to(device).int()
            v_indices = v_indices.cpu().detach().numpy()
            v_labels = batch[0][:, 1].to(device).float()

            v_premises = np.array(val_premises)[v_indices - 1]
            v_hypos = np.array(val_hypos)[v_indices - 1]

            seq_val = tokenize_mask(v_premises, v_hypos, args.max_seq_length, tokenizer)
            v_token_ids, v_input_masks = convert_tuple_to_tensor(seq_val, use_gpu)

            with torch.no_grad():
                loss, preds = model(v_token_ids, v_input_masks, v_labels)

            preds = torch.round(torch.sigmoid(preds))
            preds = preds.cpu().detach().numpy()
            v_labels = v_labels.cpu().detach().numpy()

            print('v_labels type: ', type(v_labels))
            print('preds type: ', type(preds))

            avg_val_acc += accuracy_score(v_labels, preds)
            avg_val_precision += precision_score(v_labels, preds, average='binary')
            avg_val_recall += recall_score(v_labels, preds, average='binary')
            avg_val_f1score += f1_score(v_labels, preds, average='binary')

            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()
            val_loss += loss.item()

            del v_token_ids, v_input_masks, v_labels
            del loss
            gc.collect()

        print('val_loss: ', val_loss)
        print('num steps: ', str(step + 1))
        val_loss /= (step + 1)
        avg_val_acc /= (step + 1)
        avg_val_precision /= (step + 1)
        avg_val_recall /= (step + 1)
        avg_val_f1score /= (step + 1)

        model.train()

        print('epoch: ', epoch + 1, ' val_loss: {:,}'.format(val_loss))
        logging.info("%d;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f", epoch + 1, train_loss,
                     avg_train_acc, avg_train_precision, avg_train_recall, avg_train_f1score,
                     val_loss, avg_val_acc, avg_val_precision, avg_val_recall, avg_val_f1score)

        torch.cuda.empty_cache()

        if epoch > 0:  # 20
            early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logging.info("Early stopping. Model trained.")
            break

    torch.cuda.empty_cache()
