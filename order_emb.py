from src.loader import *
from src.model_order_emb import *
from transformers import BertTokenizer, AlbertTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import gc
import argparse
import warnings
warnings.filterwarnings('ignore')


def init():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--dataset", default='Sent_rp-140k_15preds_balanced', type=str)
    args_parser.add_argument("--dataset_path", default='dataset/Sent_RP/', type=str)
    args_parser.add_argument("--batch_size", default=8, type=int)
    args_parser.add_argument("--max_seq_length", default=32, type=int)
    args_parser.add_argument("--bert_version", default='bert_base_uncased', type=str)
    args_parser.add_argument("--lr", default=2e-5, type=float)
    args_parser.add_argument("--weight_decay", default=1e-3, type=float)
    args_parser.add_argument("--lr_schedule", default='linear', type=str)
    args_parser.add_argument("--epochs", default=30, type=int)
    args_parser.add_argument("--bert_pooling", default='cls', type=str)  # cls, mean
    args_parser.add_argument("--sent_pooling", default='min', type=str)  # max, mean, min
    args_parser.add_argument("--out_dim", default=1, type=int)
    args_parser.add_argument("--error_margin", default=2, type=float)
    args_parser.add_argument("--threshold", default=1, type=float)
    args_parser.add_argument("--model_path", default='output/v6_order_embeddings', type=str)
    args_parser.add_argument("--patience", default=5, type=str)
    args = args_parser.parse_args()
    return args


if __name__ == '__main__':
    # seed = random.randint(1000)
    # Set parameters
    args = init()
    model_path = os.path.join(args.model_path, 'v6_' + args.bert_version, str(args.max_seq_length),
                              args.dataset + "_" + args.bert_pooling + "_" + args.sent_pooling
                              + "_" + str(args.error_margin) + "_" + str(args.threshold))

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_params = [str(args.bert_version), "_ordemb"]

    model_params += [str(args.lr), str(args.batch_size), str(args.weight_decay), str(args.bert_pooling),
                     str(args.sent_pooling)]

    logging_path = init_logging_path(model_path)
    print(logging_path)
    logging.basicConfig(filename=logging_path, encoding='utf-8', level=logging.INFO)
    logging.info(str(args))

    if 'albert' in args.bert_version:
        bert_tokenizer = AlbertTokenizer.from_pretrained('cache/' + args.bert_version, local_files_only=True)
    else:
        bert_tokenizer = BertTokenizer.from_pretrained('cache/' + args.bert_version, local_files_only=True)

    # 1. GPU setting
    logging.info("Setting GPU...")
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    # 2. Load the dataset and create dataloaders
    fname_train = os.path.join(args.dataset_path, args.dataset + "_train.csv")
    fname_val = os.path.join(args.dataset_path, args.dataset + "_val.csv")

    logging.info("load datasets...")

    df_train = pd.read_csv(fname_train, sep=None, engine='python')
    df_val = pd.read_csv(fname_val, sep=None, engine='python')

    train_premises = df_train.sentence1.values
    train_hypos = df_train.sentence2.values
    train_num_clauses = df_train.num_facts.values + df_train.num_rules.values

    val_premises = df_val.sentence1.values
    val_hypos = df_val.sentence2.values
    val_num_clauses = df_val.num_facts.values + df_val.num_rules.values

    df_train = df_train.drop(columns=['sentence1', 'sentence2', 'sat1', 'sat2', 'prompts'])
    df_val = df_val.drop(columns=['sentence1', 'sentence2', 'sat1', 'sat2', 'prompts'])
    print('df_train shape: ', df_train.shape)
    print('df_val: ', df_val.shape)

    train_loader = dataloader(df_train, args.batch_size)
    val_loader = dataloader(df_val, args.batch_size)
    del df_train, df_val
    gc.collect()

    # 3. Build model
    model = Bert_OrderEmb(args)

    tuned_parameters = [{'params': [param for name, param in model.named_parameters()]}]

    optimizer = AdamW(tuned_parameters, lr=args.lr)

    model_file = os.path.join(model_path, "_".join(model_params) + ".pt")
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=model_file, delta=1e-10)
    scheduler = get_linear_schedule_with_warmup(optimizer, len(train_loader) * 2,
                                                int(len(train_loader) * args.epochs))

    # 4. Start training
    device = torch.device('cpu')
    if torch.cuda.is_available():
        model.to('cuda')
        device = torch.device('cuda')

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

            sent_premises = np.array(train_premises)[tb_indices - 1]
            sent_hypos = np.array(train_hypos)[tb_indices - 1]
            tb_num_cl = np.array(train_num_clauses)[tb_indices - 1]
            max_num_cl = max(tb_num_cl)

            # print('sent_premises: ', sent_premises)
            # print('sent_hypos: ', sent_hypos)

            seq_in1 = tokenize_mask(sent_premises, args.max_seq_length, bert_tokenizer, max_num_cl, 'premises')
            seq_in2 = tokenize_mask(sent_hypos, args.max_seq_length, bert_tokenizer, max_num_cl, 'hypos')

            token_ids1, input_masks1 = convert_tuple_to_tensor(seq_in1, use_gpu)
            token_ids2, input_masks2 = convert_tuple_to_tensor(seq_in2, use_gpu)

            optimizer.zero_grad()
            loss, preds = model(token_ids1, input_masks1, token_ids2, input_masks2, tb_labels)

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

            del token_ids1, input_masks1, token_ids2, input_masks2, tb_labels
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
            vb_indices = batch[0][:, 0].to(device).int()
            vb_indices = vb_indices.cpu().detach().numpy()
            vb_labels = batch[0][:, 1].to(device).float()

            v_sent_premises = np.array(val_premises)[vb_indices - 1]
            v_sent_hypos = np.array(val_hypos)[vb_indices - 1]
            vb_num_cl = np.array(val_num_clauses)[vb_indices - 1]
            v_max_num_cl = max(vb_num_cl)

            v_seq_in1 = tokenize_mask(v_sent_premises, args.max_seq_length, bert_tokenizer, v_max_num_cl, 'premises')
            v_seq_in2 = tokenize_mask(v_sent_hypos, args.max_seq_length, bert_tokenizer, v_max_num_cl, 'hypos')

            v_token_ids1, v_input_masks1 = convert_tuple_to_tensor(v_seq_in1, use_gpu)
            v_token_ids2, v_input_masks2 = convert_tuple_to_tensor(v_seq_in2, use_gpu)

            with torch.no_grad():
                v_loss, v_preds = model(v_token_ids1, v_input_masks1, v_token_ids2, v_input_masks2, vb_labels)

            val_loss += v_loss.item()
            v_preds = v_preds.cpu().detach().numpy()
            vb_labels = vb_labels.cpu().detach().numpy()

            avg_val_acc += accuracy_score(vb_labels, v_preds)
            avg_val_precision += precision_score(vb_labels, v_preds, average='binary')
            avg_val_recall += recall_score(vb_labels, v_preds, average='binary')
            avg_val_f1score += f1_score(vb_labels, v_preds, average='binary')

            del v_token_ids1, v_input_masks1, v_token_ids2, v_input_masks2, vb_labels
            del v_loss
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
