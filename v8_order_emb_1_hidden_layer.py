from src.loader import *
from src.model_v8_rplp import *
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')


def init():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--dataset", default="Sent_140k_10preds_prop_examples.balanced_by_backward_max_6", type=str)
    args_parser.add_argument("--dataset_path", default="", type=str)
    args_parser.add_argument("--bert_version", default="bert_base_uncased", type=str)
    args_parser.add_argument("--batch_size", default=8, type=int)
    args_parser.add_argument("--max_seq_length", default=16, type=int)
    args_parser.add_argument("--lr", default=2e-5, type=float)
    args_parser.add_argument("--weight_decay", default=1e-3, type=float)
    args_parser.add_argument("--lr_schedule", default="linear", type=str)
    args_parser.add_argument("--epochs", default=100, type=int)
    args_parser.add_argument("--bert_pooling", default="cls", type=str)  # cls, mean
    args_parser.add_argument("--sent_pooling", default="min", type=str)  # max, mean
    args_parser.add_argument("--mode_train", default="RP", type=str)  # RP, LP, RPLP
    args_parser.add_argument("--error_margin", default=2, type=float)
    args_parser.add_argument("--threshold", default=1, type=float)
    args_parser.add_argument("--out_dim", default=1000, type=int)
    args_parser.add_argument("--model_path", default="v8_order_hiddim_24h_rplp", type=str)
    args_parser.add_argument("--patience", default=3, type=int)
    args = args_parser.parse_args()
    return args


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Set parameter
    args = init()

    model_path = os.path.join(args.model_path, "v8_" + args.bert_version + "_hl_hd",
                              str(args.max_seq_length), args.mode_train + "_" + args.dataset + "_" + args.bert_pooling
                              + "_" + args.sent_pooling + "_" + str(args.error_margin) + "_" + str(args.out_dim))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_params = [str(args.bert_version), "_ordemb_hl1_hd"]

    model_params += [str(args.lr), str(args.batch_size), str(args.weight_decay), str(args.bert_pooling),
                     str(args.sent_pooling)]

    logging_path = init_logging_path(model_path)
    print(logging_path)
    logging.basicConfig(filename=logging_path, encoding='utf-8', level=logging.INFO)
    logging.info(str(args))

    fname_path = args.dataset_path
    fname_train = os.path.join(fname_path, str(args.dataset) + "_train.csv")
    fname_val = os.path.join(fname_path, str(args.dataset) + "_val.csv")
    # fname_test = os.path.join(fname_path, str(args.dataset) + "_test.csv")

    df_train = pd.read_csv(fname_train, sep=None, engine="python")
    print('df_train.shape: ', df_train.shape)

    df_val = pd.read_csv(fname_val, sep=None, engine="python")
    print('df_val.shape: ', df_val.shape)

    bert_tokenizer = AutoTokenizer.from_pretrained('cache/' + args.bert_version, local_files_only=True)

    # 1. Build dataset for contrastive learning per batch_size, id_set, and mode.
    # Splits the main dataset into train:eval:test = 8:1:1
    logging.info("build datasets...")
    msg = '(x_train, x_val): ' + str(len(df_train)) + ', ' + str(len(df_val))
    logging.info(msg)

    # 2. Extract sentences and labels
    train_labels = df_train.label.values
    # load sentences
    logging.info("load sentences ...")
    main_sentences = df_train["sentence1"].tolist()
    logging.info("total number of main sentences: " + str(len(main_sentences)))
    pair_sentences = df_train["sentence2"].tolist()
    logging.info("total number of pair sentences: " + str(len(pair_sentences)))
    len_sentences = np.add(df_train["num_rules"].tolist(), df_train["num_facts"].tolist())
    cl_labels = df_train["label"].tolist()
    logging.info("total number of labels: " + str(len(cl_labels)))

    val_labels = df_val.label.values
    val_main_sentences = df_val["sentence1"].tolist()
    logging.info("total number of val main sentences: " + str(len(val_main_sentences)))
    val_pair_sentences = df_val["sentence2"].tolist()
    logging.info("total number of val pair sentences: " + str(len(val_pair_sentences)))
    val_len_sentences = np.add(df_val["num_rules"].tolist(), df_val["num_facts"].tolist())
    logging.info("total number of val labels: " + str(len(val_labels)))

    # 3. Create the loaders
    df_train = df_train.drop(columns=['sentence1', 'sentence2', 'sat1', 'sat2', 'prompts'])
    df_val = df_val.drop(columns=['sentence1', 'sentence2', 'sat1', 'sat2', 'prompts'])
    train_loader = dataloader(df_train, args.batch_size)
    print('train_loader length: ', len(train_loader))
    val_loader = dataloader(df_val, args.batch_size)
    del df_train, df_val
    gc.collect()

    # 4. Build model
    model = Bert_OrderEmb_HD_HL1(args)

    tuned_parameters = [{'params': [param for name, param in model.named_parameters()]}]

    optimizer = AdamW(tuned_parameters, lr=args.lr)

    model_file = os.path.join(model_path, "_".join(model_params) + ".pt")
    #  early_stopping = EarlyStopping(patience=20, verbose=False, path=model_file, delta=1e-10)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=model_file, delta=1e-10)
    scheduler = get_linear_schedule_with_warmup(optimizer, len(train_loader) * 2,
                                                int(len(train_loader) * args.epochs))

    # 5. GPU setting
    logging.info("Setting GPU...")
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            logging.info("use multiple GPUs")
            model = torch.nn.DataParallel(model)
        else:
            logging.info("use single GPU")
        model.to("cuda")

    # 6. Start training
    model.train()
    logging.info("Start training...")
    logging.info("Epoch; train_loss; train_acc; train_precision; train_recall; train_f1score; val_loss; val_acc; "
                 "val_precision; val_recall; val_f1score")
    torch.set_printoptions(threshold=10)
    for epoch in trange(args.epochs, desc="Epoch"):
        train_loss = 0
        avg_train_acc = 0
        avg_train_fscore = 0
        avg_train_prec = 0
        avg_train_recall = 0

        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            indices = batch[0][:, 0]
            # print(indices)
            indices = indices.type(torch.int)
            sentences_1 = np.array(main_sentences)[indices - 1]
            sentences_2 = np.array(pair_sentences)[indices - 1]
            len_sent = np.array(len_sentences)[indices - 1]
            max_len_sent = max(len_sent)
            labels = np.array(cl_labels)[indices - 1]
            labels = labels.astype(numpy.float32)

            seq_in1 = tokenize_mask(sentences_1, args.max_seq_length, bert_tokenizer, max_len_sent, 'premises')
            seq_in2 = tokenize_mask(sentences_2, args.max_seq_length, bert_tokenizer, max_len_sent, 'hypos')

            token_ids1, input_masks1 = convert_tuple_to_tensor(seq_in1, use_gpu)
            token_ids2, input_masks2 = convert_tuple_to_tensor(seq_in2, use_gpu)

            optimizer.zero_grad()
            loss, preds = model(token_ids1, input_masks1, token_ids2, input_masks2, labels)

            # train_acc = flat_accuracy(preds, labels)
            # labels = labels.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()
            train_fscore = f1_score(labels, preds, average='binary')
            train_acc = accuracy_score(labels, preds)
            train_prec = precision_score(labels, preds, average='binary')
            train_recall = recall_score(labels, preds, average='binary')

            if pd.isna(loss):
                print(indices)
                break
            #  print(f"loss: {loss}")
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            avg_train_fscore += train_fscore
            avg_train_acc += train_acc
            avg_train_prec += train_prec
            avg_train_recall += train_recall

            del seq_in1, seq_in2
            del token_ids1, token_ids2
            del input_masks1, input_masks2
            del loss
            gc.collect()

        print('train_loss: ', train_loss)
        print('num steps: ', str(step + 1))
        train_loss /= (step + 1)
        avg_train_acc /= (step + 1)
        avg_train_prec /= (step + 1)
        avg_train_recall /= (step + 1)
        avg_train_fscore /= (step + 1)

        '''
        if avg_f1_score == 0:
            avg_f1_score = 0
        else:
            avg_f1_score /= (step + 1)
        '''

        print('epoch: ', epoch, ' train_loss: {:,}'.format(train_loss),
              ' train_acc: {:,}'.format(avg_train_acc),
              ' train_precision: {:,}'.format(avg_train_prec),
              ' train_recall: {:,}'.format(avg_train_recall),
              ' train_fscore: {:,}'.format(avg_train_fscore))
        torch.cuda.empty_cache()

        # validation
        model.eval()
        val_loss = 0
        avg_val_acc = 0
        avg_val_fscore = 0
        avg_val_prec = 0
        avg_val_recall = 0

        for step, batch in enumerate(tqdm(val_loader, desc="validation")):
            idx_eval = batch[0][:, 0]
            idx_eval = idx_eval.type(torch.int)

            sent_eval_1 = np.array(val_main_sentences)[idx_eval - 1]
            sent_eval_2 = np.array(val_pair_sentences)[idx_eval - 1]
            len_sent = np.array(val_len_sentences)[idx_eval - 1]
            max_len_sent = max(len_sent)
            v_labels = np.array(val_labels)[idx_eval - 1]
            v_labels = v_labels.astype(numpy.float32)

            seq_ev1 = tokenize_mask(sent_eval_1, args.max_seq_length, bert_tokenizer, max_len_sent, 'premises')
            seq_ev2 = tokenize_mask(sent_eval_2, args.max_seq_length, bert_tokenizer, max_len_sent, 'hypos')

            token_ids1, input_masks1 = convert_tuple_to_tensor(seq_ev1, use_gpu)
            token_ids2, input_masks2 = convert_tuple_to_tensor(seq_ev2, use_gpu)

            loss, preds = model(token_ids1, input_masks1, token_ids2, input_masks2, v_labels)

            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()

            val_loss += loss.item()
            preds = preds.cpu().detach().numpy()

            avg_val_acc += accuracy_score(v_labels, preds)
            avg_val_prec += precision_score(v_labels, preds, average='binary')
            avg_val_recall += recall_score(v_labels, preds, average='binary')
            avg_val_fscore += f1_score(v_labels, preds, average='binary')

            del seq_ev1, seq_ev2
            del token_ids1, input_masks1, token_ids2, input_masks2
            del loss
            gc.collect()

        print('val_loss: ', val_loss)
        print('num steps: ', str(step + 1))
        val_loss /= (step + 1)
        avg_val_acc /= (step + 1)
        avg_val_prec /= (step + 1)
        avg_val_recall /= (step + 1)
        avg_val_fscore /= (step + 1)

        '''
        if avg_eval_f1_score == 0:
            avg_eval_f1_score = 0
        else:
            avg_eval_f1_score /= (step + 1)
        '''
        model.train()

        print('epoch: ', epoch + 1, ' val_loss: {:,}'.format(val_loss))
        logging.info("%d;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f;%.4f", epoch + 1, train_loss,
                     avg_train_acc, avg_train_prec, avg_train_recall, avg_train_fscore,
                     val_loss, avg_val_acc, avg_val_prec, avg_val_recall, avg_val_fscore)

        torch.cuda.empty_cache()

        early_stopping(val_loss, model)

        if epoch > 3:  # 20
            early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logging.info("Early stopping. Model trained.")
            break

    torch.cuda.empty_cache()

    # Testing
