from toxic.model import get_model
from toxic.nltk_utils import tokenize_sentences
from toxic.train_utils import train_folds
from toxic.train_utils import train_folds_non_lambda
from toxic.embedding_utils import read_embedding_list, clear_embedding_list, convert_tokens_to_ids

from langdetect import detect
from tools.extend_dataset import translate
from toxic.nltk_utils import clean

import argparse
import numpy as np
import os
import pandas as pd


UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4


def detect_language(row):
    try:
        return detect(row)
    except:
        return "en"


def main():
    parser = argparse.ArgumentParser(
        description="Recurrent neural network for identifying and classifying toxic online comments")

    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    parser.add_argument("embedding_path")
    parser.add_argument("--result-path", default="toxic_results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sentences-length", type=int, default=500)
    parser.add_argument("--recurrent-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--dense-size", type=int, default=32)
    parser.add_argument("--fold-count", type=int, default=10)

    args = parser.parse_args()

    if args.fold_count <= 1:
        raise ValueError("fold-count should be more than 1")

    print("Loading data...")
    train_data = pd.read_csv(args.train_file_path)
    test_data = pd.read_csv(args.test_file_path)


    # Identify language
    #train_data['language'] = train_data['comment_text'].apply(detect_language)
    #test_data['language'] = test_data['comment_text'].apply(detect_language)

    # Translate the non-english to the english.
    #train_data['comment_text'] = train_data.apply(lambda x: translate(x.comment_text, x.language),axis=1)
    #test_data['comment_text'] = test_data.apply(lambda x: translate(x.comment_text, x.language),axis=1)
    #train_data.to_csv("train_data_translated.csv")
    #test_data.to_csv("test_data_translated.csv")

    #train_data['comment_text'] = train_data.apply(lambda x: clean(x.comment_text), axis=1)
    #train_data['comment_text'] = train_data.apply(lambda x: clean(x.comment_text), axis=1)
    #train_data.to_csv("train_data_cleaned_after_translate.csv")
    #test_data.to_csv("test_data_cleaned_after_translate.csv")

    list_sentences_train = train_data["comment_text"].fillna(NAN_WORD).values
    list_sentences_test = test_data["comment_text"].fillna(NAN_WORD).values


    y_train = train_data[CLASSES].values

    print("Tokenizing sentences in train set...")
    tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})

    print("Tokenizing sentences in test set...")
    tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)

    words_dict[UNKNOWN_WORD] = len(words_dict)

    print("Loading embeddings...")
    embedding_list, embedding_word_dict = read_embedding_list(args.embedding_path)
    embedding_size = len(embedding_list[0])

    print("Preparing data...")
    embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

    embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
    embedding_list.append([0.] * embedding_size)
    embedding_word_dict[END_WORD] = len(embedding_word_dict)
    embedding_list.append([-1.] * embedding_size)

    embedding_matrix = np.array(embedding_list)

    embedding_matrix_path = os.path.join(args.result_path, "embedding_matrix.npy")
    np.save(embedding_matrix_path, embedding_matrix)
    words_dict_path = os.path.join(args.result_path, "words_dict.npy")
    np.save(words_dict_path, words_dict)

    id_to_word = dict((id, word) for word, id in words_dict.items())
    train_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_train,
        id_to_word,
        embedding_word_dict,
        args.sentences_length)
    test_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_test,
        id_to_word,
        embedding_word_dict,
        args.sentences_length)
    X_train = np.array(train_list_of_token_ids)
    X_test = np.array(test_list_of_token_ids)
    print(embedding_matrix.shape)
    print(embedding_matrix.shape[0])
    print(embedding_matrix.shape[1])

    model = get_model(
        embedding_matrix,
        args.sentences_length,
        args.dropout_rate,
        args.recurrent_units,
        args.dense_size)

    print("Starting to train models...")
    models = train_folds_non_lambda(X_train, y_train, args.fold_count, args.batch_size, model)

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    print("Predicting results...")
    test_predicts_list = []
    for fold_id, model in enumerate(models):
        model_path = os.path.join(args.result_path, "model{0}_weights.npy".format(fold_id))
        np.save(model_path, model.get_weights())
        model.save_weights("model{0}_weights.h5".format(fold_id))

        test_predicts_path = os.path.join(args.result_path, "test_predicts{0}.npy".format(fold_id))
        test_predicts = model.predict(X_test, batch_size=args.batch_size)
        test_predicts_list.append(test_predicts)
        np.save(test_predicts_path, test_predicts)

    test_predicts = np.ones(test_predicts_list[0].shape)
    for fold_predict in test_predicts_list:
        test_predicts *= fold_predict

    test_predicts **= (1. / len(test_predicts_list))
    test_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT

    test_ids = test_data["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + CLASSES]
    submit_path = os.path.join(args.result_path, "submit")
    test_predicts.to_csv(submit_path, index=False)

    print("Predicting Discussion posts...")
    posts = pd.read_csv("posts_cleaned.csv")
    posts = posts.dropna()
    discussion_posts = posts['MSG_TEXT'].tolist()
    tokenized_discussion_posts, words_dict = tokenize_sentences(discussion_posts, words_dict)
    #id_to_word = dict((id, word) for word, id in words_dict.items())
    discussion_list_of_token_ids = convert_tokens_to_ids(
        tokenized_discussion_posts,
        id_to_word,
        embedding_word_dict,
        args.sentences_length)
    X_test = np.array(discussion_list_of_token_ids)
    discussion_predict_list = []
    for fold_id, model in enumerate(models):
        discussion_predicts = model.predict(X_test, batch_size=args.batch_size)
        discussion_predict_list.append(discussion_predicts)

    discussion_predicts = np.ones(discussion_predict_list[0].shape)
    for fold_predict in discussion_predict_list:
        discussion_predicts *= fold_predict

    discussion_predicts **= (1. / len(discussion_predict_list))
    discussion_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT

    discussion_predicts = pd.DataFrame(data=discussion_predicts, columns=CLASSES)
    discussion_predicts['MSG_TEXT']=discussion_posts
    discussion_predicts = discussion_predicts[["MSG_TEXT"] + CLASSES]
    discussion_predicts_path = os.path.join(args.result_path, "discussion_predicts.csv")
    discussion_predicts.to_csv(discussion_predicts_path, index=False)
if __name__ == "__main__":
    main()
