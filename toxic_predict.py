from keras.layers import Dense, Embedding, Input, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate, Reshape
from keras.layers import Bidirectional, Dropout, CuDNNGRU, GRU
from keras.models import Model
from keras.optimizers import RMSprop
import pandas as pd
from toxic.nltk_utils import tokenize_sentences
from toxic.embedding_utils import read_embedding_list, clear_embedding_list, convert_tokens_to_ids
from toxic.nltk_utils import clean

import numpy as np
UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4

train_file_path = "train.csv"
test_file_path = "test.csv"
embedding_path = "crawl-300d-2M.vec"
print("Loading data...")
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# train_data['comment_text'] = train_data.apply(lambda x: clean(x.comment_text), axis=1)
# train_data['comment_text'] = train_data.apply(lambda x: clean(x.comment_text), axis=1)

list_sentences_train = train_data["comment_text"].fillna(NAN_WORD).values
list_sentences_test = test_data["comment_text"].fillna(NAN_WORD).values
y_train = train_data[CLASSES].values

#marking comments without any tags as "clean"
rowsums=train_data.iloc[:,2:8].sum(axis=1)
train_data['clean']=(rowsums==0)
#count number of clean entries
train_data['clean'].sum()
print("Total comments = ",len(train_data))
print("Total clean comments = ",train_data['clean'].sum())
print("Total tags =",rowsums.sum())

print("Tokenizing sentences in train set...")
tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})

print("Tokenizing sentences in test set...")
tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)

print(embedding_path)
words_dict[UNKNOWN_WORD] = len(words_dict)

print("Loading embeddings...")
embedding_list, embedding_word_dict = read_embedding_list(file_path=embedding_path)
embedding_size = len(embedding_list[0])

import numpy as np
print("Preparing data...")
embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
embedding_list.append([0.] * embedding_size)
embedding_word_dict[END_WORD] = len(embedding_word_dict)
embedding_list.append([-1.] * embedding_size)

embedding_matrix = np.array(embedding_list)
print(embedding_matrix.shape)

# Variables for the model
sequence_length = 500
result_path = "toxic_results"
batch_size = 256
sentences_length = 500
recurrent_units=64
dropout_rate = 0.3
dense_size=32
fold_count=10


# Model Architecture
input_layer = Input(shape=(sequence_length,))
embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                            weights=[embedding_matrix], trainable=False)(input_layer)
x = Bidirectional(CuDNNGRU(recurrent_units, reset_after=True, recurrent_activation='sigmoid', return_sequences=True))(embedding_layer)
x = Dropout(dropout_rate)(x)
x = Bidirectional(CuDNNGRU(recurrent_units, reset_after=True,  recurrent_activation='sigmoid', return_sequences=True))(x)
x_max = GlobalMaxPool1D()(x)
x_avg = GlobalAveragePooling1D()(x)
x = concatenate([x_max, x_avg])
output_layer = Dense(6, activation="sigmoid")(x)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer=RMSprop(clipvalue=1, clipnorm=1), metrics=['accuracy'])

comment_text = "Yo bitch Ja Rule is more succesful then you'll ever be whats up with you and hating you sad mofuckas...i should bitch slap ur pethedic white faces and get you to kiss my ass you guys sicken me. Ja rule is about pride in da music man. dont diss that shit on him. and nothin is wrong bein like tupac he was a brother too...fuckin white boys get things right next time."

list_texts_to_predict= []
list_texts_to_predict.append(comment_text)

tokenized_sentences_test, words_dict = tokenize_sentences(list_texts_to_predict, words_dict)

id_to_word = dict((id, word) for word, id in words_dict.items())
test_list_of_token_ids = convert_tokens_to_ids(
    tokenized_sentences_test,
    id_to_word,
    embedding_word_dict,
    sequence_length)
X_test = np.array(test_list_of_token_ids)

test_predicts_list = []
for i  in range(0,10):
    print("model{0}_weights.h5".format(i))
    model.load_weights("model{0}_weights.h5".format(i))
    test_predicts = model.predict(X_test, batch_size=1)
    print(test_predicts)
    test_predicts_list.append(test_predicts)

print(test_predicts_list)

test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

test_predicts **= (1. / len(test_predicts_list))
print(test_predicts)
print(PROBABILITIES_NORMALIZE_COEFFICIENT)
test_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT
print(test_predicts)

print(test_predicts)
