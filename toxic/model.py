from keras.layers import Dense, Embedding, Input, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate, Reshape
from keras.layers import Bidirectional, Dropout, CuDNNGRU, GRU
from keras.models import Model
from keras.optimizers import RMSprop

# def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
#     input_layer = Input(shape=(sequence_length,))
#     embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
#                                 weights=[embedding_matrix], trainable=False)(input_layer)
#     x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
#     #x = Bidirectional(GRU(recurrent_units, return_sequences=True))(embedding_layer)
#     x = Dropout(dropout_rate)(x)
#     x = Bidirectional(GRU(recurrent_units, return_sequences=False))(x)
#     x = Dense(dense_size, activation="relu")(x)
#     output_layer = Dense(6, activation="sigmoid")(x)
#
#     model = Model(inputs=input_layer, outputs=output_layer)
#     model.compile(loss='binary_crossentropy',
#                   optimizer=RMSprop(clipvalue=1, clipnorm=1),
#                   metrics=['accuracy'])
#
#     return model


def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(x)
    x_max = GlobalMaxPool1D()(x)
    x_avg = GlobalAveragePooling1D()(x)
    x = concatenate([x_max, x_avg])
    # x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(clipvalue=1, clipnorm=1), metrics=['accuracy'])

    return model