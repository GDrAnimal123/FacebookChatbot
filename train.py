import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from Class.Tokenizer import TokenizerWrap
from model import define_nmt
import create_data

# Variables for model
start_word = 'ssss'
end_word = 'eeee'
num_words = 10000
embedding_size = 128
hidden_size = 512

# Variables for training
epoch = 30
batch_size = 128
lr = 1e-3


# def plot_attention_weights(encoder_inputs, attention_weights, en_id2word, fr_id2word):
#     """
#     Plots attention weights
#     :param encoder_inputs: Sequence of word ids (list/numpy.ndarray)
#     :param attention_weights: Sequence of (<word_id_at_decode_step_t>:<attention_weights_at_decode_step_t>)
#     :param en_id2word: dict
#     :param fr_id2word: dict
#     :return:
#     """

#     assert_msg = 'Your attention weights was empty. Please check if the decoder produced  a proper translation'
#     assert len(attention_weights) != 0, assert_msg

#     mats = []
#     dec_inputs = []
#     for dec_ind, attn in attention_weights:
#         mats.append(attn.reshape(-1))
#         dec_inputs.append(dec_ind)
#     attention_mat = np.transpose(np.array(mats))

#     fig, ax = plt.subplots(figsize=(32, 32))
#     ax.imshow(attention_mat)

#     ax.set_xticks(np.arange(attention_mat.shape[1]))
#     ax.set_yticks(np.arange(attention_mat.shape[0]))

#     ax.set_xticklabels([fr_id2word[inp] if inp != 0 else "<Res>" for inp in dec_inputs])
#     ax.set_yticklabels([en_id2word[inp] if inp != 0 else "<Res>" for inp in encoder_inputs.ravel()])

#     ax.tick_params(labelsize=32)
#     ax.tick_params(axis='x', labelrotation=90)

#     if not os.path.exists(os.path.join('..', 'results')):
#         os.mkdir(os.path.join('..', 'results'))
#     plt.savefig(os.path.join('..', 'results', 'attention.png'))


if __name__ == '__main__':
    LIMIT = 1_000_000
    THRESHOLD_SCORE = 5

    # Optional: Load existing tokenizers
    # @@ Change to False if you want to create and save your tokenizers
    is_tokenizer_exist = True
    if is_tokenizer_exist:
        # Load existing tokenizer.
        print("Loading tokenizers...")
        x_tokenizer = create_data.deserialize("data/x_tokenizer.p")
        y_tokenizer = create_data.deserialize("data/y_tokenizer.p")
    else:
        print("Creating new tokenizers...")
        # Step 1: Get Conversation between A n B data.
        # x, y = create_data.load_rdany_data("data/rdany_conversations_2016-03-01.csv", start_word, end_word)
        x, y = create_data.load_reddit_data("data/2015-05.db", start_word, end_word, threshold_score=THRESHOLD_SCORE, limit=LIMIT)

        # Step 2: Build Tokenizer for A and B respectively
        x_tokenizer = TokenizerWrap(x, padding="pre", reverse=True, num_words=num_words, max_tokens=None)
        y_tokenizer = TokenizerWrap(y, padding="post", reverse=False, num_words=num_words, max_tokens=None)

        create_data.serialize(x_tokenizer, "data/x_tokenizer.p")
        create_data.serialize(y_tokenizer, "data/y_tokenizer.p")

    x_train = x_tokenizer.tokens_padded
    y_train = y_tokenizer.tokens_padded

    encoder_input_data = x_train
    decoder_input_data = y_train[:, :-1]
    decoder_output_data = np.expand_dims(y_train[:, 1:], -1)

    x_data = \
        {
            'encoder_input': encoder_input_data,
            'decoder_input': decoder_input_data
        }
    y_data = \
        {
            'decoder_output': decoder_output_data
        }

    model_train, _, _ = define_nmt(num_words, embedding_size, hidden_size)
    optimizer = RMSprop(lr=lr)
    model_train.compile(optimizer=optimizer,
                        loss="sparse_categorical_crossentropy")

    checkpoint_name = 'model/reddit_emb{}_h{}_lr{}_'.format(embedding_size, hidden_size, lr)

    callback_checkpoint = ModelCheckpoint(filepath=checkpoint_name + 'epoch{epoch:02d}-loss{val_loss:.2f}.h5',
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True,
                                          period=5)
    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                            patience=3, verbose=1)
    callback_tensorboard = TensorBoard(log_dir='./logs/reddit/',
                                       histogram_freq=0,
                                       write_graph=False)
    callbacks = [callback_early_stopping,
                 callback_checkpoint,
                 callback_tensorboard]
    model_train.fit(x=x_data,
                    y=y_data,
                    batch_size=batch_size,
                    epochs=epoch,
                    callbacks=callbacks,
                    validation_split=0.1,
                    )
    # Show predict every epoch
    # model_train.save_weights('model/rdany_weight.h5')
