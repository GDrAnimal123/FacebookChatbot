from keras.models import Model
from keras.layers import Input, Dense, Embedding

from Class.Seq2Seq import Encoder, Decoder


def define_nmt(num_words, embedding_size=64, hidden_size=128):
    encoder_inputs = Input(shape=(None,), name='encoder_input')
    decoder_inputs = Input(shape=(None,), name='decoder_input')

    encoder = Encoder(num_words, embedding_size, hidden_size)
    decoder = Decoder(num_words, embedding_size, hidden_size)
    decoder_dense = Dense(num_words, activation="softmax", name='decoder_output')

    # For Training
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_outputs, _, _ = decoder(decoder_inputs, encoder_states)
    # Apply dot product
    # decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])
    decoder_outputs = decoder_dense(decoder_outputs)

    model_train = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # For Inference
    decoder_state_input_h = Input(shape=(hidden_size,))
    decoder_state_input_c = Input(shape=(hidden_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder(decoder_inputs, decoder_states_inputs)
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_states = [state_h, state_c]
    model_encoder = Model(encoder_inputs, encoder_states)
    model_decoder = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model_train, model_encoder, model_decoder

# def define_nmt(hidden_size, batch_size, en_timesteps, en_vsize, fr_timesteps, fr_vsize):
#     """ Defining a NMT model """

#     # Define an input sequence and process it.
#     encoder_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inputs')
#     decoder_inputs = Input(batch_shape=(batch_size, fr_timesteps - 1, fr_vsize), name='decoder_inputs')

#     # Encoder GRU
#     encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru')
#     encoder_out, encoder_state = encoder_gru(encoder_inputs)

#     # Set up the decoder GRU, using `encoder_states` as initial state.
#     decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True, name='decoder_gru')
#     decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_state)

#     # Attention layer
#     attn_layer = AttentionLayer(name='attention_layer')
#     attn_out, attn_states = attn_layer([encoder_out, decoder_out])

#     # Concat attention input and decoder GRU output
#     decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

#     # Dense layer
#     dense = Dense(fr_vsize, activation='softmax', name='softmax_layer')
#     dense_time = TimeDistributed(dense, name='time_distributed_layer')
#     decoder_pred = dense_time(decoder_concat_input)

#     # Full model
#     full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
#     full_model.compile(optimizer='adam', loss='categorical_crossentropy')

#     full_model.summary()

#     """ Inference model """
#     batch_size = 1

#     """ Encoder (Inference) model """
#     encoder_inf_inputs = Input(batch_shape=(batch_size, en_timesteps, en_vsize), name='encoder_inf_inputs')
#     encoder_inf_out, encoder_inf_state = encoder_gru(encoder_inf_inputs)
#     encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])

#     """ Decoder (Inference) model """
#     decoder_inf_inputs = Input(batch_shape=(batch_size, 1, fr_vsize), name='decoder_word_inputs')
#     encoder_inf_states = Input(batch_shape=(batch_size, en_timesteps, hidden_size), name='encoder_inf_states')
#     decoder_init_state = Input(batch_shape=(batch_size, hidden_size), name='decoder_init')

#     decoder_inf_out, decoder_inf_state = decoder_gru(decoder_inf_inputs, initial_state=decoder_init_state)
#     attn_inf_out, attn_inf_states = attn_layer([encoder_inf_states, decoder_inf_out])
#     decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
#     decoder_inf_pred = TimeDistributed(dense)(decoder_inf_concat)
#     decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
#                           outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])

#     return full_model, encoder_model, decoder_model
