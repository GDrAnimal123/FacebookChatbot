from keras.layers import Input, Dense, LSTM, Embedding

# Step 3: Build simple Encoder (encoder_input, embedding, 3 LSTM) -> return state.


class Encoder():
    def __init__(self, num_words, embedding_size, hidden_size):
        self.embedding = Embedding(input_dim=num_words,
                                   output_dim=embedding_size,
                                   name="encoder_embedding")
        self.lstm1 = LSTM(hidden_size, name="encoder_lstm1", return_sequences=True)
        self.lstm2 = LSTM(hidden_size, name="encoder_lstm2", return_sequences=True)
        self.lstm3 = LSTM(hidden_size, name="encoder_lstm3", return_sequences=False, return_state=True)

    def __call__(self, input):
        net = input

        # Connect the embedding-layer
        net = self.embedding(net)

        # Connect all the LSTM-layers
        # net = self.lstm1(net)
        # net = self.lstm2(net)
        net = self.lstm3(net)

        output, state_h, state_c = net
        return output, state_h, state_c

# Step 4: Build simple Decoder (encoder_state, embedding, 3 LSTM) -> return sequence.


class Decoder():
    def __init__(self, num_words, embedding_size=128, hidden_size=128):
        self.embedding = Embedding(input_dim=num_words,
                                   output_dim=embedding_size,
                                   name="decoder_embedding")
        self.lstm1 = LSTM(hidden_size, name="decoder_lstm1", return_sequences=True)
        self.lstm2 = LSTM(hidden_size, name="decoder_lstm2", return_sequences=True)
        self.lstm3 = LSTM(hidden_size, name="decoder_lstm3", return_sequences=True, return_state=True)

    def __call__(self, input, intial_state):
        net = input

        # Connect the embedding-layer
        net = self.embedding(net)

        # Connect all the LSTM-layers
        # net = self.lstm1(net, intial_state)
        # net = self.lstm2(net, intial_state)
        net = self.lstm3(net, intial_state)

        output, state_h, state_c = net
        return output, state_h, state_c


class Attention():
    def __init__(self, encoder_out, decoder_out):
        self.attention = AttentionLayer(name='attention_layer')([encoder_out, decoder_out])

    def __call__(self):
        attn_out, attn_states = attn_layer([encoder_out, decoder_out])
        return attn_out, attn_states
