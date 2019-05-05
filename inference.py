import sys
import os
import argparse
import pandas as pd
import numpy as np
import create_data

from Class.Tokenizer import TokenizerWrap
from model import define_nmt

# Variables for model
start_word = 'ssss'
end_word = 'eeee'
num_words = 10000
embedding_size = 32
hidden_size = 512

beam = 3

# parser = argparse.ArgumentParser()
# parser.add_argument("echo")
# args = parser.parse_args()


def process_text(input):
    # This is only necessary when we need to process our input text
    # before convert into tokens
    # Example: (remove stopwords, lementize, ...)
    pass

# Step 6: Predict with beam search technique.


def get_inference_model(path):
    model_train, model_encoder, model_decoder = define_nmt(num_words, embedding_size, hidden_size)
    try:
        print("Get inference model {}".format(path))
        model_train.load_weights(path)
    except Exception as error:
        print(error)
        print("Error trying to load checkpoint.")

    return (model_encoder, model_decoder)


def beam_search_predict(text, x_tokenizer, y_tokenizer, model_encoder, model_decoder, beam=3):

    start_token = y_tokenizer.word_index[start_word.strip()]
    end_token = y_tokenizer.word_index[end_word.strip()]
    max_seq_length = y_tokenizer.max_tokens

    # convert text into tokens using tokenizer.
    encoder_input_data = x_tokenizer.text_to_tokens(text, reverse=True, padding=True)
    decoder_input_data = np.zeros(shape=(1, 1))

    # Get the output of the encoder's LSTM which will be
    # used as the initial state in the decoder's LSTM.
    # state_values includes (state_h, state_c)
    encoder_initial_states = model_encoder.predict(encoder_input_data)

    # This keep track of the top BEAM sequences with [seq, probability]
    # Length of this list corresponds with BEAM(n)
    sequences = [[[start_token], 0.0, encoder_initial_states]]

    while len(sequences[0][0]) < max_seq_length:
        timestep = len(sequences[0][0]) - 1  # has to be ZERO
        all_candidates = list()

        for beam_preds in sequences:
            decoder_input_data[0][0] = beam_preds[0][timestep]
            initial_state_values = beam_preds[2]

            # We return the LSTM-states(h, c) when calling predict() and then
            # feeding these LSTM-states as well the next time we call predict()

            # NOTE: Since we have top BEAM(n) sequences, each sequence produce its own
            # LSTM-states(h, c) at predict stage. Therefore, we append into our sequences
            # with its return states.

            # See more: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

            # Predicted output (None, timesteps, num_words)
            softmax_outputs, state_h, state_c = model_decoder.predict([decoder_input_data] + initial_state_values)
            timestep_output = softmax_outputs[0][0]
            # Top BEAM(n) vocabularies' prob
            topBeam_vocabs = np.argsort(timestep_output)[-beam:]

            # Getting the top BEAM(n) predictions and update
            # the sequence [seq, prob, state_values] again to feed to our model.
            for w in topBeam_vocabs:
                # array requires [:] to avoid making
                # reference to original object.
                seq, prob, state_values = beam_preds[0][:], beam_preds[1], beam_preds[2]
                # token in top beam
                seq += [w]
                # probability score of token in top beam
                prob += timestep_output[w]
                # state_values of the sequence
                state_values = [state_h, state_c]
                all_candidates.append([seq, prob, state_values])

        # order all candidates by probability
        ordered = sorted(all_candidates, reverse=False, key=lambda vec: vec[1])
        # select top BEAM(n)
        sequences = ordered[-beam:]

    # This is a sequence that has the highest prob.
    best_sequence = sequences[-1][0]

    # Here is where we handle the end of sentence token.
    # *** Fix this into while loop -> this cause beam_search little mal-func
    final_sequence = []
    for token in best_sequence:
        if token == end_token:
            break
        if token == start_token:
            continue
        final_sequence.append(token)

    predict_text = y_tokenizer.tokens_to_string(final_sequence)
    return predict_text


def load_inference():
    # x, y = create_data.load_rdany_data("data/rdany_conversations_2016-03-01.csv", start_word, end_word)
    x_tokenizer = create_data.deserialize("data/x_tokenizer.p")
    y_tokenizer = create_data.deserialize("data/y_tokenizer.p")

    infer_model = get_inference_model("model/rdany_weight.h5")
    tokenizers = (x_tokenizer, y_tokenizer)

    return infer_model, tokenizers


# interactive mode
if __name__ == "__main__":

    # Input file
    if sys.stdin.isatty() == False:
        sys.exit()

    # Interactive mode
    print("\n\nStarting interactive mode (first response will take a while):")
    infer_model, tokenizers = load_inference()
    # QAs
    while True:
        question = input("\n(You)> ")

        if question in ["exit()", "quit()"]:
            break

        answers = beam_search_predict(question,
                                      *tokenizers, *infer_model,
                                      beam=1)
        if answers is None:
            print(colorama.Fore.RED + "! Question can't be empty" + colorama.Fore.RESET)
        else:
            print("(Bot)> ", answers)
