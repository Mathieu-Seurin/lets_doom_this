import sonnet as snt
import tensorflow as tf


def compute_dynamic_rnn(inputs, config, sequence_length=None):

    if sequence_length != None:
        sequence_length = tf.squeeze(sequence_length, axis=1)

    # todo : bi lstm ?
    lstm_cell = snt.LSTM(hidden_size=config["lstm_hidden_size"],
                        use_layer_norm=config["use_layer_norm"])


    # todo : optionnal init state with vision value
    _, last_ht_rnn = tf.nn.dynamic_rnn(lstm_cell,
                                       inputs=inputs,
                                       dtype=tf.float32,
                                       time_major=False,
                                       sequence_length=sequence_length #if available, reduce time and complexity
                                       #  todo : add sequence length
                                       )

    last_ht_rnn = last_ht_rnn.hidden

    return last_ht_rnn


def compute_embedding(objective, config):

    objective = tf.cast(objective, dtype=tf.int32)

    # if objective is rank 3 -> one-hot encoded (batch_dim, sequence, vocab_size)
    # else objective is rank 2 -> need to compute embedding (batch_dim, sequence_of_embedding)
    if config["embedding_size"] > 0:
        embedded_obj = snt.Embed(vocab_size=config["vocab_size"],  # todo : super ugly, but cannot avoid it.
                                 embed_dim=config["embedding_size"],
                                 densify_gradients=True,
                                 )(objective)

    else:  # embedding_size == 0 -> onehot vector directly to lstm
        embedded_obj = tf.one_hot(indices=objective,
                                  depth=config["vocab_size"],
                                  on_value=1.0,
                                  off_value=0.0,
                                  axis=1
                                  )

    return embedded_obj