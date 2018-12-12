import tensorflow as tf
import sonnet as snt

def fuse_modality(vision_vectorized, text_embedded, config):

    if config["method"] == "concat":
        fusing_resblock_lstm = tf.concat((text_embedded, vision_vectorized), axis=1)

    elif config["method"] == "hadamard":
        proj = snt.Linear(output_size=vision_vectorized.get_shape()[1])(text_embedded)
        fusing_resblock_lstm = tf.multiply(proj, vision_vectorized)  # hadamard product

    else:
        assert False, "Fusing can be 'concat' or 'hadamard' not '{}'".format(config["fusing_method"])

    return fusing_resblock_lstm