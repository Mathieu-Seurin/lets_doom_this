import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog, Model

import tensorflow as tf
import sonnet as snt

from neural_toolbox.film_utils import get_init_conv, get_init_mlp, ResBlock, FilmedResblock, FilmLayer
from neural_toolbox.text_utils import compute_dynamic_rnn, compute_embedding
from neural_toolbox.fuse_utils import fuse_modality

#####################################
#        MULTI-MODAL POLICY
#####################################

class FilmedResnetPolicy(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):

        config = options["custom_options"]

        n_resblock = config["vision"]["resblock_config"]["n_resblock"]
        initializers_mlp = get_init_mlp()
        initializers_conv = get_init_conv()

        # Objective pipeline
        # ==================
        objective = input_dict["obs"]["mission"]

        # Embedding : if one-hot encoding for word -> no embedding
        embedded_obj = compute_embedding(objective=objective, config=config["text_objective_config"])


        # (+bi) lstm ( + layer_norm), specified in config
        last_ht_rnn = compute_dynamic_rnn(inputs=embedded_obj, config=config["text_objective_config"])

        # Image part
        #============
        state = input_dict["obs"]["image"]

        # Stem
        feat_stem = state
        stem_config = config["vision"]["stem_config"]

        for layer in range(stem_config["n_layers"]):
            feat_stem = snt.Conv2D(output_channels=stem_config["n_channels"][layer],
                                   # the number of channel is marked as list, index=channel at this layer
                                   kernel_shape=stem_config["kernel"][layer],
                                   stride=stem_config["stride"][layer],
                                   padding=snt.VALID,
                                   initializers=initializers_conv)(feat_stem)

        # Resblock and modulation
        resconfig = config["vision"]["resblock_config"]

        next_block = feat_stem
        for block in range(n_resblock):
            film_layer = FilmLayer()
            next_block = FilmedResblock(film_layer=film_layer,
                                        n_conv_channel=resconfig["n_channels"][block],
                                        kernel=resconfig["kernel"][block],
                                        stride=resconfig["stride"][block])({"state" : next_block,
                                                                            "objective" : last_ht_rnn})

        head_config = config["vision"]["head_config"]
        head_conv = snt.Conv2D(output_channels=head_config["n_channels"],
                                kernel_shape=head_config["kernel"],
                                stride=head_config["stride"],
                                padding=snt.SAME,
                                initializers=initializers_conv)(next_block)

        flatten_resblock = snt.BatchFlatten(preserve_dims=1)(head_conv)

        # Classifier
        # ===========
        out_mlp1 = tf.nn.relu(snt.Linear(config["last_layer_hidden"], initializers=initializers_mlp)(flatten_resblock))
        out_mlp2 = snt.Linear(num_outputs, initializers=initializers_mlp)(out_mlp1)

        return out_mlp2, out_mlp1


class EarlyMergeCNNPolicy(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):

        config = options["custom_options"]

        initializers_conv = get_init_conv()
        initializers_mlp = get_init_mlp()

        state = input_dict['obs']["image"]

        # Objective Processing
        # ====================

        objective = input_dict['obs']["mission"]

        # Embedding : if one-hot encoding for word -> no embedding
        embedded_obj = compute_embedding(objective=objective, config=config["text_objective_config"])
        # (+bi) lstm ( + layer_norm), specified in config
        last_ht_rnn = compute_dynamic_rnn(inputs=embedded_obj, config=config["text_objective_config"],
                                          sequence_length=input_dict['obs']['sentence_length'])


        reducing_rnn_state = config["fusing"]["reduce_text_before_fuse"]
        if reducing_rnn_state:
            last_ht_rnn = snt.Linear(reducing_rnn_state)(last_ht_rnn)


        # Duplicate rnn state to append it to the image as in "The impact of early fusion and VQA in details"
        last_ht_rnn = tf.expand_dims(tf.expand_dims(last_ht_rnn, axis=1), axis=2)

        tiled_last_ht = tf.tile(last_ht_rnn, multiples=[1,7,7,1]) # todo : don't hardcode 7x7

        # Features Extractor
        # ==================
        to_next_conv = state
        vision_config = config["vision"]
        for layer in range(vision_config["n_layers"]):

            # Early text fusing, in the conv pipeline
            if layer == config["fusing"]["layer_to_fuse"]-1:
                to_next_conv = tf.concat((to_next_conv, tiled_last_ht), axis=3)

            conv_layer = snt.Conv2D(output_channels= vision_config["n_channels"][layer],
                                                 kernel_shape=vision_config["kernel"][layer],
                                                 stride=vision_config["stride"][layer],
                                                 initializers=initializers_conv)(to_next_conv)

            to_next_conv = tf.nn.relu(conv_layer)

        flatten_vision = snt.BatchFlatten(preserve_dims=1)(to_next_conv)

        # If necessary add direction of character
        if input_dict['obs'].get('direction', False) is not False:
            direction_one_hot = input_dict['obs']['direction']
            flatten_vision = tf.concat((flatten_vision, direction_one_hot), axis=1)

        # MLP layers -> Q function or policy
        out_mlp1 = tf.nn.relu(snt.Linear(config["last_layer_hidden"],initializers=initializers_mlp)(flatten_vision))
        out_mlp2 = snt.Linear(num_outputs,initializers=initializers_mlp)(out_mlp1)

        return out_mlp2, out_mlp1 # you need to return output (out_mlp2) and input of the last layer (out_mlp1)


class BaseCNNPolicyLateFuse(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].
        """

        config = options["custom_options"]

        initializers_conv = get_init_conv()
        initializers_mlp = get_init_mlp()

        state = input_dict['obs']["image"]

        # Features Extractor
        # ==================
        to_next_conv = state
        vision_config = config["vision"]
        for layer in range(vision_config["n_layers"]):
            conv_layer = snt.Conv2D(output_channels= vision_config["n_channels"][layer],
                                                 kernel_shape=vision_config["kernel"][layer],
                                                 stride=vision_config["stride"][layer],
                                                 initializers=initializers_conv)(to_next_conv)

            to_next_conv = tf.nn.relu(conv_layer)


        flatten_vision = snt.BatchFlatten(preserve_dims=1)(to_next_conv)


        # Objective Processing
        # ====================
        objective = input_dict['obs']["mission"]

        # Embedding : if one-hot encoding for word -> no embedding
        embedded_obj = compute_embedding(objective=objective, config=config["text_objective_config"])
        # (+bi) lstm ( + layer_norm), specified in config
        last_ht_rnn = compute_dynamic_rnn(inputs=embedded_obj, config=config["text_objective_config"],
                                          sequence_length=input_dict['obs']['sentence_length'])

        # Fusing both modalities
        # ======================
        fusing_resblock_lstm = fuse_modality(vision_vectorized=flatten_vision, text_embedded=last_ht_rnn,
                                             config=config["fusing"])

        # If necessary add direction of character
        if input_dict['obs'].get('direction', False) is not False:
            direction_one_hot = input_dict['obs']['direction']
            fusing_resblock_lstm = tf.concat((fusing_resblock_lstm, direction_one_hot), axis=1)

        # MLP layers -> Q function or policy
        out_mlp1 = tf.nn.relu(snt.Linear(config["last_layer_hidden"],initializers=initializers_mlp)(fusing_resblock_lstm))
        out_mlp2 = snt.Linear(num_outputs,initializers=initializers_mlp)(out_mlp1)

        return out_mlp2, out_mlp1 # you need to return output (out_mlp2) and input of the last layer (out_mlp1)


#####################################
#        UNIMODAL_POLICY
#####################################

class BaseCNNPolicy(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.

        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.

        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].
        """
        config = options["custom_options"]

        initializers_conv = get_init_conv()
        initializers_mlp = get_init_mlp()

        state = input_dict['obs']["image"]

        # Features Extractor
        to_next_conv = state

        for layer in range(config["n_layers"]):
            conv_layer = snt.Conv2D(output_channels= config["n_channels"][layer],
                                    kernel_shape=config["kernel"][layer],
                                    stride=config["stride"][layer],
                                    initializers=initializers_conv)(to_next_conv)

            to_next_conv = tf.nn.relu(conv_layer)

        # Flatten input then mlp
        flatten_vision = snt.BatchFlatten(preserve_dims=1)(to_next_conv)
        out_mlp1 = tf.nn.relu(snt.Linear(config["last_layer_hidden"],initializers=initializers_mlp)(flatten_vision))
        out_mlp2 = snt.Linear(num_outputs,initializers=initializers_mlp)(out_mlp1)

        return out_mlp2, out_mlp1 # you need to return output (out_mlp2) and input of the last layer (out_mlp1)

class ResnetPolicy(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):

        config = options["custom_options"]

        n_resblock = config["n_resblock"]
        initializers_mlp = get_init_mlp()
        initializers_conv = get_init_conv()

        # Don't use objective at the moment.
        state = input_dict["obs"]["image"]

        # Stem
        feat_stem = state
        stem_config = config["stem_config"]

        for layer in range(stem_config["n_layer"]):
            feat_stem = snt.Conv2D(output_channels=stem_config["channel"][layer],
                                   #the number of channel is marked as list, index=channel at this layer
                                   kernel_shape=stem_config["kernel_size"][layer],
                                   stride=stem_config["stride"],
                                   padding=snt.VALID,
                                   initializers=initializers_conv)(feat_stem)


        next_block = feat_stem

        for block in range(n_resblock):
            next_block = ResBlock(config["resblock_config"])(next_block)

        flatten_resblock = snt.BatchFlatten(preserve_dims=1)(next_block)

        out_mlp1 = tf.nn.relu(snt.Linear(options["last_layer_hidden"], initializers=initializers_mlp)(flatten_resblock))
        out_mlp2 = snt.Linear(num_outputs, initializers=initializers_mlp)(out_mlp1)

        return out_mlp2, out_mlp1



#####################################
#        REGISTER MODELS
#####################################
ModelCatalog.register_custom_model("base_cnn", BaseCNNPolicy)
ModelCatalog.register_custom_model("base_resnet", ResnetPolicy)

#ModelCatalog.register_custom_model("resnet_concat", ResnetPolicyConcat)
ModelCatalog.register_custom_model("cnn_early", EarlyMergeCNNPolicy)
ModelCatalog.register_custom_model("cnn_late", BaseCNNPolicyLateFuse)

ModelCatalog.register_custom_model("resnet_film", FilmedResnetPolicy)
