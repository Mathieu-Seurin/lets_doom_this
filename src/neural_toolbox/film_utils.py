import sonnet as snt
import tensorflow as tf

def get_init_conv():
    return {
        "w": tf.contrib.layers.xavier_initializer_conv2d(),
        "b": tf.truncated_normal_initializer(stddev=1.0)
    }

def get_init_mlp():
    return {
        "w": tf.contrib.layers.xavier_initializer(),
        "b": tf.truncated_normal_initializer(stddev=1.0)
    }


class FilmLayer(snt.AbstractModule):
    def __init__(self):
        """
        A very basic FiLM layer with a linear transformation from context to FiLM parameters
        """
        super(FilmLayer, self).__init__(name="film_layer")
        self.init_mlp = get_init_mlp()

    def _build(self, inputs):
        """
        :param inputs["state"] : images features map to modulate. Must be a 3-D input vector (+batch size)
        :param inputs["objective"] : conditioned FiLM parameters. Must be a 1-D input vector (+batch size)
        :return: modulated features
        """

        state = inputs["state"]
        context = inputs["objective"]

        height = int(state.get_shape()[1])
        width = int(state.get_shape()[2])
        feature_size = int(state.get_shape()[3])

        film_params = snt.Linear(output_size=2 * feature_size, initializers=self.init_mlp)(context)

        film_params = tf.expand_dims(film_params, axis=[1])
        film_params = tf.expand_dims(film_params, axis=[1])
        film_params = tf.tile(film_params, [1, height, width, 1])

        gammas = film_params[:, :, :, :feature_size]
        betas = film_params[:, :, :, feature_size:]

        output = (1 + gammas) * state + betas
        return output


class ResBlock(snt.AbstractModule):
    def __init__(self, options):
        super(ResBlock, self).__init__(name="resblock")

        self.initializers_conv = get_init_conv()
        self.conv_channel = options["conv_channel"]

    def _build(self, inputs):

        state = inputs

        # First conv + relu
        after_relu1 = tf.nn.relu(snt.Conv2D(output_channels=self.conv_channel,
                                            kernel_shape=[3, 3], stride=1, padding=snt.SAME,
                                            initializers=self.initializers_conv)(state))

        # Second conv + bn ?(not learned) + relu
        after_conv2 = snt.Conv2D(output_channels=self.conv_channel,
                                 kernel_shape=[3, 3], stride=1, padding=snt.SAME,
                                 initializers=self.initializers_conv)(after_relu1)

        # todo : check batchnorm, but can destroy performance in RL.
        # after_bn1 = snt.BatchNorm(scale=False, offset=False)(after_conv2) # No learned parameters
        after_relu2 = tf.nn.relu(after_conv2)

        # Adding skip connection
        result = tf.add(after_relu1, after_relu2)
        return result



class FilmedResblock(snt.AbstractModule):
    def __init__(self, film_layer, conv_layer):
        raise NotImplementedError("Still under test")
        super(FilmedResblock, self).__init__(name="mod_resblock")

        self.initializers_conv = get_init_conv()
        self.film_layer = film_layer
        self.conv1_channel = conv_layer
        self.conv2_channel = conv_layer

    def _build(self, inputs, is_training):

        state = inputs["state"]
        context = inputs["objective"]

        # First conv + relu
        after_relu1 = tf.nn.relu(snt.Conv2D(output_channels=self.conv1_channel,
                                            kernel_shape=[3,3], stride=2, padding=snt.SAME,
                                            initializers=self.initializers_conv)(state))

        # Second conv + bn (not learned) + relu
        after_conv2 = snt.Conv2D(output_channels=self.conv2_channel,
                                 kernel_shape=[3,3], stride=2, padding=snt.SAME,
                                 initializers=self.initializers_conv)(after_relu1)

        after_bn1 = snt.BatchNorm(scale=False, offset=False)(after_conv2, is_training) # No learned parameters
        after_film = self.film_layer({"state" : after_bn1, "objective": context})

        after_relu2 = tf.nn.relu(after_bn1)

        # Adding skip connection
        result = after_relu1 + after_relu2
        return result


if __name__ == "__main__":
    import numpy as np

    def test_film_layer():

        feature_maps = tf.placeholder(tf.float32, shape=[None, 3, 3, 2])
        lstm_state = tf.placeholder(tf.float32, shape=[None, 6])

        inputs = {}
        inputs["state"] = feature_maps
        inputs["objective"] = lstm_state

        film_layer = FilmLayer()
        modulated_feat_cst = FilmedResblock(inputs)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        feature_maps_np = np.array(
            [
                [
                    [[1, 2, 3], [1, 2, 3], [0, 0, 0]],
                    [[1, 2, 3], [1, 2, 3], [1, 1, 1]]
                ],
                [
                    [[-1, 1, 0], [0, 0, 0], [-1, 1, 0]],
                    [[-1, 1, 0], [-1, 1, 1], [-4, 1, 4]]
                ]
            ])

        feature_maps_np = tf.constant(np.transpose(feature_maps_np, axes=[0, 2, 3, 1]), dtype=tf.float32)
        context = tf.constant(np.array([[1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0]]), dtype=tf.float32)

        value_inputs = {"state" : feature_maps_np, "objective" : context}

        # feature_maps_cst = tf.constant(feature_maps_np, dtype=tf.float32)
        # lstm_state_cst = tf.constant(np.array(), dtype=tf.float32)

        output = film_layer(value_inputs).eval()
        return output

    def test_film_resblock():

        feature_maps = tf.placeholder(tf.float32, shape=[None, 3, 3, 2])
        lstm_state = tf.placeholder(tf.float32, shape=[None, 6])

        film_layer = FilmLayer()
        resblock = FilmedResblock(film_layer, 10)

        inputs = {}
        inputs["state"] = feature_maps
        inputs["objective"] = lstm_state

        output = resblock(inputs, is_training=True)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        feature_maps_np = np.array(
            [
                [
                    [[1, 2, 3], [1, 2, 3], [0, 0, 0]],
                    [[1, 2, 3], [1, 2, 3], [1, 1, 1]]
                ],
                [
                    [[-1, 1, 0], [0, 0, 0], [-1, 1, 0]],
                    [[-1, 1, 0], [-1, 1, 1], [-4, 1, 4]]
                ]
            ])

        feature_maps_np = tf.constant(np.transpose(feature_maps_np, axes=[0, 2, 3, 1]), dtype=tf.float32)
        context = tf.constant(np.array([[1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0]]), dtype=tf.float32)

        value_inputs = {"state": feature_maps_np, "objective": context}

        # feature_maps_cst = tf.constant(feature_maps_np, dtype=tf.float32)
        # lstm_state_cst = tf.constant(np.array(), dtype=tf.float32)

        output = resblock(value_inputs, is_training=True).eval()
        return output

