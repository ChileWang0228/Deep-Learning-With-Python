import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
import six


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        # args = tf.cond(is_train, lambda: tf.nn.dropout(
        #     args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args

# use_age

# rnn = cudnn_gru(num_layers=self.config.rnn_layer_num,
#                 num_units=self.config.rnn_hidden_size,
#                 batch_size=max_sentence_num * batch_size,
#                 input_size=sample_embedding.get_shape().as_list()[-1])
# memory_embeddings, final_state = rnn(sample_embedding, seq_len=tf.reshape(self.input_sentences_lens, [-1]))

class cudnn_gru(object):

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask_fw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            mask_bw = dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None)
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=False):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, _ = gru_fw(
                    outputs[-1] * mask_fw, initial_state=(init_fw, ))
            with tf.variable_scope("bw_{}".format(layer)):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
                out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        # if concat_layers:
        #     res = tf.concat(outputs[1:], axis=2)
        # else:
        # final layer
        res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        final_states = tf.reverse_sequence(res, seq_lengths=seq_len, seq_axis=1, batch_axis=0)[:, 0, :]
        return res, final_states


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.
    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def mkMask(input_tensor, maxLen):
    shape_of_input = tf.shape(input_tensor)
    shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])

    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)


def weight_noise(weight, stddev, is_training):
    weight_shape = weight.get_shape().as_list()
    return tf.cond(is_training,
                   lambda: weight + tf.random_normal(shape=weight_shape,
                                                     stddev=stddev,
                                                     mean=0.0,
                                                     dtype=tf.float32),
                   lambda: weight)


def cuda_rnn(inputs, num_layers, hidden_size, seq_len, init_states=None, cell_type="GRU"):
    """Run the CuDNN RNN.
    Arguments:
        - inputs:   A tensor of shape [batch, length, input_size] of inputs.
        - layers:   Number of RNN layers.
        - hidden_size:     Number of units in each layer.
        - is_training:     tf.bool indicating whether training mode is enabled.
        - init_states:
    Return a tuple of (outputs, init_state, final_state).
    """
    input_size = inputs.get_shape()[-1].value
    if input_size is None:
        raise ValueError("Number of input dimensions to CuDNN RNNs must be "
                         "known, but was None.")

    # CUDNN expects the inputs to be time major
    inputs = tf.transpose(inputs, [1, 0, 2])
    if cell_type.lower() == "gru":
        cudnn_cell = cudnn_rnn.CudnnGRU(num_layers, hidden_size, input_mode="linear_input",
                                        direction="bidirectional")
    elif cell_type.lower() == "lstm":
        cudnn_cell = cudnn_rnn.CudnnLSTM(num_layers, hidden_size, input_mode="linear_input",
                                         direction="bidirectional")
    else:
        raise Exception("LSTM or GRU is required.")

    if init_states is None:
        init_state = tf.tile(
            tf.zeros([2 * num_layers, 1, hidden_size], dtype=tf.float32),
            [1, tf.shape(inputs)[1], 1])
        if cell_type.lower() == "gru":
            init_states = (init_state, )
        else:
            init_states = (init_state, init_state)

    output, *_ = cudnn_cell(
        inputs,
        initial_state=init_states,
        training=True)

    # Convert to batch major
    output = tf.transpose(output, [1, 0, 2])
    final_states = tf.reverse_sequence(output, seq_lengths=seq_len, seq_axis=1, batch_axis=0)[:, 0, :]

    return output, final_states


def cudnn_lstm_parameter_size(input_size, hidden_size):
    """Number of parameters in a single CuDNN LSTM cell."""
    biases = 8 * hidden_size
    weights = 4 * (hidden_size * input_size) + 4 * (hidden_size * hidden_size)
    return biases + weights


def direction_to_num_directions(direction):
    if direction == "unidirectional":
        return 1
    elif direction == "bidirectional":
        return 2
    else:
        raise ValueError("Unknown direction: %r." % (direction,))


def estimate_cudnn_parameter_size(num_layers,
                                  input_size,
                                  hidden_size,
                                  input_mode,
                                  direction):
    """
    Compute the number of parameters needed to
    construct a stack of LSTMs. Assumes the hidden states
    of bidirectional LSTMs are concatenated before being
    sent to the next layer up.
    """
    num_directions = direction_to_num_directions(direction)
    params = 0
    isize = input_size
    for layer in range(num_layers):
        for direction in range(num_directions):
            params += cudnn_lstm_parameter_size(
                isize, hidden_size
            )
        isize = hidden_size * num_directions
    return params


def gelu(input_tensor):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      input_tensor: float Tensor to perform activation.
    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf
