# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
# import memory_saving_gradients
# tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

from tensorflow.python.training.optimizer import *
from tensorflow.python.training.optimizer import _OptimizableVariable, _DenseResourceVariableProcessor, \
    _RefVariableProcessor


def _var_key(var):
    # TODO(ashankar): Consolidate handling for eager and graph
    if hasattr(var, "op"):
        return (var.op.graph, var.op.name)
    return var._unique_id  # pylint: disable=protected-access


# def get_filtered_grad_fn(grad_fn):
#     # `distributed_context.join()` requires that its arguments are parallel
#     # across threads, and in particular that `grads_and_vars` has the same
#     # variables in the same order.
#
#     # When computing gradients in eager mode with multiple threads, you
#     # can get extra variables with a gradient of `None`. This happens when
#     # those variables are accessed in another thread during the gradient
#     # computation. To get a consistent set of variables, we filter out
#     # those with `None` gradients.
#     def filtered_grad_fn(*args, **kwargs):
#         return [(g, v) for g, v in grad_fn(*args, **kwargs) if g is not None]
#
#     return filtered_grad_fn


class _TensorProcessor(_OptimizableVariable):
    """Processor for ordinary Tensors.

    Even though a Tensor can't really be updated, sometimes it is useful to
    compute the gradients with respect to a Tensor using the optimizer. Updating
    the Tensor is, of course, unsupported.
    """

    def __init__(self, v):
        self._v = v

    def target(self):
        return self._v

    def update_op(self, optimizer, g):
        raise NotImplementedError("Trying to update a Tensor ", self._v)


def _get_processor(v):
    """The processor of v."""
    if context.executing_eagerly():
        if isinstance(v, ops.Tensor):
            return _TensorProcessor(v)
        else:
            return _DenseResourceVariableProcessor(v)
    if isinstance(
            v, resource_variable_ops.ResourceVariable) and not v._in_graph_mode:  # pylint: disable=protected-access
        # True if and only if `v` was initialized eagerly.
        return _DenseResourceVariableProcessor(v)
    if v.op.type == "VarHandleOp":
        return _DenseResourceVariableProcessor(v)
    if isinstance(v, variables.Variable):
        return _RefVariableProcessor(v)
    if isinstance(v, ops.Tensor):
        return _TensorProcessor(v)
    raise NotImplementedError("Trying to optimize unsupported type ", v)


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu, variable_list=None):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    if variable_list:
        tvars = variable_list
        print('bert only!')
    else:
        tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op, learning_rate, global_step


# tf.train.Optimizer
class AdamWeightDecayOptimizer(tf.train.AdamOptimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""

        # if distribution_strategy_context.get_cross_tower_context():
        #     raise RuntimeError("Use `_distributed_apply()` instead of "
        #                        "`apply_gradients()` in a cross-tower context.")
        # # TODO(isaprykin): Get rid of `has_distribution_strategy()` check by
        # always calling _distributed_apply(), using the default distribution
        # as needed.
        # if distribution_strategy_context.has_distribution_strategy():
        #     grads_and_vars = get_filtered_grad_fn(lambda: grads_and_vars)()
        #     return distribution_strategy_context.get_tower_context().merge_call(
        #         self._distributed_apply, grads_and_vars, global_step, name)

        # No DistributionStrategy case.
        grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
        if not grads_and_vars:
            raise ValueError("No variables provided.")
        converted_grads_and_vars = []
        for g, v in grads_and_vars:
            if g is not None:
                try:
                    # Convert the grad to Tensor or IndexedSlices if necessary.
                    g = ops.convert_to_tensor_or_indexed_slices(g)
                except TypeError:
                    raise TypeError(
                        "Gradient must be convertible to a Tensor"
                        " or IndexedSlices, or None: %s" % g)
                if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
                    raise TypeError(
                        "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
            p = _get_processor(v)
            converted_grads_and_vars.append((g, v, p))

        converted_grads_and_vars = tuple(converted_grads_and_vars)
        var_list = [v for g, v, _ in converted_grads_and_vars if g is not None]

        if not var_list:
            raise ValueError("No gradients provided for any variable: %s." %
                             ([str(v) for _, v, _ in converted_grads_and_vars],))
        with ops.init_scope():
            self._create_slots(var_list)

        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)
            # m = tf.get_variable(
            #     name=param_name + "/adam_m",
            #     shape=param.shape.as_list(),
            #     dtype=tf.float32,
            #     trainable=False,
            #     initializer=tf.zeros_initializer())
            # v = tf.get_variable(
            #     name=param_name + "/adam_v",
            #     shape=param.shape.as_list(),
            #     dtype=tf.float32,
            #     trainable=False,
            #     initializer=tf.zeros_initializer())
            m = self.get_slot(param, "m")
            v = self.get_slot(param, "v")
            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
