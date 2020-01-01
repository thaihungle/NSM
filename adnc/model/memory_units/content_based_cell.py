# Copyright 2018 Jörg Franke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf

from adnc.model.memory_units.dnc_cell import DNCMemoryUnitCell
from adnc.model.utils import oneplus, layer_norm, unit_simplex_initialization

"""
The content-based memory unit.
"""

def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax(logits, temperature=10, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def get_lookup_weighting(memory_matrix, keys, strengths):
    normalized_memory = tf.nn.l2_normalize(tf.tanh(memory_matrix), 1)  # M=M/|M|
    normalized_keys = tf.nn.l2_normalize(tf.tanh(keys), 1)  # k=k/|k|

    similiarity = tf.matmul(normalized_keys,
                            tf.transpose(normalized_memory, [1, 0]))  # cosine sim: (batch_size, word_num)

    return tf.nn.softmax(similiarity * strengths, 1)  # each batch, every row of mem is multiplied with strength and then softmax
    # return gumbel_softmax(similiarity * strengths)


class ContentBasedMemoryUnitCell(DNCMemoryUnitCell):

    @property
    def state_size(self):
        init_memory = tf.TensorShape([self.h_N, self.h_W])
        init_usage_vector = tf.TensorShape([self.h_N])
        init_write_weighting = tf.TensorShape([self.h_N])
        init_read_weighting = tf.TensorShape([self.h_RH, self.h_N])
        return (init_memory, init_usage_vector, init_write_weighting, init_read_weighting)

    def zero_state(self, batch_size, dtype=tf.float32):

        init_memory = tf.fill([batch_size, self.h_N, self.h_W], tf.cast(1 / (self.h_N * self.h_W), dtype=dtype))
        init_usage_vector = tf.zeros([batch_size, self.h_N], dtype=dtype)
        init_write_weighting = unit_simplex_initialization(self.rng, batch_size, [self.h_N], dtype=dtype)
        init_read_weighting = unit_simplex_initialization(self.rng, batch_size, [self.h_RH, self.h_N], dtype=dtype)
        zero_states = (init_memory, init_usage_vector, init_write_weighting, init_read_weighting,)

        return zero_states

    def analyse_state(self, batch_size, dtype=tf.float32):

        alloc_gate = tf.zeros([batch_size, 1], dtype=dtype)
        free_gates = tf.zeros([batch_size, self.h_RH, 1], dtype=dtype)
        write_gate = tf.zeros([batch_size, 1], dtype=dtype)
        write_keys = tf.zeros([batch_size, 1, self.h_W], dtype=dtype)
        write_strengths = tf.zeros([batch_size, 1], dtype=dtype)
        write_vector = tf.zeros([batch_size, 1, self.h_W], dtype=dtype)
        erase_vector = tf.zeros([batch_size, 1, self.h_W], dtype=dtype)
        read_keys = tf.zeros([batch_size, self.h_RH, self.h_W], dtype=dtype)
        read_strengths = tf.zeros([batch_size, self.h_RH, 1], dtype=dtype)

        analyse_states = alloc_gate, free_gates, write_gate, write_keys, write_strengths, write_vector, \
                         erase_vector, read_keys, read_strengths

        return analyse_states

    def _weight_input(self, inputs):

        input_size = inputs.get_shape()[1].value
        total_signal_size = (3 + self.h_RH) * self.h_W + 2 * self.h_RH + 3

        with tf.variable_scope('{}'.format(self.name), reuse=self.reuse):
            w_x = tf.get_variable("mu_w_x", (input_size, total_signal_size),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            b_x = tf.get_variable("mu_b_x", (total_signal_size,), initializer=tf.constant_initializer(0.),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            weighted_input = tf.matmul(inputs, w_x) + b_x

            if self.dnc_norm:
                weighted_input = layer_norm(weighted_input, name='layer_norm', dtype=self.dtype)

        return weighted_input

    def _weight_input_mprogram(self, inputs):

        batch_size = inputs.get_shape()[0].value
        input_size = inputs.get_shape()[1].value
        total_signal_size = (3 + self.h_RH) * self.h_W + 2 * self.h_RH + 3

        with tf.variable_scope('{}'.format(self.name), reuse=self.reuse):
            w_k = tf.get_variable("w_p_k", (input_size, self.kS),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            w_sk = tf.get_variable("w_p_sk", (input_size, 1),
                                   initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                   collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            p_k = tf.get_variable("program_keys", (self.pS, self.kS),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            w_x = tf.get_variable("interface_weights", (self.pS, input_size, total_signal_size),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            b_x = tf.get_variable("interface_bias", (self.pS, total_signal_size,),
                                  initializer=tf.constant_initializer(0.),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)


            pkeys = tf.matmul(inputs, w_k)  # [bs, key_size]
            pstrengths = tf.nn.softplus(tf.matmul(inputs, w_sk))  # [bs, 1]

            w_ind = get_lookup_weighting(p_k, pkeys, pstrengths)  # [bs, ps]

            mW = tf.matmul(w_ind, tf.reshape(w_x,
                                             [self.pS, -1]))  # [bs,nn_out_size*if_size]
            mb = tf.matmul(w_ind, tf.reshape(b_x,
                                             [self.pS, -1]))  # [bs,if_size]
            mW = tf.reshape(mW, [batch_size, input_size, total_signal_size])
            weighted_input = tf.matmul(tf.expand_dims(inputs, axis=1), mW)  # [bs,1,if_size]
            weighted_input = tf.reshape(weighted_input, [batch_size, total_signal_size])
            weighted_input = weighted_input + mb

            if self.dnc_norm:
                weighted_input = layer_norm(weighted_input, name='layer_norm', dtype=self.dtype)

        return weighted_input

    def process_mp(self, inputs, w_k, w_sk, p_k, w_x, b_x, batch_size, input_size, total_signal_size):
        pkeys = tf.matmul(inputs, w_k)  # [bs, key_size]
        pstrengths = tf.nn.softplus(tf.matmul(inputs, w_sk))  # [bs, 1]

        w_ind = get_lookup_weighting(p_k, pkeys, pstrengths)  # [bs, ps]

        mW = tf.matmul(w_ind, tf.reshape(w_x,
                                         [self.pS, -1]))  # [bs,nn_out_size*if_size]
        mb = tf.matmul(w_ind, tf.reshape(b_x,
                                         [self.pS, -1]))  # [bs,if_size]
        mW = tf.reshape(mW, [batch_size, input_size, total_signal_size])
        weighted_input = tf.matmul(tf.expand_dims(inputs, axis=1), mW)  # [bs,1,if_size]
        weighted_input = tf.reshape(weighted_input, [batch_size, total_signal_size])
        weighted_input = weighted_input + mb
        return weighted_input

    def _weight_input_mprogram2(self, inputs):


        batch_size = inputs.get_shape()[0].value
        input_size = inputs.get_shape()[1].value
        total_signal_size = (3 + self.h_RH) * self.h_W + 2 * self.h_RH + 3
        total_signal_sizer = self.h_RH * self.h_W + self.h_RH*2
        total_signal_sizew = total_signal_size - total_signal_sizer



        with tf.variable_scope('{}'.format(self.name), reuse=self.reuse):
            w_k = tf.get_variable("w_p_k", (input_size, self.kS),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            w_sk = tf.get_variable("w_p_sk", (input_size, 1),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            p_k = tf.get_variable("program_keys", (self.pS, self.kS),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            w_x = tf.get_variable("interface_weights", (self.pS, input_size, total_signal_sizer),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            b_x = tf.get_variable("interface_bias", (self.pS, total_signal_sizer,), initializer=tf.constant_initializer(0.),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            w_k2 = tf.get_variable("w_p_k2", (input_size, self.kS),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            w_sk2 = tf.get_variable("w_p_sk2", (input_size, 1),
                                   initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                   collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            p_k2 = tf.get_variable("program_keys2", (self.pS, self.kS),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            w_x2 = tf.get_variable("interface_weights2", (self.pS, input_size, total_signal_sizew),
                                  initializer=tf.contrib.layers.xavier_initializer(seed=self.seed),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            b_x2 = tf.get_variable("interface_bias2", (self.pS, total_signal_sizew,),
                                  initializer=tf.constant_initializer(0.),
                                  collections=['memory_unit', tf.GraphKeys.GLOBAL_VARIABLES], dtype=self.dtype)

            weighted_inputr = self.process_mp(inputs, w_k, w_sk, p_k, w_x, b_x, batch_size, input_size, total_signal_sizer)
            weighted_inputw = self.process_mp(inputs, w_k2, w_sk2, p_k2, w_x2, b_x2, batch_size, input_size, total_signal_sizew)

            weighted_input = tf.concat([weighted_inputr, weighted_inputw], axis=-1)

            if self.dnc_norm:
                weighted_input = layer_norm(weighted_input, name='layer_norm', dtype=self.dtype)

        return weighted_input

    def __call__(self, inputs, pre_states, scope=None):

        self.h_B = inputs.get_shape()[0].value

        memory_ones, batch_memory_range = self._create_constant_value_tensors(self.h_B, self.dtype)
        self.const_memory_ones = memory_ones
        self.const_batch_memory_range = batch_memory_range

        pre_memory, pre_usage_vector, pre_write_weightings, pre_read_weightings = pre_states

        if self.pS <=1:
            weighted_input = self._weight_input(inputs)
        else:
            print('use multiple programs')
            weighted_input = self._weight_input_mprogram2(inputs)

        control_signals = self._create_control_signals(weighted_input)
        alloc_gate, free_gates, write_gate, write_keys, write_strengths, write_vector, \
        erase_vector, read_keys, read_strengths = control_signals

        alloc_weightings, usage_vector = self._update_alloc_and_usage_vectors(pre_write_weightings, pre_read_weightings,
                                                                              pre_usage_vector, free_gates)
        write_content_weighting = self._calculate_content_weightings(pre_memory, write_keys, write_strengths)
        write_weighting = self._update_write_weighting(alloc_weightings, write_content_weighting, write_gate,
                                                       alloc_gate)
        memory = self._update_memory(pre_memory, write_weighting, write_vector, erase_vector)
        read_content_weightings = self._calculate_content_weightings(memory, read_keys, read_strengths)
        read_vectors = self._read_memory(memory, read_content_weightings)

        read_vectors = tf.reshape(read_vectors, [self.h_B, self.h_W * self.h_RH])

        if self.bypass_dropout:
            input_bypass = tf.nn.dropout(inputs, self.bypass_dropout)
        else:
            input_bypass = inputs

        output = tf.concat([read_vectors, input_bypass], axis=-1)

        if self.analyse:
            output = (output, control_signals)

        return output, (memory, usage_vector, write_weighting, read_content_weightings)

    def _create_constant_value_tensors(self, batch_size, dtype):

        memory_ones = tf.ones([batch_size, self.h_N, self.h_W], dtype=dtype, name="memory_ones")

        batch_range = tf.range(0, batch_size, delta=1, dtype=tf.int32, name="batch_range")
        repeat_memory_length = tf.fill([self.h_N], tf.constant(self.h_N, dtype=tf.int32), name="repeat_memory_length")
        batch_memory_range = tf.matmul(tf.expand_dims(batch_range, -1), tf.expand_dims(repeat_memory_length, 0),
                                       name="batch_memory_range")
        return memory_ones, batch_memory_range

    def _create_control_signals(self, weighted_input):

        write_keys = weighted_input[:, :         self.h_W]  # W
        write_strengths = weighted_input[:, self.h_W:         self.h_W + 1]  # 1
        erase_vector = weighted_input[:, self.h_W + 1:       2 * self.h_W + 1]  # W
        write_vector = weighted_input[:, 2 * self.h_W + 1:       3 * self.h_W + 1]  # W
        alloc_gates = weighted_input[:, 3 * self.h_W + 1:       3 * self.h_W + 2]  # 1
        write_gates = weighted_input[:, 3 * self.h_W + 2:       3 * self.h_W + 3]  # 1
        read_keys = weighted_input[:, 3 * self.h_W + 3: (self.h_RH + 3) * self.h_W + 3]  # R * W
        read_strengths = weighted_input[:,
                         (self.h_RH + 3) * self.h_W + 3: (self.h_RH + 3) * self.h_W + 3 + 1 * self.h_RH]  # R
        free_gates = weighted_input[:, (self.h_RH + 3) * self.h_W + 3 + 1 * self.h_RH: (
                                                                                           self.h_RH + 3) * self.h_W + 3 + 2 * self.h_RH]

        alloc_gates = tf.sigmoid(alloc_gates, 'alloc_gates')
        free_gates = tf.sigmoid(free_gates, 'free_gates')
        free_gates = tf.expand_dims(free_gates, 2)
        write_gates = tf.sigmoid(write_gates, 'write_gates')

        write_keys = tf.expand_dims(write_keys, axis=1)
        write_strengths = oneplus(write_strengths)
        write_vector = tf.reshape(write_vector, [self.h_B, 1, self.h_W])
        erase_vector = tf.sigmoid(erase_vector, 'erase_vector')
        erase_vector = tf.reshape(erase_vector, [self.h_B, 1, self.h_W])

        read_keys = tf.reshape(read_keys, [self.h_B, self.h_RH, self.h_W])
        read_strengths = oneplus(read_strengths)
        read_strengths = tf.expand_dims(read_strengths, axis=2)

        return alloc_gates, free_gates, write_gates, write_keys, write_strengths, write_vector, \
               erase_vector, read_keys, read_strengths

