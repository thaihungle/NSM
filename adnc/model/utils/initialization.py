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
import numpy as np
import tensorflow as tf


def unit_simplex_initialization(rng, batch_size, shape, dtype=tf.float32):
    mat = []
    for i in range(batch_size):
        mat.append(rng.uniform(0, 1 / np.sum(shape), shape))
    mat = np.array(mat)
    return tf.constant(mat, dtype=dtype)
