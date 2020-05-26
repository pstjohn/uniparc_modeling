# Patch to layernorm to allow mixed-precision training in TF2.1
# https://github.com/tensorflow/tensorflow/issues/35817
# https://github.com/tensorflow/tensorflow/commit/f9e899854cc96db28564fa65f22d32a647268fc1

from tensorflow.keras import layers

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class MixedLayerNormalization(layers.LayerNormalization):
  def call(self, inputs):
    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.shape
    ndims = len(input_shape)

    # Broadcasting only necessary for norm where the axis is not just
    # the last dimension
    broadcast_shape = [1] * ndims
    for dim in self.axis:
      broadcast_shape[dim] = input_shape.dims[dim].value
    def _broadcast(v):
      if (v is not None and len(v.shape) != ndims and
          self.axis != [ndims - 1]):
        return array_ops.reshape(v, broadcast_shape)
      return v

    if not self._fused:
      # Calculate the moments on the last axis (layer activations).
      mean, variance = nn.moments(inputs, self.axis, keep_dims=True)

      scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

      # Compute layer normalization using the batch_normalization function.
      outputs = nn.batch_normalization(
          inputs,
          mean,
          variance,
          offset=offset,
          scale=scale,
          variance_epsilon=self.epsilon)
    else:
      # Collapse dims before self.axis, and dims in self.axis
      pre_dim, in_dim = (1, 1)
      axis = sorted(self.axis)
      tensor_shape = array_ops.shape(inputs)
      for dim in range(0, ndims):
        dim_tensor = tensor_shape[dim]
        if dim < axis[0]:
          pre_dim = pre_dim * dim_tensor
        else:
          assert dim in axis
          in_dim = in_dim * dim_tensor

      squeezed_shape = [1, pre_dim, in_dim, 1]
      # This fused operation requires reshaped inputs to be NCHW.
      data_format = 'NCHW'

      inputs = array_ops.reshape(inputs, squeezed_shape)

      def _set_const_tensor(val, dtype, shape):
        return array_ops.fill(shape, constant_op.constant(val, dtype=dtype))

      # self.gamma and self.beta have the wrong shape for fused_batch_norm, so
      # we cannot pass them as the scale and offset parameters. Therefore, we
      # create two constant tensors in correct shapes for fused_batch_norm and
      # later constuct a separate calculation on the scale and offset.
      scale = _set_const_tensor(1.0, self.dtype, [pre_dim])
      offset = _set_const_tensor(0.0, self.dtype, [pre_dim])

      # Compute layer normalization using the fused_batch_norm function.
      outputs, _, _ = nn.fused_batch_norm(
          inputs,
          scale=scale,
          offset=offset,
          epsilon=self.epsilon,
          data_format=data_format)

      outputs = array_ops.reshape(outputs, tensor_shape)

      scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

      if scale is not None:
        outputs = outputs * math_ops.cast(scale, outputs.dtype)
      if offset is not None:
        outputs = outputs + math_ops.cast(offset, outputs.dtype)

    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    return outputs