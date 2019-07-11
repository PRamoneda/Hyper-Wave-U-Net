import tensorflow as tf

import Utils

import Models.Mhe
from Models.Mhe import add_thomson_constraint_final

def independent_outputs(featuremap, source_names, num_channels, filter_width, padding, activation):
    outputs = dict()
    for name in source_names:
        outputs[name] = tf.layers.conv1d(featuremap, num_channels, filter_width, activation=activation, padding=padding)
    return outputs

#Models.OutputLayer.d(cropped_input, current_layer, self.source_names, self.num_channels, self.output_filter_size, self.padding, out_activation, training, self.mhe, self.mhe_power)
def difference_output(input_mix, featuremap, source_names, num_channels, filter_width, padding, activation, training, mhe, mhe_power):
    outputs = dict()
    sum_source = 0
    for name in source_names[:-1]:
        # Variable scope corresponding to each output layer
        with tf.variable_scope("output_"+name, reuse=reuse):
            # Define weights tensor
            n_filt = filter_width
            shape = [filter_width, num_channels, n_filt]
            W = tf.get_variable('W', shape=shape, initializer=tf.random_normal_initializer())
            
            # Add MHE (thompson constraint) to the collection when in use
            if mhe:
                add_thomson_constraint_final(W, n_filt, mhe_power)
             
            # Change implementation to tf.nn.conv1d
            out = tf.nn.conv1d(featuremap, W, padding=padding)
            out = out_activation(out)
            outputs[name] = out
            sum_source = sum_source + out

    # Compute last source based on the others
    last_source = Utils.crop(input_mix, sum_source.get_shape().as_list()) - sum_source # input_mix: contains all sources
    last_source = Utils.AudioClip(last_source, training)
    outputs[source_names[-1]] = last_source
    return outputs