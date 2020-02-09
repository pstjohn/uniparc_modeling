import numpy as np
import tensorflow as tf

    
vocab = ['^', '$', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

encoding_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=vocab, values=tf.range(len(vocab)) + 2),
    default_value=0)


def create_masked_input_dataset(sequence_path,
                                sequence_compression='GZIP',
                                max_sequence_length=512,
                                batch_size=20,
                                buffer_size=1024,
                                mask_index=1,
                                vocab_start=4,
                                fix_sequence_length=False,
                                masking_freq=.15,
                                mask_token_freq=.8,
                                mask_random_freq=.1,
                                filter_bzux=True,
                                no_mask_pad=1,
                                shard_num_workers=None,
                                shard_worker_index=None):

    @tf.function
    def encode(x):
        chars = tf.strings.bytes_split(x)
        
        # Append start and end tokens
        chars = tf.concat([tf.constant(['^']), chars, tf.constant(['$'])], 0)
        
        # If chars is greater than max_sequence_length, take a random crop
        chars = tf.cond(tf.shape(chars) > max_sequence_length,
                lambda: tf.image.random_crop(chars, (max_sequence_length,)),
                lambda: chars)

        return encoding_table.lookup(chars)


    @tf.function
    def mask_input(input_tensor):
        """ Randomly mask the input tensor according to the formula perscribed by BERT. 
        Randomly masks 15% of input tokens, with 80% recieving the [MASK] token,
        10% randomized, 10% left unchanged. 

        Returns
        -------

        masked_tensor: (batch_size, seq_length) 
            Tensor with masked values
        input_tensor: (batch_size, seq_length)
            Original input tensor (true values)
        input_mask: (batch_size, seq_length)
            Boolean mask that selects the desired inputs.    
        """

        input_shape = tf.shape(input_tensor)
        mask_score = tf.random.uniform(input_shape - no_mask_pad * 2, maxval=1, dtype=tf.float32)
        # Ensure that no_mask_pad tokens on edges are not masked
        mask_score = tf.concat([tf.ones(no_mask_pad), mask_score, tf.ones(no_mask_pad)], 0)
        input_mask = mask_score < masking_freq

        # Mask with [MASK] token 80% of the time
        mask_mask = mask_score <= masking_freq * mask_token_freq

        # Mask with random token 10% of the time
        mask_random = (mask_score >= masking_freq * (1. - mask_random_freq)) & input_mask

        # Tensors to replace with where input is masked or randomized
        mask_value_tensor = tf.ones(input_shape, dtype=tf.int32) * mask_index
        random_value_tensor = tf.random.uniform(
            input_shape, minval=vocab_start, maxval=len(vocab) + 2, dtype=tf.int32)
        pad_value_tensor = tf.zeros(input_shape, dtype=tf.int32)

        # Use the replacements to mask the input tensor
        masked_tensor = tf.where(mask_mask, mask_value_tensor, input_tensor)
        masked_tensor = tf.where(mask_random, random_value_tensor, masked_tensor)

        # Make 1 correspond to the first amino acid in the output.
        label_offset = vocab_start - 1 
        input_offset = input_tensor - label_offset        
        
        # Set true values to zero (pad value) where not masked
        true_tensor = tf.where(input_mask, input_offset, pad_value_tensor)

        return masked_tensor, true_tensor

    dataset = tf.data.TextLineDataset(
        sequence_path, compression_type=sequence_compression)\
        .shuffle(buffer_size=buffer_size)\
        .repeat()
    
    if shard_num_workers:
        dataset = dataset.shard(shard_num_workers, shard_worker_index)

    if filter_bzux:
        bzux_filter = lambda string: tf.math.logical_not(
            tf.strings.regex_full_match(string, '.*[BZUOX].*'))
        dataset = dataset.filter(bzux_filter)

    # This argument controls whether to fix the size of the sequences
    tf_seq_len = -1 if not fix_sequence_length else max_sequence_length

    dataset = dataset\
        .map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .map(mask_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .padded_batch(batch_size, padded_shapes=(
            ([tf_seq_len], [tf_seq_len])))\
        .prefetch(tf.data.experimental.AUTOTUNE)
        
    return dataset
