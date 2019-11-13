import numpy as np
import tensorflow as tf
import sentencepiece as spm


def create_masked_input_dataset(language_model_path,
                                sequence_path,
                                max_sequence_length=512,
                                batch_size=20,
                                buffer_size=1024,
                                vocab_size=32000,
                                mask_index=4,
                                vocab_start=5,
                                fix_sequence_length=False,
                                masking_freq=.15,
                                mask_token_freq=.8,
                                mask_random_freq=.1,
                                ):


    sp = spm.SentencePieceProcessor()
    sp.Load(language_model_path)

    def sp_encode(line_tensor):
        encoded_array = np.asarray(
            sp.SampleEncodeAsIds(line_tensor.numpy(), nbest_size=-1, alpha=0.5))

        # If the protein sequence is too long, take a random slice.
        if len(encoded_array) > max_sequence_length:
            offset = np.random.randint(
                low=0, high=len(encoded_array) - max_sequence_length + 1)
            encoded_array = encoded_array[offset:(offset + max_sequence_length)]

        return encoded_array

    def sp_decode(line_tensor):
        return sp.DecodeIds(line_tensor.numpy().tolist())

    def sp_encode_tf(line_tensor):
        return tf.py_function(sp_encode, inp=[line_tensor], Tout=[tf.int32,])

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

        mask_score = tf.random.uniform(input_tensor.shape, maxval=1, dtype=tf.float32)
        input_mask = mask_score < masking_freq

        # Mask with [MASK] token 80% of the time
        mask_mask = mask_score <= 0.15 * mask_token_freq

        # Mask with random token 10% of the time
        mask_random = (mask_score >= 0.15 * (1. - mask_random_freq)) & input_mask

        # Tensors to replace with where input is masked or randomized
        mask_value_tensor = tf.ones(input_tensor.shape, dtype=tf.int32) * mask_index
        random_value_tensor = tf.random.uniform(
            input_tensor.shape, minval=vocab_start, maxval=vocab_size, dtype=tf.int32)
        pad_value_tensor = tf.zeros(input_tensor.shape, dtype=tf.int32)

        # Use the replacements to mask the input tensor
        masked_tensor = tf.where(mask_mask, mask_value_tensor, input_tensor)
        masked_tensor = tf.where(mask_random, random_value_tensor, masked_tensor)
        
        # Set true values to zero (pad value) where not masked
        true_tensor = tf.where(input_mask, input_tensor, pad_value_tensor)

        return masked_tensor, input_mask, true_tensor


    def mask_input_tf(input_tensor):
        a, b, c = tf.py_function(mask_input, inp=[input_tensor],
                                 Tout=[tf.int32, tf.bool, tf.int32])
        return (a, b), c


    valid_data = tf.data.TextLineDataset(sequence_path)

    encoded_data = valid_data\
        .map(sp_encode_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .map(mask_input_tf, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    
    # This argument controls whether to fix the size of the sequences
    tf_seq_len = -1 if not fix_sequence_length else max_sequence_length

    encoded_data = encoded_data\
        .shuffle(buffer_size=buffer_size)\
        .padded_batch(batch_size, padded_shapes=(
            ([tf_seq_len], [tf_seq_len]), [tf_seq_len]))
        

    return encoded_data
