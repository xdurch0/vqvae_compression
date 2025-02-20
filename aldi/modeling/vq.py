from typing import Optional, Callable

import tensorflow as tf
from sklearn.cluster import KMeans, MiniBatchKMeans
tfkl = tf.keras.layers


class RVQ(tfkl.Layer):
    def __init__(self,
                 codebook_size: int,
                 dim_code: int,
                 n_levels: int,
                 init_scale: float = 1.,
                 **kwargs):
        super().__init__(**kwargs)
        self.codebooks = [
            tf.Variable(init_scale * tf.random.normal((codebook_size, dim_code)),
                        trainable=False)
            for _ in range(n_levels)]
        self.codebook_size = codebook_size

        self.cb_e_avgs = [
            tf.Variable(tf.zeros((codebook_size, dim_code)), trainable=False)
            for _ in range(n_levels)]
        self.cb_n_avgs = [
            tf.Variable(tf.zeros((codebook_size,)), trainable=False)
            for _ in range(n_levels)]

    def call(self,
             inputs,
             with_indices: bool = False,
             up_to: Optional[int] = None):
        all_residuals = []
        all_quantized = []
        if with_indices:
            all_indices = []
        # quantize for each codebook
        quantized = tf.zeros_like(inputs)
        residual = inputs

        if up_to is not None:
            codebooks = self.codebooks[:up_to + 1]
        else:
            codebooks = self.codebooks

        for codebook in codebooks:
            all_residuals.append(residual)

            quantized_here = self.quantize2(residual, codebook, with_indices)
            if with_indices:
                quantized_here, indices_here = quantized_here
                all_indices.append(indices_here)
            quantized += quantized_here
            residual -= quantized_here

            all_quantized.append(quantized_here)

        quantized = inputs + tf.stop_gradient(quantized - inputs)
        if with_indices:
            return quantized, all_residuals, all_quantized, all_indices
        return quantized, all_residuals, all_quantized

    def call_wrong(self, inputs, with_indices=False, up_to=None):
        all_residuals = []
        all_quantized = []
        if with_indices:
            all_indices = []
        # quantize for each codebook
        quantized = tf.zeros_like(inputs)
        residual = inputs

        if up_to is not None:
            codebooks = self.codebooks[:up_to + 1]
        else:
            codebooks = self.codebooks

        for codebook in codebooks:
            all_residuals.append(residual)

            quantized_here = self.quantize2(residual, codebook, with_indices)
            if with_indices:
                quantized_here, indices_here = quantized_here
                all_indices.append(indices_here)
            quantized += residual + tf.stop_gradient(quantized_here - residual)
            residual -= quantized_here

            all_quantized.append(quantized_here)

        if with_indices:
            return quantized, all_residuals, all_quantized, all_indices
        return quantized, all_residuals, all_quantized

    def call_not_wrong(self, inputs, with_indices=False, up_to=None):
        all_residuals = []
        all_quantized = []
        if with_indices:
            all_indices = []
        # quantize for each codebook
        quantized = tf.zeros_like(inputs)
        residual = inputs

        if up_to is not None:
            codebooks = self.codebooks[:up_to + 1]
        else:
            codebooks = self.codebooks

        for codebook in codebooks:
            all_residuals.append(residual)

            quantized_here = self.quantize2(residual, codebook, with_indices)
            if with_indices:
                quantized_here, indices_here = quantized_here
                all_indices.append(indices_here)
            quantized += residual + tf.stop_gradient(quantized_here - residual)
            residual = inputs - quantized

            all_quantized.append(quantized_here)

        if with_indices:
            return quantized, all_residuals, all_quantized, all_indices
        return quantized, all_residuals, all_quantized

    def quantize(self,
                 encoder_outputs: tf.Tensor,
                 codebook: tf.Tensor,
                 with_indices: bool = False):
        distances = tf.reduce_sum(tf.square(encoder_outputs[:, :, None, :]
                                            - codebook[None, None, :, :]),
                                  axis=-1)

        min_distance_inds = tf.math.argmin(distances, axis=-1)
        codes = tf.gather(codebook, min_distance_inds)

        if with_indices:
            return codes, min_distance_inds
        return codes

    def quantize2(self,
                  encoder_outputs: tf.Tensor,
                  codebook: tf.Tensor,
                  with_indices: bool = False):
        encoder_flat = tf.reshape(encoder_outputs,
                                  [-1, tf.shape(encoder_outputs)[-1]])

        dotprod = tf.matmul(encoder_flat, codebook, transpose_b=True)
        distances = (tf.reduce_sum(encoder_flat ** 2, axis=-1)[:, None]
                     - 2 * dotprod
                     + tf.reduce_sum(codebook ** 2, axis=-1)[None])

        min_distance_inds = tf.math.argmin(distances, axis=-1)
        codes = tf.gather(codebook, min_distance_inds)

        codes = tf.reshape(codes, tf.shape(encoder_outputs))
        if with_indices:
            min_distance_inds = tf.reshape(min_distance_inds,
                                           tf.shape(encoder_outputs)[:-1])
            return codes, min_distance_inds
        return codes

    def init_with_k_means(self,
                          encoded_batch: tf.Tensor,
                          n_init: int = 10,
                          batch_n_multiplier: int = 1,
                          minibatch_size: int = 0):
        for index in range(len(self.codebooks)):
            print("Running K-Means for index {}...".format(index))
            if index > 0:
                quantized_partial = self(encoded_batch, up_to=index - 1)[0]
                use_batch = encoded_batch - quantized_partial
            else:
                use_batch = encoded_batch
            numpy_batch = use_batch.numpy().reshape((-1, use_batch.shape[-1]))

            if minibatch_size > 0:
                kmeans = MiniBatchKMeans(n_clusters=self.codebooks[index].shape[0],
                                         n_init=n_init,
                                         batch_size=minibatch_size)
            else:
                kmeans = KMeans(n_clusters=self.codebooks[index].shape[0],
                                n_init=n_init)
            kmeans.fit(numpy_batch)
            self.codebooks[index].assign(kmeans.cluster_centers_)

        # now we also need to initialize the stuff for EMA
        _, all_residuals, _, all_indices = self(encoded_batch,
                                                with_indices=True)
        for index, (residual, indices) in enumerate(
                zip(all_residuals, all_indices)):
            flat_indices = tf.one_hot(tf.reshape(indices, (-1,)),
                                      depth=self.codebook_size)
            flat_encoder = tf.reshape(residual, [-1, tf.shape(residual)[-1]])

            cb_update_e = tf.matmul(flat_indices, flat_encoder,
                                    transpose_a=True)  # k x c
            cb_update_n = tf.reduce_sum(flat_indices, axis=0) # k

            self.cb_e_avgs[index].assign(cb_update_e / batch_n_multiplier)
            self.cb_n_avgs[index].assign(cb_update_n / batch_n_multiplier)


class Autoencoder(tf.keras.Model):
    def __init__(self,
                 inputs: tf.Tensor,
                 encoder: tfkl.Layer,
                 decoder: tfkl.Layer,
                 loss_fn: Callable,
                 quantizer: Optional[tfkl.Layer] = None,
                 commitment_style: str = "all",
                 beta: float = 0.,
                 lambda_: float = 0.,
                 gamma: float = 0.99,
                 **kwargs):
        if quantizer is None:
            super().__init__(inputs, decoder(encoder(inputs)), **kwargs)
        else:
            super().__init__(inputs, decoder(quantizer(encoder(inputs))[0]),
                             **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.commitment_style = commitment_style

        self.lambda_ = lambda_
        self.beta = beta
        self.gamma = gamma

        self.loss_tracker = tf.keras.metrics.Mean("loss")
        self.l2_tracker = tf.keras.metrics.Mean("code_l2")
        if quantizer is not None:
            self.codebook_loss_tracker = tf.keras.metrics.Mean("codebook_loss")

        self.loss_fn = loss_fn

    def train_step(self, data):
        with tf.GradientTape() as tape:
            encoder_outputs = self.encoder(data, training=True)

            if self.quantizer is not None:
                codes, all_residuals, all_quantized, all_indices = self.quantizer(
                    encoder_outputs, with_indices=True)
            else:
                codes = encoder_outputs
            reconstructions = self.decoder(codes, training=True)
            # reconstructions = self.decoder(encoder_outputs, training=True)

            activity_l2 = tf.reduce_mean(encoder_outputs ** 2)
            loss = self.loss_fn(data, reconstructions)

            total_loss = loss + self.lambda_ * activity_l2

            if self.quantizer is not None:
                if self.commitment_style == "all":
                    commitment_loss = 0
                    # note, in each pair, residual is the residual before quantization
                    # i.e. the thing that is being quantized.
                    # so this compares encoder output vs code, so to say.
                    for residual, quantized in zip(all_residuals, all_quantized):
                        commitment_loss += tf.reduce_mean(tf.reduce_sum(
                            tf.square(residual - tf.stop_gradient(quantized)),
                            axis=-1))

                    commitment_loss /= len(self.quantizer.codebooks)
                elif self.commitment_style == "final":
                    commitment_loss = tf.reduce_mean(tf.reduce_sum(
                        tf.square(encoder_outputs - tf.stop_gradient(codes)),
                        axis=-1))
                else:
                    raise ValueError
                total_loss += self.beta * commitment_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # update codebook
        # get encoding matrix E ~ (b*t) x c
        # get codebook usage matrix U ~ b*t x k
        # then U.T * E has the correct shape, k x c but is it the correct thing?
        # in U.T, each row says: for this codebook entry, these are the encodings that were assigned here
        # so it seems like we are contracting the correct rows!
        if self.quantizer is not None:
            for index, (residual, indices) in enumerate(
                    zip(all_residuals, all_indices)):
                flat_indices = tf.one_hot(tf.reshape(indices, (-1,)),
                                          depth=self.quantizer.codebook_size)
                flat_encoder = tf.reshape(residual, [-1, tf.shape(residual)[-1]])

                cb_update_e = tf.matmul(flat_indices, flat_encoder,
                                        transpose_a=True)  # k x c
                cb_update_n = tf.reduce_sum(flat_indices, axis=0)  # k

                self.quantizer.cb_e_avgs[index].assign(
                    self.gamma * self.quantizer.cb_e_avgs[index]
                    + (1 - self.gamma) * cb_update_e)
                self.quantizer.cb_n_avgs[index].assign(
                    self.gamma * self.quantizer.cb_n_avgs[index]
                    + (1 - self.gamma) * cb_update_n)

                self.quantizer.codebooks[index].assign(
                    self.quantizer.cb_e_avgs[index]
                    / (self.quantizer.cb_n_avgs[index][:, None] + 1e-8))

        self.loss_tracker.update_state(loss)
        self.l2_tracker.update_state(activity_l2)
        result_dict = {"loss": self.loss_tracker.result(),
                       "code_l2": self.l2_tracker.result()}
        if self.quantizer is not None:
            self.codebook_loss_tracker.update_state(commitment_loss)
            result_dict.update({"codebook_loss": self.codebook_loss_tracker.result()})

        return result_dict

    def test_step(self, data):
        encoder_outputs = self.encoder(data, training=False)
        if self.quantizer is not None:
            codes, all_residuals, all_quantized = self.quantizer(encoder_outputs)
        else:
            codes = encoder_outputs
        reconstructions = self.decoder(codes, training=False)
        # reconstructions = self.decoder(encoder_outputs, training=True)

        activity_l2 = tf.reduce_mean(encoder_outputs ** 2)
        loss = self.loss_fn(data, reconstructions)

        if self.quantizer is not None:
            if self.commitment_style == "all":
                commitment_loss = 0
                # note, in each pair, residual is the residual before quantization
                # i.e. the thing that is being quantized.
                # so this compares encoder output vs code, so to say.
                for residual, quantized in zip(all_residuals, all_quantized):
                    commitment_loss += tf.reduce_mean(tf.reduce_sum(
                        tf.square(residual - tf.stop_gradient(quantized)),
                        axis=-1))

                commitment_loss /= len(self.quantizer.codebooks)
            elif self.commitment_style == "final":
                commitment_loss = tf.reduce_mean(tf.reduce_sum(
                    tf.square(encoder_outputs - tf.stop_gradient(codes)),
                    axis=-1))
            else:
                raise ValueError

        self.loss_tracker.update_state(loss)
        self.l2_tracker.update_state(activity_l2)
        result_dict = {"loss": self.loss_tracker.result(),
                       "code_l2": self.l2_tracker.result()}
        if self.quantizer is not None:
            self.codebook_loss_tracker.update_state(commitment_loss)
            result_dict.update({"codebook_loss": self.codebook_loss_tracker.result()})

        return result_dict
