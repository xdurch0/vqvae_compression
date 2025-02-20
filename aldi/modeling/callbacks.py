from typing import Union, Iterable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class ReconstructionPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 reference_images: Union[tf.Tensor, np.ndarray],
                 frequency: int,
                 clip: bool = False,
                 colormap: str = "Greys",
                 **kwargs):
        super().__init__(**kwargs)
        self.frequency = frequency
        self.images = reference_images
        self.clip = clip
        self.colormap = colormap

    def on_epoch_begin(self,
                       epoch,
                       logs=None):
        # TODO do not hardcode number of images, subplot shape
        if not epoch % self.frequency:
            reconstructed_batch = self.model(self.images[:16]).numpy()
            if self.clip:
                reconstructed_batch = np.clip(reconstructed_batch, 0, 1)

            plt.figure(figsize=(15, 15))
            for ind, img in enumerate(reconstructed_batch):
                concat = np.concatenate([self.images[ind], img], axis=1)
                plt.subplot(4, 4, ind + 1)
                plt.imshow(concat, cmap=self.colormap)
                plt.axis("off")
            plt.show()


class CodebookResetter(tf.keras.callbacks.Callback):
    def __init__(self,
                 frequency: int,
                 threshold: float,
                 iteration_source: Iterable,
                 verbose: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.frequency = frequency
        self.threshold = threshold
        self.iteration_source = iteration_source
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if not epoch % self.frequency:
            batches = []
            ind = 0
            for batch in self.iteration_source:
                batches.append(batch)
                ind += 1
                if ind >= 1:
                    break
            reference_batch = tf.concat(batches, axis=0)

            reference_encodings = self.model.encoder(reference_batch)
            for index in range(len(self.model.quantizer.codebooks)):
                average_usage = self.model.quantizer.cb_n_avgs[index]
                unused_code_indices = np.where(average_usage < self.threshold)[0]
                if len(unused_code_indices):
                    # re-do residuals each iteration to use updated codebook
                    _, residuals, _ = self.model.quantizer(reference_encodings,
                                                           up_to=index)

                    # TODO adapt so it also works for 1D
                    b, w, h = residuals[index].shape[:-1]
                    new_codebook_entries = tf.stack(
                        [residuals[index][np.random.choice(b),
                         np.random.choice(w), np.random.choice(h)]
                         for _ in unused_code_indices],
                        axis=0)
                    if self.verbose:
                        print("DEBUG NEW ENTRIES (INDEX {})".format(index),
                              new_codebook_entries.shape)

                    sparse_update = tf.IndexedSlices(new_codebook_entries,
                                                     tf.convert_to_tensor(
                                                         unused_code_indices,
                                                         dtype=tf.int32))
                    self.model.quantizer.codebooks[index].scatter_update(
                        sparse_update)

                    # TODO this can be improved, induces some numerical error
                    # i.e. do sparse_update on cb_e_avgs as well using
                    # corresponding N
                    self.model.quantizer.cb_e_avgs[index].assign(
                        self.model.quantizer.codebooks[index]
                        * self.model.quantizer.cb_n_avgs[index][:, None])
