import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt

print("Num GPUs Available: ", tf.config.experimental.list_physical_devices())
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tb_path = "./logs/"
tf.compat.v1.disable_eager_execution()


class CGHNet:
    def __init__(self, params, phase_factors):
        self._trained = False
        self.params = params  # if params else cfg.CGHNET.default
        self.phase_factors = phase_factors
        self.params["input_shape"] = (self.params["Mx"], self.params["My"], self.params["nz"])
        self._fix_compatibility()
        self.params["IF"] = 8

    def _fix_compatibility(self):
        w_bs = [max(1, w / b) for w, b in zip(self.params["ws"], self.params["bs"])]
        gs = [int(min(g, v)) for g, v in zip(self.params["gs"], w_bs)]
        w_bs = [int(round(w_b / g) * g) for w_b, g in zip(w_bs, gs)]
        self.params["ws"] = [int(w_b * b) for w_b, b in zip(w_bs, self.params["bs"])]
        self.params["gs"] = gs

    def _prop_to_slm(self, inputs):
        # We need to propagate the input backwards to the SLM with ifft2
        real, imag = inputs
        field_z0 = tf.complex(tf.squeeze(real), 0.) * tf.exp(tf.complex(0., tf.squeeze(imag)))
        shift = tf.signal.fftshift(field_z0, axes=[1, 2])
        slm = tf.math.angle(tf.signal.ifftshift(tf.signal.ifft2d(shift)))
        return tf.expand_dims(slm, axis=-1)

    def _prop_to_planes(self, slm_phase):
        # Then propagate to the z planes we have defined
        phi_slm = tf.complex(np.float32(0.), tf.squeeze(slm_phase))
        phi_slm = tf.math.exp(phi_slm)

        output_list = []
        for factor in self.phase_factors:
            phased_slm_layer = tf.multiply(phi_slm, factor)
            fft = tf.signal.fftshift(tf.signal.fft2d(phased_slm_layer))
            I = tf.cast(tf.math.square(tf.math.abs(fft)), tf.float32)
            output_list.append(tf.squeeze(I))
        return tf.stack(output_list, axis=3)

    def _loss_func(self, y_true, y_pred):
        y_predict = self._prop_to_planes(y_pred)
        num = tf.reduce_sum(y_predict * y_true, axis=[1, 2, 3])
        denom = tf.sqrt(
            tf.reduce_sum(tf.pow(y_predict, 2), axis=[1, 2, 3]) * tf.reduce_sum(tf.pow(y_true, 2), axis=[1, 2, 3]))
        return 1 - tf.reduce_mean((num + 1) / (denom + 1), axis=0)

    def _unet(self):
        def setup(prev, cx):
            x, cx = self._stage(self.params["ds"][0], self.params["bs"][0], self.params["gs"][0], self.params["ws"][0],
                                1, sampling="down")(prev, cx)  # r/2 x r/2
            x, cx = self._stage(self.params["ds"][1], self.params["bs"][1], self.params["gs"][1], self.params["ws"][1],
                                2, sampling="down")(x, cx)  # r/4 x r/4
            x, cx = self._stage(self.params["ds"][2], self.params["bs"][2], self.params["gs"][2], self.params["ws"][2],
                                3, sampling="up")(x, cx)  # r/2 x r/2
            x, cx = self._stage(self.params["ds"][3], self.params["bs"][3], self.params["gs"][3], self.params["ws"][3],
                                4, sampling="up")(x, cx)  # r x r
            return x, cx

        return setup

    def _stage(self, di, bi, gi, wi, stage_no, sampling="down"):
        def setup(prev, cx):
            x, cx = self._block2s(bi, gi, wi, stage_no, 1, sampling=sampling)(prev, cx)
            for block_no in range(1, di):
                x, cx = self._block1s(bi, gi, wi, stage_no, block_no + 1)(x, cx)
            return x, cx

        return setup

    def _block1s(self, bi, gi, wi, stage_no, block_no):
        def setup(prev, cx):
            w_b = int(round(wi / bi))
            groups = w_b // gi
            x = layers.Conv2D(w_b, (1, 1), activation='relu', padding='same',
                              name="Stage-" + str(stage_no) + "-block-" + str(block_no))(prev)
            cx = conv2d_cx(cx, prev.shape[-1], w_b, k=1)
            x = layers.BatchNormalization()(x)
            cx = batchnorm2d_cx(cx, w_b)
            x = layers.Conv2D(w_b, (3, 3), activation='relu', padding='same')(x)
            cx = conv2d_cx(cx, w_b, w_b, k=3, groups=gi)
            x = layers.BatchNormalization()(x)
            cx = batchnorm2d_cx(cx, w_b)
            x = layers.Conv2D(wi, (1, 1), activation='relu', padding='same')(x)
            cx = conv2d_cx(cx, w_b, wi, k=1)
            x = layers.BatchNormalization()(x)
            cx = batchnorm2d_cx(cx, wi)
            x = layers.add([prev, x])
            return x, cx

        return setup

    def _block2s(self, bi, gi, wi, stage_no, block_no, sampling="down"):
        def setup(prev, cx):
            w_b = int(round(wi / bi))
            groups = w_b // gi
            stride = 1
            x = layers.Conv2D(w_b, (1, 1), activation='relu', padding='same',
                              name="Stage-" + str(stage_no) + "-block-" + str(block_no))(prev)
            cx = conv2d_cx(cx, prev.shape[-1], w_b, k=1)
            x = layers.BatchNormalization()(x)
            cx = batchnorm2d_cx(cx, w_b)
            if sampling == "down":
                res = layers.Conv2D(wi, (1, 1), activation='relu', strides=2, padding='same')(prev)
                h, w = cx["h"], cx["w"]
                cx = conv2d_cx(cx, prev.shape[-1], wi, k=1, stride=2)
                cx["h"] = h
                cx["w"] = w
                stride = 2
            elif sampling == "up":
                cx = conv2d_cx(cx, prev.shape[-1], wi, k=1, stride=1)
                cx["h"] = cx["h"] * 2
                cx["w"] = cx["w"] * 2
                res = layers.Conv2D(wi, (1, 1), activation='relu', strides=1, padding='same')(prev)
                res = layers.UpSampling2D((2, 2))(res)
                x = layers.UpSampling2D((2, 2))(x)
                stride = 1
            elif sampling == "none":
                cx = conv2d_cx(cx, prev.shape[-1], wi, k=1, stride=1)
                res = layers.Conv2D(wi, (1, 1), activation='relu', strides=1, padding='same')(prev)
                stride = 1
            x = layers.Conv2D(w_b, (3, 3), activation='relu', strides=stride, padding='same')(x)
            cx = conv2d_cx(cx, w_b, w_b, k=3, groups=groups, stride=stride)
            x = layers.BatchNormalization()(x)
            cx = batchnorm2d_cx(cx, w_b)
            x = layers.Conv2D(wi, (1, 1), activation='relu', padding='same')(x)
            cx = conv2d_cx(cx, w_b, wi, k=1)
            x = layers.BatchNormalization()(x)
            cx = batchnorm2d_cx(cx, wi)
            x = layers.add([res, x])
            return x, cx

        return setup

    def _interleave(self, input):
        return tf.nn.space_to_depth(input=input, block_size=self.params["IF"])

    def _deinterleave(self, input):
        return tf.nn.depth_to_space(input=input, block_size=self.params["IF"])

    def _branching(self):
        def setup(prev, cx):
            h, w = cx["h"], cx["w"]
            real_branch, cx = self._stage(self.params["ds"][-2], self.params["bs"][-2], self.params["gs"][-2],
                                          self.params["ws"][-2], 5, sampling="none")(prev, cx)
            cx = conv2d_cx(cx, self.params["ws"][-2], self.params["IF"] ** 2, k=1)
            real_branch = layers.Conv2D(self.params["IF"] ** 2, (1, 1), padding='same', activation='relu')(real_branch)
            cx["h"] = h
            cx["w"] = w
            imag_branch, cx = self._stage(self.params["ds"][-1], self.params["bs"][-1], self.params["gs"][-1],
                                          self.params["ws"][-1], 6, sampling="none")(prev, cx)
            cx = conv2d_cx(cx, self.params["ws"][-1], self.params["IF"] ** 2, k=1)
            imag_branch = layers.Conv2D(self.params["IF"] ** 2, (1, 1), padding='same', activation='relu')(imag_branch)

            de_int_real = layers.Lambda(self._deinterleave, name="De-interleave_real")(real_branch)
            de_int_imag = layers.Lambda(self._deinterleave, name="De-interleave_imag")(imag_branch)

            slm_phase = layers.Lambda(self._prop_to_slm, name="SLM_phase")([de_int_real, de_int_imag])

            return slm_phase, cx

        return setup

    def build_model(self):

        cx = {"h": self.params["Mx"], "w": self.params["My"], "flops": 0, "params": 0, "acts": 0}
        inp = Input(shape=(self.params["Mx"], self.params["My"], self.params["nz"]), name='Input',
                    batch_size=self.params["batch_size"])
        interleaved = layers.Lambda(self._interleave, name="interleave")(inp)
        cx["h"] = interleaved.shape[1]
        cx["w"] = interleaved.shape[2]
        unet, cx = self._unet()(interleaved, cx)

        branches, cx = self._branching()(unet, cx)

        self.params["complexity"] = {"flops": cx["flops"], "params": cx["params"], "acts": cx["acts"]}

        return Model(inp, branches), cx

    def train_network(self, epochs, nT, training_data):
        train_path, val_path = training_data

        self.model, _ = self.build_model()

        self.model.compile(
            loss=self._loss_func,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["acc"]
        )

        train_input_fn = self._get_input_fn(train_path, nT, epochs)
        eval_input_fn = self._get_input_fn(val_path, nT, epochs)

        training_history = self.model.fit(
            train_input_fn,
            epochs=epochs,
            steps_per_epoch=nT // self.params["batch_size"],
        )


        self._save_training_result(training_history.history)

        clear_session()

        return self.params

    def _read_tfrecord(self, path):
        tfrecord_format = ({"image": tf.io.FixedLenFeature([], tf.string)})
        example = tf.io.parse_single_example(path, tfrecord_format)
        image = tf.cast(tf.reshape(tf.io.decode_raw(example['image'], tf.float64), self.params["input_shape"]),
                        tf.float32)
        return image, image

    def _load_dataset(self, filenames):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
        dataset = dataset.with_options(
            ignore_order)  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(self._read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def _get_input_fn(self, filenames, nT, epochs):
        dataset = self._load_dataset(filenames)
        dataset = dataset.shuffle(nT)
        dataset = dataset.repeat(epochs)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.params["batch_size"])
        return dataset

    def show_batch(self, image_batch):
        plt.figure(figsize=(10, 10))
        for n in range(25):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(image_batch[n] / 256.0)
            plt.axis("off")
        plt.show()

    def _save_training_result(self, logs):
        if logs is not None:
            self.params["final_training_loss"] = [loss.item() for loss in logs["loss"]]
            self.params["final_training_acc"] = [acc.item() for acc in logs["acc"]]

    def get_params(self):
        return self.params


# ---- COMPLEXITY OF PRIMITIVES ---- #
def conv2d_cx(cx, w_in, w_out, k, stride=1, groups=1):
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    flops += k * k * w_in * w_out * h * w // groups
    params += k * k * w_in * w_out // groups
    acts += w_out * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def batchnorm2d_cx(cx, w_in):
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    params += 2 * w_in
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}
