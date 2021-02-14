import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt
from pathlib import Path
from utils import _get_input_fn
from datetime import datetime
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter import filedialog

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tb_path = "./logs/"


# tf.compat.v1.enable_eager_execution()

class CGHNet:
    def __init__(self, dataprovider):
        self.path = Path(os.getcwd())
        self.training_data = dataprovider.get_training_data()
        self._trained = False

        self.params = dataprovider.params  # if params else cfg.CGHNET.default
        self.lr = dataprovider.lr
        self.phase_factors = dataprovider.phase_factors
        self.params["input_shape"] = (self.params["Mx"], self.params["My"], self.params["nz"])
        self.model_name = "MODEL-Mx{}-My{}-nz{}-lp{}-nT{}-bs{}-eps{}".format(self.params["Mx"],
                                                                                     self.params["My"],
                                                                                     self.params["nz"],
                                                                                     self.params["lp"],
                                                                                     self.params["nT"],
                                                                                     self.params[
                                                                                         "batchsize"],
                                                                                     self.params["epochs"])
        self.vgg_layers = ["block1_conv2", "block2_conv2", "block3_conv2", "block4_conv2", "block5_conv2"]
        existing_path = self.path / "saved_models" / self.model_name
        if os.path.exists(existing_path):
            print("Such a model already exists.")
            new_model_answer = input("Would you like to create new model? [y/N] ")
            if new_model_answer == "y":
                print("Building new model")
                self.pretrained = False
                self.model = self.build_model()
            else:
                print("Loading existing model")
                root = Tk()
                root.update()
                model_path = filedialog.askdirectory(title="Select saved model folder", initialdir=str(self.path / "saved_models"))
                root.destroy()
                self.model = tf.keras.models.load_model(model_path, custom_objects={"_deinterleave": self._deinterleave,
                                                                                   "_prop_to_slm": self._prop_to_slm,
                                                                                   "_interleave": self._interleave,
                                                                                   "_loss_func": self._loss_func})
                self.pretrained = True
        else:
            self.pretrained = False
            self.model = self.build_model()

    # def _fix_compatibility(self):
    #     w_bs = [max(1, w / b) for w, b in zip(self.params["ws"], self.params["bs"])]
    #     gs = [int(min(g, v)) for g, v in zip(self.params["gs"], w_bs)]
    #     w_bs = [int(round(w_b / g) * g) for w_b, g in zip(w_bs, gs)]
    #     self.params["ws"] = [int(w_b * b) for w_b, b in zip(w_bs, self.params["bs"])]
    #     self.params["gs"] = gs

    def _prop_to_slm(self, inputs):
        # We need to propagate the input backwards to the SLM with ifft2
        real, imag = inputs
        field_z0 = tf.complex(tf.squeeze(real, axis=-1), 0.) * tf.exp(tf.complex(0., tf.squeeze(imag, axis=-1)))
        shift = tf.signal.ifftshift(field_z0, axes=[1, 2])
        #field_slm = tf.signal.ifft2d(shift)
        slm = tf.math.angle(tf.signal.ifft2d(shift))
        return tf.expand_dims(slm, axis=-1)

    def _prop_to_planes(self, slm_phase):
        # Then propagate to the z planes we have defined
        phi_slm = tf.complex(np.float32(0.), tf.squeeze(slm_phase, axis=-1))
        phi_slm = tf.math.exp(phi_slm)

        output_list = []
        for i, factor in enumerate(self.phase_factors):
            if i != len(self.phase_factors) // 2:
                H = tf.broadcast_to(tf.expand_dims(factor, axis=0), tf.shape(phi_slm))
                phi_slm *= tf.signal.fftshift(H, axes=[1, 2])
            fft = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(phi_slm, axes=[1, 2])), axes=[1, 2])
            I = tf.cast(tf.math.square(tf.math.abs(fft)), tf.float32)
            output_list.append(tf.squeeze(I))
        return tf.stack(output_list, axis=-1)

    def _vgg_layers(self, layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on imagenet data
        input_shape = [self.params["Mx"], self.params["My"], self.params["nz"]]
        input_shape[2] = 3 if input_shape[2] == 1 else input_shape[2]
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          input_shape=input_shape)
        vgg.trainable = False
        for layer in vgg.layers:
            layer.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    def _l2_loss(self, y_true, y_pred):
        return tf.reduce_sum(tf.math.pow(y_pred - y_true, 2))

    def _perceptual_loss(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=-1)
        y_true = tf.stack([y_true, y_true, y_true], axis=-1)
        y_pred = tf.squeeze(y_pred, axis=-1)
        y_pred = tf.stack([y_pred, y_pred, y_pred], axis=-1)
        vgg_loss = 0
        vgg_true = self.vgg_model(y_true)
        vgg_pred = self.vgg_model(y_true)
        for vgg_val_true, vgg_val_pred in zip(vgg_true, vgg_pred):
            shape = vgg_val_true.shape
            scale_factor = shape[1] * shape[2] * shape[3]
            vgg_loss += 0.025 * self._l2_loss(vgg_val_true, vgg_val_pred) / scale_factor
        return vgg_loss

    def _loss_func(self, y_true, y_pred):
        y_predict, diffraction_constant = tf.split(y_pred, [self.params["nz"], 1], axis=-1)
        y_predict = tf.multiply(diffraction_constant, self._prop_to_planes(y_predict))

        #y_predict = tf.multiply(diffraction_constant, self._prop_to_planes(y_pred))

        #y_predict = tf.math.divide(
        #    tf.subtract(
        #        y_predict,
        #        tf.reduce_min(y_predict)
        #   ),
        #    tf.subtract(
        #        tf.reduce_max(y_predict),
        #        tf.reduce_min(y_predict)
        #    )
        #)

        #ssim = tf.image.ssim(y_true, y_predict, 1)
        #ssim = tf.reduce_mean(tf.math.add(tf.math.divide(ssim, 2.0), 0.5))


        #num = tf.reduce_sum(y_predict * y_true, axis=[1, 2, 3])
        #denom = tf.sqrt(
        #    tf.reduce_sum(tf.pow(y_predict, 2), axis=[1, 2, 3]) * tf.reduce_sum(tf.pow(y_true, 2), axis=[1, 2, 3]))

        #sq_err = tf.reduce_mean((num + 1) / (denom + 1), axis=0)
        shape = y_true.shape
        scale_factor = shape[1] * shape[2] * shape[3]
        return self._l2_loss(y_true, y_predict) / scale_factor# + self._perceptual_loss(y_true, y_predict)

    # LAYERS
    def _cc_layer(self, n_feature_maps, input):
        x = layers.Conv2D(n_feature_maps, (3, 3), activation='relu', padding='same')(input)
        x = layers.Conv2D(n_feature_maps, (3, 3), activation='relu', padding='same')(x)
        return x

    def _cbn_layer(self, n_feature_maps, activation, input):
        if activation == 'tanh' or activation == 'relu':
            x = layers.Conv2D(n_feature_maps, (3, 3), activation=activation, padding='same')(input)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(n_feature_maps, (3, 3), activation=activation, padding='same')(x)
            x = layers.BatchNormalization()(x)
        else:
            x = layers.Conv2D(n_feature_maps, (3, 3), activation=None, padding='same')(input)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(n_feature_maps, (3, 3), activation=None, padding='same')(x)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.BatchNormalization()(x)
        return x

    def _interleave(self, input):
        return tf.nn.space_to_depth(input=input, block_size=self.params["IF"])

    def _deinterleave(self, input):
        return tf.nn.depth_to_space(input=input, block_size=self.params["IF"])

    def _target_field(self, init_num_features, input_layer):
        x1 = self._cbn_layer(init_num_features, 'LeakyReLu', input_layer)
        x = layers.MaxPooling2D((2, 2), padding='same')(x1)

        x2 = self._cbn_layer(init_num_features * 2, 'LeakyReLu', x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x2)

        x3 = self._cbn_layer(init_num_features * 4, 'LeakyReLu', x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x3)

        x4 = self._cbn_layer(init_num_features * 8, 'LeakyReLu', x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x4)

        x5 = self._cbn_layer(init_num_features * 16, 'relu', x)
        x = layers.UpSampling2D()(x5)

        concat4 = layers.concatenate([x4, x])
        x6 = self._cbn_layer(init_num_features * 8, 'relu', concat4)
        x = layers.UpSampling2D()(x6)

        concat3 = layers.concatenate([x3, x])
        x7 = self._cbn_layer(init_num_features * 4, 'relu', concat3)
        x = layers.UpSampling2D()(x7)

        concat2 = layers.concatenate([x2, x])
        x8 = self._cbn_layer(init_num_features * 2, 'relu', concat2)
        x = layers.UpSampling2D()(x8)

        concat1 = layers.concatenate([x1, x])
        x9 = self._cbn_layer(init_num_features, 'tanh', concat1)

        return x9

    def _branching(self, previous, before_unet):

        real_branch = self._cc_layer(self.params["unet-ker-init"], previous)
        real_branch = layers.concatenate([real_branch, before_unet])
        real_branch = layers.Conv2D(self.params["IF"] ** 2, (3, 3), activation='relu', padding='same')(real_branch)

        imag_branch = self._cc_layer(self.params["unet-ker-init"], previous)
        imag_branch = layers.concatenate([imag_branch, before_unet])
        imag_branch = layers.Conv2D(self.params["IF"] ** 2, (3, 3), activation=None, padding='same')(imag_branch)

        de_int_real = layers.Lambda(self._deinterleave, name="De-interleave_real")(real_branch)
        de_int_imag = layers.Lambda(self._deinterleave, name="De-interleave_imag")(imag_branch)

        slm_field = layers.Lambda(self._prop_to_slm, name="SLM_phase")([de_int_real, de_int_imag])

        return slm_field

    def build_model(self):
        inp = Input(shape=(self.params["Mx"],
                           self.params["My"],
                           self.params["nz"]),
                    name='Input',
                    batch_size=self.params["batchsize"])
        interleaved = layers.Lambda(self._interleave, name="interleave")(inp)
        target_field = self._target_field(self.params["unet-ker-init"], interleaved)
        slm_phase = self._branching(target_field, interleaved)
        loss = PerceptualLoss(self.vgg_layers, self.params, self.phase_factors)(inp, slm_phase)

        model = Model(inp, outputs=loss)
        return model

    def train_network(self):
        train_dir, val_dir = self.training_data
        train_files = tf.io.gfile.glob(train_dir + "/file_*.tfrecords")
        val_files = tf.io.gfile.glob(val_dir + "/file_*.tfrecords")

        if self.pretrained:
            answer = input("Are you sure you want to train this model again? [y/n]")
            if answer != 'y':
                return

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0005, patience=3, mode="min")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
        )

        train_input_fn = _get_input_fn(filenames=train_files, epochs=self.params["epochs"], batchsize=self.params["batchsize"], shape=self.params["input_shape"])
        eval_input_fn = _get_input_fn(filenames=val_files, epochs=self.params["epochs"], batchsize=self.params["batchsize"], shape=self.params["input_shape"])
        training_history = self.model.fit(
            train_input_fn,
            None,
            epochs=self.params["epochs"],
            validation_data=eval_input_fn,
            steps_per_epoch=self.params["nT"] // self.params["batchsize"],
            callbacks=[early_stop]
        )
        return self.params

    def save_model(self):
        existing_path = self.path / "saved_models" / self.model_name
        if os.path.exists(existing_path):
            now = datetime.now()
            model_name = self.model_name + now.strftime("%m/%d/%Y-%H:%M:%S")
        else:
            model_name = self.model_name
        model_save_path = self.path / "saved_models" / model_name
        self.model.save(model_save_path)
        print("Model was saved")

    def get_hologram(self, target):
        print("Generating hologram")
        return self.model(target)




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


class PerceptualLoss(layers.Layer):
    def __init__(self, layer_names, params, phase_factors):
        super(PerceptualLoss, self).__init__()
        self.diffraction_constant = K.variable(1)  # or tf.Variable(var1) etc.
        self.vgg_model = self._vgg_layers(layer_names, params)
        self.params = params
        self.phase_factors = phase_factors

    def _prop_to_planes(self, slm_phase):
        # Then propagate to the z planes we have defined
        phi_slm = tf.complex(np.float32(0.), tf.squeeze(slm_phase, axis=-1))
        phi_slm = tf.math.exp(phi_slm)

        output_list = []
        for i, factor in enumerate(self.phase_factors):
            if i != len(self.phase_factors) // 2:
                H = tf.broadcast_to(tf.expand_dims(factor, axis=0), tf.shape(phi_slm))
                phi_slm *= tf.signal.fftshift(H, axes=[1, 2])
            fft = tf.signal.ifftshift(tf.signal.fft2d(tf.signal.fftshift(phi_slm, axes=[1, 2])), axes=[1, 2])
            I = tf.cast(tf.math.square(tf.math.abs(fft)), tf.float32)
            output_list.append(tf.squeeze(I))
        return tf.stack(output_list, axis=-1)

    def _vgg_layers(self, layer_names, params):
        """ Creates a vgg model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on imagenet data
        input_shape = [params["Mx"], params["My"], params["nz"]]
        input_shape[2] = 3 if input_shape[2] == 1 else input_shape[2]
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          input_shape=input_shape)
        vgg.trainable = False
        for layer in vgg.layers:
            layer.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    def _l2_loss(self, y_true, y_pred):
        return tf.reduce_sum(tf.math.pow(y_pred - y_true, 2))

    def _perceptual_loss(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=-1)
        #y_true = tf.multiply(y_true, 255.0)
        y_true = tf.stack([y_true, y_true, y_true], axis=-1)
        y_true = preprocess_input(y_true)
        y_pred = tf.squeeze(y_pred, axis=-1)
        #y_pred = tf.multiply(y_pred, 255.0)
        y_pred = tf.stack([y_pred, y_pred, y_pred], axis=-1)
        y_pred = preprocess_input(y_pred)
        vgg_loss = 0
        vgg_true = self.vgg_model(y_true)
        vgg_pred = self.vgg_model(y_pred)
        for vgg_val_true, vgg_val_pred in zip(vgg_true, vgg_pred):
            shape = vgg_val_true.shape
            scale_factor = shape[1] * shape[2] * shape[3]
            vgg_loss += 0.025 * self._l2_loss(vgg_val_true, vgg_val_pred) / scale_factor
        return vgg_loss

    def get_vars(self):
        return self.diffraction_constant

    def loss(self, y_true, y_pred):
        y_predict = tf.divide(self._prop_to_planes(y_pred), self.diffraction_constant)
        shape = y_true.shape
        scale_factor = shape[1] * shape[2] * shape[3]
        return self._l2_loss(y_true, y_predict) / scale_factor + self._perceptual_loss(y_true, y_predict)

    def call(self, y_true, y_pred):
        self.add_loss(self.loss(y_true, y_pred))
        return y_pred
