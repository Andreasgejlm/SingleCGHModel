import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from pathlib import Path
from utils import _get_input_fn, mean_sq_errs, normalize_images, show_batch
from datetime import datetime
import yaml
import math
from tqdm import tqdm
import scipy.io
import PIL
from tkinter import Tk
from tkinter import filedialog

class DeepCGH:
    def __init__(self):
        self.path = Path(os.getcwd())
        self._trained = False
        self.model = None
        self.input_shape = None
        self.test_data_path = None

    def _load_config_file(self, model_name):
        with open(self.path / "pretrained_model_configs.yaml") as config_file:
            config = yaml.safe_load(config_file)
        assert config[model_name] is not None, "No model with the given name present in config file."
        model_config = config[model_name]

        # Network specific
        self.model_name = model_name
        self.input_shape = model_config["network"]["shape"]
        self.nT = model_config["network"]["num_training_images"]
        self.nV = model_config["network"]["num_val_images"]
        self.nK = model_config["network"]["unet_n_kernels"]
        self.epochs = model_config["network"]["epochs"]
        self.batchsize = model_config["network"]["batchsize"]
        self.IF = model_config["network"]["IF"]
        self.lr = model_config["network"]["lr"]
        self.dataset_types = model_config["network"]["training_types"]

        # SLM Specific
        self.lp = model_config["slm"]["pixel_pitch"]
        self.Mx, self.My, self.nz = self.input_shape
        self.l0x = self.Mx * self.lp
        self.l0y = self.My * self.lp

        # Optical system Specific
        self.wl = model_config["system"]["wl"]
        self.dz = model_config["system"]["dz"]

        self.test_data_path = (self.path / "dataprovider" / "test_data" / model_config["test_data_folder"] / model_config["test_data_file"]).with_suffix(
            '.tfrecords')

    def load_model(self, model_name):
        existing_path = self.path / "saved_models" / model_name
        assert os.path.exists(existing_path), "No model with the given name present in saved_models folder."
        self._load_config_file(model_name)
        self.model = tf.keras.models.load_model(existing_path, custom_objects={"_deinterleave": self._deinterleave,
                                                                                "_prop_to_slm": self._prop_to_slm,
                                                                                "_interleave": self._interleave,
                                                                                "_loss_func": self._loss_func})
        self.phase_factors = self._calculate_phase_factors()
        print(f"Model {model_name} loaded into memory.")

    def create_model(self, dataprovider):
        self.training_data = dataprovider.get_training_data()
        self.model_name = dataprovider.model_name
        self._load_config_file(self.model_name)

        self.phase_factors = self._calculate_phase_factors()
        existing_path = self.path / "saved_models" / self.model_name
        if os.path.exists(existing_path):
            print("Such a model already exists.")
            new_model_answer = input("Would you like to create new model? [y/N] ")
            if new_model_answer == "y":
                print("Building new model")
                self.model = self.build_model()
            else:
                print("Not building new model. Returning...")
                return
        else:
            self.model = self.build_model()

    def _calculate_phase_factors(self):
        x, y = np.meshgrid(np.linspace(-self.My // 2 + 1, self.My // 2, self.My),
                           np.linspace(-self.Mx // 2 + 1, self.Mx // 2, self.Mx))
        Fx = x / self.lp / self.Mx
        Fy = y / self.lp / self.My

        center = self.nz // 2
        phase_factors = []

        for n in range(self.nz):
            zn = n - center
            p = np.exp(-1j * math.pi * float(self.wl) * (float(zn) * float(self.dz)) * (Fx ** 2 + Fy ** 2))
            phase_factors.append(p.astype(np.complex64))
        return phase_factors

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


    def _loss_func(self, y_true, slm_phase):
        y_predict = self._prop_to_planes(slm_phase)
        acc = self._acc(y_true, y_predict)
        return 1 - tf.reduce_mean(acc, axis=0)

    def _acc(self, y_true, y_pred):
        num = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
        denom = tf.sqrt(
            tf.reduce_sum(tf.pow(y_pred, 2), axis=[1, 2, 3]) * tf.reduce_sum(tf.pow(y_true, 2), axis=[1, 2, 3]))
        sq_err = (num + 1) / (denom + 1)
        return sq_err

    # LAYERS
    def _cc_layer(self, n_feature_maps, input):
        x = layers.Conv2D(n_feature_maps, (3, 3), activation='relu', padding='same')(input)
        x = layers.Conv2D(n_feature_maps, (3, 3), activation='relu', padding='same')(x)
        return x

    def _cbn_layer(self, n_feature_maps, input):
        x = layers.Conv2D(n_feature_maps, (3, 3), activation='relu', padding='same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(n_feature_maps, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        return x

    def _interleave(self, input):
        return tf.nn.space_to_depth(input=input, block_size=self.IF)

    def _deinterleave(self, input):
        return tf.nn.depth_to_space(input=input, block_size=self.IF)

    def _target_field(self, init_num_features, input_layer):
        x1 = self._cbn_layer(init_num_features, input_layer)
        x = layers.MaxPooling2D((2, 2), padding='same')(x1)

        x2 = self._cbn_layer(init_num_features * 2, x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x2)

        x3 = self._cc_layer(init_num_features * 4, x)

        x = layers.UpSampling2D()(x3)
        concat2 = layers.concatenate([x2, x])

        x4 = self._cc_layer(init_num_features * 2, concat2)

        x = layers.UpSampling2D()(x4)
        concat1 = layers.concatenate([x1, x])

        x5 = self._cc_layer(init_num_features, concat1)

        return x5

    def _branching(self, previous, before_unet):

        splitter = self._cc_layer(self.nK, previous)
        splitter = layers.concatenate([splitter, before_unet])

        real_branch = layers.Conv2D(self.IF ** 2, (3, 3), activation='relu', padding='same')(splitter)
        imag_branch = layers.Conv2D(self.IF ** 2, (3, 3), activation=None, padding='same')(splitter)

        de_int_real = layers.Lambda(self._deinterleave, name="De-interleave_real")(real_branch)
        de_int_imag = layers.Lambda(self._deinterleave, name="De-interleave_imag")(imag_branch)

        slm_field = layers.Lambda(self._prop_to_slm, name="SLM_phase")([de_int_real, de_int_imag])

        return slm_field

    def build_model(self):
        inp = Input(shape=self.input_shape,
                    name='Input',
                    batch_size=self.batchsize)
        interleaved = layers.Lambda(self._interleave, name="interleave")(inp)
        target_field = self._target_field(self.nK, interleaved)
        slm_phase = self._branching(target_field, interleaved)

        model = Model(inp, slm_phase)
        return model

    def train_network(self):
        train_dir, val_dir = self.training_data
        train_files = tf.io.gfile.glob(train_dir + "/file_*.tfrecords")
        val_files = tf.io.gfile.glob(val_dir + "/file_*.tfrecords")

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0005, patience=3, mode="min")

        self.model.compile(
            loss=self._loss_func,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
        )

        train_input_fn = _get_input_fn(filenames=train_files, epochs=self.epochs, batchsize=self.batchsize, shape=self.input_shape)
        eval_input_fn = _get_input_fn(filenames=val_files, epochs=self.epochs, batchsize=self.batchsize, shape=self.input_shape)
        training_history = self.model.fit(
            train_input_fn,
            epochs=self.epochs,
            validation_data=eval_input_fn,
            steps_per_epoch=self.nT // self.batchsize,
            callbacks=[early_stop]
        )

    def save_model(self, model_name=None):
        model_name = model_name if model_name is not None else self.model_name
        existing_path = self.path / "saved_models" / model_name
        if os.path.exists(existing_path):
            now = datetime.now()
            model_name = model_name + now.strftime("%m/%d/%Y-%H:%M:%S")
        else:
            model_name = model_name
        model_save_path = self.path / "saved_models" / model_name
        self.model.save(model_save_path)
        print("Model was saved")

    def get_hologram(self, target):
        print("Generating hologram")
        return self.model(target)

    def test(self):
        # Import test images
        test_input_function = _get_input_fn(filenames=str(self.test_data_path), epochs=1, batchsize=32, shuffle=1, shape=self.input_shape)
        test_images = next(iter(test_input_function))
        test_images = test_images[0].numpy()
        test_images = normalize_images(test_images)

        # Results of Network
        holograms = self.get_hologram(test_images)
        multi_planes = self._prop_to_planes(holograms).numpy()
        multi_planes = normalize_images(multi_planes)
        mse_cnn = mean_sq_errs(multi_planes, test_images)
        ssim_cnn = tf.image.ssim(tf.cast(multi_planes, tf.float32), tf.cast(test_images, tf.float32), max_val=1)
        acc = self._acc(test_images, multi_planes)

        # Save results
        result_save = tqdm(range(test_images.shape[0]))
        result_save.set_description("Saving test results...")
        test_result_dir = self.path / "test_results" / self.model_name
        if not os.path.isdir(test_result_dir):
            os.mkdir(test_result_dir)
        for image_index in result_save:
            path = self.path / "test_results" / self.model_name / f"image_{image_index}.mat"
            test = {f"test_z{image_plane}": test_images[image_index, :, :, image_plane] for image_plane in range(test_images.shape[-1])}
            predict = { f"pred_z{image_plane}": multi_planes[image_index, :, :, image_plane] for image_plane in range(test_images.shape[-1])}
            test_result = {"test": test,
                           "predict": predict,
                           "mse": mse_cnn[image_index],
                           "ssim": ssim_cnn.numpy()[image_index],
                           "acc": acc.numpy()[image_index]}
            scipy.io.savemat(path, test_result)
        scipy.io.savemat(test_result_dir / "errors.mat", {"mse": mse_cnn, "ssim": ssim_cnn.numpy(), "acc": acc.numpy()})


# ---- COMPLEXITY OF PRIMITIVES ---- #
def conv2d_cx(cx, w_in, w_out, k, stride=1, groups=1):
    h, w, flops, dataprovider, acts = cx["h"], cx["w"], cx["flops"], cx["dataprovider"], cx["acts"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    flops += k * k * w_in * w_out * h * w // groups
    dataprovider += k * k * w_in * w_out // groups
    acts += w_out * h * w
    return {"h": h, "w": w, "flops": flops, "dataprovider": dataprovider, "acts": acts}


def batchnorm2d_cx(cx, w_in):
    h, w, flops, dataprovider, acts = cx["h"], cx["w"], cx["flops"], cx["dataprovider"], cx["acts"]
    dataprovider += 2 * w_in
    return {"h": h, "w": w, "flops": flops, "dataprovider": dataprovider, "acts": acts}

