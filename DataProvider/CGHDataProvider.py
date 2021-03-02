import os
import numpy as np
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
import tensorflow as tf
import math
import yaml
from pathlib import Path
from utils import line_plane_intersection, show_batch_from_filename
np.seterr(all = "raise")
import matplotlib.pyplot as plt



class CGHDataProvider:
    def __init__(self, model_name):
        self.path = Path(os.getcwd())
        self.training_data_path = self.path / "DataProvider" / "training_data"
        self.validation_data_path = self.path / "DataProvider" / "validation_data"
        self.n_images_per_file = 512
        self.writer_options = tf.io.TFRecordOptions(compression_type='GZIP')
        self._load_config_file(model_name)

        self._generate_training_data()
        self._generate_test_data()

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

    # TRAINING, VALIDATION AND TEST DATA

    def _create_line(self, shape=None):
        image_shape = shape if shape is not None else (self.Mx, self.My, self.nz)
        image = np.zeros(image_shape)
        while image.max() == 0:
            for plane in range(image_shape[2]):
                im = Image.fromarray(image[:, :, plane])
                draw_im = ImageDraw.Draw(im)
                num_lines = random.randint(5, 10)
                for n_line in range(num_lines):
                    start_x = random.randint(10, image_shape[0] - 10)
                    start_y = random.randint(10, image_shape[1] - 10)
                    stop_x = random.randint(10, image_shape[0] - 10)
                    stop_y = random.randint(10, image_shape[1] - 10)
                    draw_im.line((start_x, start_y, stop_x, stop_y),
                                 fill=1,
                                 width=1)
                image[:, :, plane] = np.array(im, dtype='float32')
                if plane != 0 and image[:, :, 0].max() != 0 and image[:, :, plane].max() != 0:
                    image[:, :, plane] *= np.sqrt(np.sum(image[:, :, 0] ** 2) / np.sum(image[:, :, plane] ** 2))
        image -= image.min()
        image /= image.max()

        return image

    def _create_circle(self, shape=None):
        image_shape = shape if shape is not None else (self.Mx, self.My, self.nz)
        image = np.zeros(image_shape)
        while image.max() == 0:
            for plane in range(image_shape[2]):
                im = Image.fromarray(image[:, :, plane])
                draw_im = ImageDraw.Draw(im)
                num_circles = random.randint(5, 10)
                for n_circle in range(num_circles):
                    diameter = random.randint(3, int(min(image_shape[0], image_shape[1]) - 2))
                    x_0 = random.randint(10, image_shape[0] - 10)
                    y_0 = random.randint(10, image_shape[1] - 10)
                    x_1 = x_0 + diameter
                    y_1 = y_0 + diameter
                    draw_im.ellipse([(x_0, y_0), (x_1, y_1)],
                                    outline=1,
                                    fill=1)
                image[:, :, plane] = np.array(im, dtype='float32')
                if plane != 0 and image[:, :, 0].max() != 0 and image[:, :, plane].max() != 0:
                    image[:, :, plane] *= np.sqrt(np.sum(image[:, :, 0] ** 2) / np.sum(image[:, :, plane] ** 2))
        image -= image.min()
        image /= image.max()
        return image

    def _create_sphere(self, shape=None):
        image_shape = shape if shape is not None else (self.Mx, self.My, self.nz)
        if image_shape[-1] == 1:
            return self._create_circle(image_shape)
        image = np.zeros(image_shape)
        min_R = 1.1 * self.dz
        max_R = min(self.l0x, self.l0y) // 6
        n_spheres = random.randint(2, 8)
        center = image_shape[2] // 2
        nnz = 64
        zs = [(zn - center) * self.dz for zn in range(image_shape[2])]
        z_samples_axis = np.linspace(zs[0] - self.dz, zs[-1] + self.dz, nnz)
        while image.max() == 0:
            for sphere in range(n_spheres):
                centerx = random.choice(range(image_shape[0]))
                centery = random.choice(range(image_shape[1]))
                centerz = random.choice(z_samples_axis)
                sphere_radius = np.random.uniform(min_R, max_R)
                for z in range(image_shape[2]):
                    im = Image.fromarray(image[:, :, z])
                    draw_im = ImageDraw.Draw(im)
                    delta_z = np.abs(centerz - zs[z])
                    if sphere_radius > delta_z:
                        cross_section_radius = np.sqrt(sphere_radius ** 2 - delta_z ** 2) // self.lp
                        draw_im.ellipse([(centerx - cross_section_radius, centery - cross_section_radius),
                                        (centerx + cross_section_radius, centery + cross_section_radius)],
                                        fill=1)
                        image[:, :, z] = np.array(im, dtype='float32')
                    if z != 0 and image[:, :, 0].max() != 0 and image[:, :, z].max() != 0:
                        image[:, :, z] *= np.sqrt(np.sum(image[:, :, 0] ** 2) / np.sum(image[:, :, z] ** 2))

        image -= image.min()
        image /= image.max()
        return image

    def _create_cylinder(self, shape=None):
        image_shape = shape if shape is not None else (self.Mx, self.My, self.nz)
        image = np.zeros(image_shape)
        plane_vector = [0, 0, 1]
        num_cyls = random.randint(3, 8)
        max_R = min(image_shape[0], image_shape[1]) / 2
        center = image_shape[2] // 2
        zs = [(zn - center) * self.dz for zn in range(image_shape[2])]
        startz = zs[0]
        endz = zs[-1]
        while image.max() == 0:
            for cyl in range(num_cyls):
                startx = random.choice(range(image_shape[0]))
                starty = random.choice(range(image_shape[1]))

                endx = random.choice(range(image_shape[0]))
                endy = random.choice(range(image_shape[1]))

                line_vector = [endx - startx, endy - starty, endz - startz]
                norm_line_v = line_vector / np.linalg.norm(line_vector)
                line_point = [startx, starty, startz]
                radius = random.uniform(0.1, 1) * max_R
                radii = radius * norm_line_v
                for plane in range(image_shape[2]):
                    plane_point = [startx, starty, zs[plane]]
                    I = line_plane_intersection(line_vector, line_point, plane_vector, plane_point)
                    if I is not None:
                        im = Image.fromarray(image[:, :, plane])
                        draw_im = ImageDraw.Draw(im)
                        x_1 = I[0] + radii[0]
                        y_1 = I[1] + radii[1]
                        draw_im.ellipse([(I[0] - radii[0], I[1] - radii[1]), (x_1, y_1)], fill=1)
                        image[:, :, plane] = np.array(im, dtype='float32')
                    if plane != 0 and image[:, :, 0].max() != 0 and image[:, :, plane].max() != 0:
                        image[:, :, plane] *= np.sqrt(np.sum(image[:, :, 0] ** 2) / np.sum(image[:, :, plane] ** 2))
        image -= image.min()
        try:
            image /= image.max()
        except FloatingPointError:
            print("cylinder")
            print(image)
        return image

    def _create_polygon(self, shape=None):
        image_shape = shape if shape is not None else (self.Mx, self.My, self.nz)
        image = np.zeros(image_shape)
        while image.max() == 0:
            for plane in range(image_shape[2]):
                im = Image.fromarray(image[:, :, plane])
                draw_im = ImageDraw.Draw(im)
                num_polys = random.randint(5, 10)
                for n_poly in range(num_polys):
                    radius = random.randint(5, int(image_shape[0] - 2))
                    x_0 = random.randint(10, image_shape[0] - 10)
                    y_0 = random.randint(10, image_shape[1] - 10)
                    n_sides = random.randint(3, 6)
                    xs = [random.randint(x_0, x_0 + radius) for n in range(n_sides)]
                    ys = [random.randint(y_0, y_0 + radius) for n in range(n_sides)]
                    xy = [val for pair in zip(xs, ys) for val in pair]
                    draw_im.polygon(xy, outline=1, fill=1)
                image[:, :, plane] = np.array(im, dtype='float32')
                if plane != 0 and image[:, :, 0].max() != 0 and image[:, :, plane].max() != 0:
                    image[:, :, plane] *= np.sqrt(np.sum(image[:, :, 0] ** 2) / np.sum(image[:, :, plane] ** 2))
        image -= image.min()
        image /= image.max()

        return image

    def _generate_training_data(self):

        # Check whether training dataset exists already
        dir_name = "TRAIN-Mx{}-My{}-nz{}-nT{}".format(self.Mx,
                                                      self.My,
                                                      self.nz,
                                                      self.nT)
        if os.path.isdir(os.path.join(self.training_data_path, dir_name)):
            print("Chosen training data already exists. Continuing...")
        else:
            os.mkdir(self.training_data_path / dir_name)
            n_files = self.nT // self.n_images_per_file
            if n_files == 0:
                n_files = 1
                n_images = self.nT
            else:
                n_images = self.n_images_per_file

            for file_index in range(n_files):
                file_name = "file_{}.tfrecords".format(file_index)
                with tf.io.TFRecordWriter(str(self.training_data_path / dir_name / file_name), options=self.writer_options) as writer:
                    progress = tqdm(range(n_images))
                    progress.set_description("Writing training file {} of {} to tfrecords ...".format(file_index+1, n_files))
                    for i in progress:
                        rand_select = random.choice(self.dataset_types)
                        if rand_select == 'line':
                            training_image = self._create_line()
                        elif rand_select == 'circle':
                            training_image = self._create_circle()
                        elif rand_select == 'sphere':
                            training_image = self._create_sphere()
                        elif rand_select == 'cylinder':
                            training_image = self._create_cylinder()
                        else:
                            training_image = self._create_polygon()

                        image_bytes = training_image.tostring()

                        f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))

                        feature = {'image': f}

                        features = tf.train.Features(feature=feature)
                        example = tf.train.Example(features=features)
                        example_to_string = example.SerializeToString()

                        writer.write(example_to_string)
        self.train_file_path = str(self.training_data_path / dir_name)

        self._generate_validation_data()


    def _generate_validation_data(self):
        # Check whether validation dataset exists already
        dir_name = "VAL-Mx{}-My{}-nz{}-nV{}".format(self.Mx,
                                                    self.My,
                                                    self.nz,
                                                    self.nV,)
        if os.path.isdir(os.path.join(self.validation_data_path, dir_name)):
            print("Chosen validation data already exists. Continuing...")
        else:
            os.mkdir(self.validation_data_path / dir_name)
            n_files = self.nV // self.n_images_per_file
            if n_files == 0:
                n_files = 1
                n_images = self.nV
            else:
                n_images = self.n_images_per_file

            for file_index in range(n_files):
                file_name = "file_{}.tfrecords".format(file_index)
                with tf.io.TFRecordWriter(str(self.validation_data_path / dir_name / file_name), options=self.writer_options) as writer:
                    progress = tqdm(range(n_images))
                    progress.set_description("Writing validation file {} of {} to tfrecords ...".format(file_index + 1, n_files))
                    for i in progress:
                        rand_select = random.choice(self.dataset_types)
                        if rand_select == 'line':
                            val_image = self._create_line()
                        elif rand_select == 'circle':
                            val_image = self._create_circle()
                        elif rand_select == 'sphere':
                            val_image = self._create_sphere()
                        elif rand_select == 'cylinder':
                            val_image = self._create_cylinder()
                        else:
                            val_image = self._create_polygon()

                        image_bytes = val_image.tostring()

                        f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))

                        feature = {'image': f}

                        features = tf.train.Features(feature=feature)
                        example = tf.train.Example(features=features)
                        example_to_string = example.SerializeToString()

                        writer.write(example_to_string)
        self.val_file_path = str(self.validation_data_path / dir_name)

    def _generate_test_data(self):
        # Test data consists of a batch of 32 multiplane images, one batch for each num. of planes.
        # Images are a mix of 2D and 3D objects, and portraits.

        sizes = [(256, 256), (512, 512), (1024, 1024), (1920, 1080)]
        planes = [1, 3, 5, 7, 11]
        orig_faces_path = self.path / "test_batch_orig_faces"
        progress = tqdm(enumerate(sizes))
        progress.set_description(f"Writing test data")
        for sizeIndex, size in progress:
            for nPlanes in planes:
                plane_folder = self.path / "dataprovider" / "test_data" / f"size_{size[0]}x{size[1]}"
                if not os.path.isdir(plane_folder):
                    os.mkdir(plane_folder)
                filename = plane_folder / f"TESTBATCH_z_{nPlanes}.tfrecords"
                shape = (size[1], size[0], nPlanes)
                if not os.path.isfile(filename):
                    nPortraitImages = 8
                    n2DImages = 8
                    n3DImages = 16
                    with tf.io.TFRecordWriter(str(filename), options=self.writer_options) as writer:
                        portraits = [orig_faces_path / f for f in os.listdir(orig_faces_path) if os.path.isfile(orig_faces_path / f) and f"{size[0]}x{size[1]}" in f]
                        for i in range(nPortraitImages):
                            stackedPortrait = np.zeros(shape)
                            selectedPortraits = random.choices(portraits, k=nPlanes)
                            for index, portrait in enumerate(selectedPortraits):
                                portraitImage = np.array(plt.imread(portrait), dtype='float32')
                                portraitImage = np.expand_dims(portraitImage[:, :, 0], axis=[0])
                                portraitImage -= portraitImage.min()
                                portraitImage /= portraitImage.max()
                                stackedPortrait[:, :, index] = portraitImage
                            stackedPortraitString = stackedPortrait.tostring()
                            f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[stackedPortraitString]))
                            feature = {'image': f}
                            features = tf.train.Features(feature=feature)
                            example = tf.train.Example(features=features)
                            example_to_string = example.SerializeToString()
                            writer.write(example_to_string)

                        for i in range(n2DImages):
                            rand_select = random.choice(['line', 'circle', 'polygon'])
                            if rand_select == 'line':
                                image2D = self._create_line(shape)
                            elif rand_select == 'circle':
                                image2D = self._create_circle(shape)
                            else:
                                image2D = self._create_polygon(shape)
                            image_bytes = image2D.tostring()
                            f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                            feature = {'image': f}
                            features = tf.train.Features(feature=feature)
                            example = tf.train.Example(features=features)
                            example_to_string = example.SerializeToString()
                            writer.write(example_to_string)

                        for i in range(n3DImages):
                            rand_select = random.choice(['sphere', 'cylinder'])
                            if rand_select == 'sphere':
                                image3D = self._create_sphere(shape)
                            elif rand_select == 'cylinder':
                                image3D = self._create_cylinder(shape)
                            image_bytes = image3D.tostring()
                            f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                            feature = {'image': f}
                            features = tf.train.Features(feature=feature)
                            example = tf.train.Example(features=features)
                            example_to_string = example.SerializeToString()
                            writer.write(example_to_string)




    # FOURIER OPTICS SPECIFIC FUNCTIONS
    def _calculate_phase_factors(self):
        x, y = np.meshgrid(np.linspace(-self.My // 2 + 1, self.My // 2, self.My),
                           np.linspace(-self.Mx // 2 + 1, self.Mx // 2, self.Mx))
        Fx = x / self.lp / self.Mx
        Fy = y / self.lp / self.My

        center = self.nz // 2
        phase_factors = []

        for n in range(self.nz):
            zn = n - center
            p = np.exp(-1j * math.pi * self.wl * (zn * self.dz) * (Fx ** 2 + Fy ** 2))
            phase_factors.append(p.astype(np.complex64))
        self.phase_factors = phase_factors

    # TODO: Get training data https://keras.io/examples/keras_recipes/tfrecord/
    def get_training_data(self):
        return self.train_file_path, self.val_file_path
