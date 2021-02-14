import os
import numpy as np
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
import tensorflow as tf
import math
from pathlib import Path
from utils import line_plane_intersection
np.seterr(all = "raise")
import matplotlib.pyplot as plt



class CGHDataProvider:
    def __init__(self, slm, system, config):
        self.path = Path(os.getcwd())
        self.training_data_path = self.path / "DataProvider" / "training_data"
        self.validation_data_path = self.path / "DataProvider" / "validation_data"
        self.n_images_per_file = 512
        self.writer_options = tf.io.TFRecordOptions(compression_type='GZIP')
        self.dataset_types = config.training_types
        self.lr = config.lr
        self.params = {
            "nK": config.n_kern_unet,
            "nT": config.nT,
            "nV": config.nV,
            "IF": config.IF,
            "Mx": slm.Mx,
            "My": slm.My,
            "lp": slm.lp,
            "l0x": slm.Mx * slm.lp,
            "l0y": slm.My * slm.lp,
            "nz": system.nz,
            "dz": system.dz,
            "wl": system.wl,
            "batchsize": config.batchsize,
            "epochs": config.epochs,
            "unet-ker-init": config.n_kern_unet
        }

        self.x_samples_fp = np.linspace(-self.params["Mx"]/2+1, self.params["Mx"]/2, self.params["Mx"]) / self.params["Mx"] * self.params["l0x"]
        self.y_samples_fp = np.linspace(-self.params["My"] / 2 + 1, self.params["My"] / 2, self.params["My"]) / self.params["My"] * self.params["l0y"]
        nnz = 64
        center = self.params["nz"] // 2
        self.zs = [(zn - center) * self.params["dz"] for zn in range(self.params["nz"])]
        self.z_samples_axis = np.linspace(self.zs[0] - self.params["dz"], self.zs[-1] + self.params["dz"], nnz)

        self._generate_training_data()
        self._calculate_phase_factors()

    # TRAINING AND VALIDATION DATA

    def _create_line(self):
        image = np.zeros((self.params["Mx"], self.params["My"], self.params["nz"]))
        while image.max() == 0:
            for plane in range(self.params["nz"]):
                im = Image.fromarray(image[:, :, plane])
                draw_im = ImageDraw.Draw(im)
                num_lines = random.randint(1, 5)
                for n_line in range(num_lines):
                    start_x = random.randint(10, self.params["Mx"] - 10)
                    start_y = random.randint(10, self.params["My"] - 10)
                    stop_x = random.randint(10, self.params["Mx"] - 10)
                    stop_y = random.randint(10, self.params["My"] - 10)
                    draw_im.line((start_x, start_y, stop_x, stop_y),
                                 fill=random.randint(20, 256),
                                 width=random.randint(2, 10))
                image[:, :, plane] = np.array(im, dtype='float32')
                if plane != 0:
                    image[:, :, plane] *= np.sqrt(np.sum(image[:, :, 0] ** 2) / np.sum(image[:, :, plane] ** 2))
        image -= image.min()
        image /= image.max()

        return image

    def _create_circle(self):
        image = np.zeros((self.params["Mx"], self.params["My"], self.params["nz"]))
        while image.max() == 0:
            for plane in range(self.params["nz"]):
                im = Image.fromarray(image[:, :, plane])
                draw_im = ImageDraw.Draw(im)
                num_circles = random.randint(1, 3)
                for n_circle in range(num_circles):
                    diameter = random.randint(3, int(self.params["Mx"] - 2))
                    x_0 = random.randint(self.params["Mx"] // 2 - 10, self.params["Mx"] // 2 + 10)
                    y_0 = random.randint(self.params["My"] // 2 - 10, self.params["My"] // 2 + 10)
                    x_1 = x_0 + diameter
                    y_1 = y_0 + diameter
                    draw_im.ellipse([(x_0, y_0), (x_1, y_1)],
                                    outline=random.randint(20, 256),
                                    fill=random.randint(20, 256),
                                    width=random.randint(1, 3))
                image[:, :, plane] = np.array(im, dtype='float32')
                if plane != 0:
                    image[:, :, plane] *= np.sqrt(np.sum(image[:, :, 0] ** 2) / np.sum(image[:, :, plane] ** 2))
        image -= image.min()
        image /= image.max()
        return image

    def _create_sphere(self):
        image = np.zeros((self.params["Mx"], self.params["My"], self.params["nz"]))
        min_R = 1.1 * self.params["dz"]
        max_R = min(self.params["l0x"], self.params["l0y"]) // 4
        n_spheres = random.randint(2, 5)
        while image.max() == 0:
            for sphere in range(n_spheres):
                centerx = random.choice(range(self.params["Mx"]))
                centery = random.choice(range(self.params["My"]))
                centerz = random.choice(self.z_samples_axis)
                sphere_radius = np.random.uniform(min_R, max_R)
                for z in range(self.params["nz"]):
                    im = Image.fromarray(image[:, :, z])
                    draw_im = ImageDraw.Draw(im)
                    delta_z = np.abs(centerz - self.zs[z])
                    if sphere_radius > delta_z:
                        cross_section_radius = np.sqrt(sphere_radius ** 2 - delta_z ** 2) // self.params["lp"]
                        draw_im.ellipse([(centerx - cross_section_radius, centery - cross_section_radius),
                                        (centerx + cross_section_radius, centery + cross_section_radius)],
                                        fill=1)
                        image[:, :, z] = np.array(im, dtype='float32')
                    if z != 0 and image[:, :, 0].max() != 0 and image[:, :, z].max() != 0:
                        image[:, :, z] *= np.sqrt(np.sum(image[:, :, 0] ** 2) / np.sum(image[:, :, z] ** 2))

        image -= image.min()
        image /= image.max()
        return image

    def _create_cylinder(self):
        image = np.zeros((self.params["Mx"], self.params["My"], self.params["nz"]))
        plane_vector = [0, 0, 1]
        num_cyls = random.randint(2, 5)
        max_R = min(self.params["Mx"], self.params["My"]) / 2
        startz = self.zs[0]
        endz = self.zs[-1]
        while image.max() == 0:
            for cyl in range(num_cyls):
                startx = random.choice(range(self.params["Mx"]))
                starty = random.choice(range(self.params["My"]))

                endx = random.choice(range(self.params["Mx"]))
                endy = random.choice(range(self.params["My"]))

                line_vector = [endx - startx, endy - starty, endz - startz]
                norm_line_v = line_vector / np.linalg.norm(line_vector)
                line_point = [startx, starty, startz]
                radius = random.uniform(0.1, 1) * max_R
                radii = radius * norm_line_v
                for plane in range(self.params["nz"]):
                    plane_point = [startx, starty, self.zs[plane]]
                    I = line_plane_intersection(line_vector, line_point, plane_vector, plane_point)
                    if I is not None:
                        im = Image.fromarray(image[:, :, plane])
                        draw_im = ImageDraw.Draw(im)
                        x_1 = I[0] + radii[0]
                        y_1 = I[1] + radii[1]
                        draw_im.ellipse([(I[0] - radii[0], I[1] - radii[1]), (x_1, y_1)], fill=1)
                        image[:, :, plane] = np.array(im, dtype='float32')
                    if plane != 0 and image[:, :, plane].max() != 0:
                        image[:, :, plane] *= np.sqrt(np.sum(image[:, :, 0] ** 2) / np.sum(image[:, :, plane] ** 2))
        image -= image.min()
        try:
            image /= image.max()
        except FloatingPointError:
            print("cylinder")
            print(image)
        return image

    def _create_polygon(self):
        image = np.zeros((self.params["Mx"], self.params["My"], self.params["nz"]))
        while image.max() == 0:
            for plane in range(self.params["nz"]):
                im = Image.fromarray(image[:, :, plane])
                draw_im = ImageDraw.Draw(im)
                num_polys = random.randint(1, 5)
                for n_poly in range(num_polys):
                    radius = random.randint(5, int(self.params["Mx"] - 2))
                    x_0 = random.randint(self.params["Mx"] // 2 - 10, self.params["Mx"] // 2 + 10)
                    y_0 = random.randint(self.params["My"] // 2 - 10, self.params["My"] // 2 + 10)
                    n_sides = random.randint(3, 6)
                    xs = [random.randint(x_0, x_0 + radius) for n in range(n_sides)]
                    ys = [random.randint(y_0, y_0 + radius) for n in range(n_sides)]
                    xy = [val for pair in zip(xs, ys) for val in pair]
                    draw_im.polygon(xy, outline=random.randint(20, 256), fill=random.randint(20, 256))
                image[:, :, plane] = np.array(im, dtype='float32')
                if plane != 0 and image[:, :, 0].max() != 0 and image[:, :, plane].max() != 0:
                    image[:, :, plane] *= np.sqrt(np.sum(image[:, :, 0] ** 2) / np.sum(image[:, :, plane] ** 2))
        image -= image.min()
        image /= image.max()

        return image

    def _generate_training_data(self):

        # Check whether training dataset exists already
        dir_name = "TRAIN-Mx{}-My{}-nz{}-nT{}".format(self.params["Mx"],
                                                                          self.params["My"],
                                                                          self.params["nz"],
                                                                          self.params["nT"])
        if os.path.isdir(os.path.join(self.training_data_path, dir_name)):
            print("Chosen training data already exists. Continuing...")
        else:
            os.mkdir(self.training_data_path / dir_name)
            n_files = self.params["nT"] // self.n_images_per_file
            if n_files == 0:
                n_files = 1
                n_images = self.params["nT"]
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
        dir_name = "VAL-Mx{}-My{}-nz{}-nV{}".format(self.params["Mx"],
                                                                          self.params["My"],
                                                                          self.params["nz"],
                                                                          self.params["nV"],)
        if os.path.isdir(os.path.join(self.validation_data_path, dir_name)):
            print("Chosen validation data already exists. Continuing...")
        else:
            os.mkdir(self.validation_data_path / dir_name)
            n_files = self.params["nV"] // self.n_images_per_file
            if n_files == 0:
                n_files = 1
                n_images = self.params["nV"]
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

    # FOURIER OPTICS SPECIFIC FUNCTIONS
    def _calculate_phase_factors(self):
        x, y = np.meshgrid(np.linspace(-self.params["My"] // 2 + 1, self.params["My"] // 2, self.params["My"]),
                           np.linspace(-self.params["Mx"] // 2 + 1, self.params["Mx"] // 2, self.params["Mx"]))
        Fx = x / self.params["lp"] / self.params["Mx"]
        Fy = y / self.params["lp"] / self.params["My"]

        center = self.params["nz"] // 2
        phase_factors = []

        for n in range(self.params["nz"]):
            zn = n - center
            p = np.exp(-1j * math.pi * self.params["wl"] * (zn * self.params["dz"]) * (Fx ** 2 + Fy ** 2))
            phase_factors.append(p.astype(np.complex64))
        self.phase_factors = phase_factors

    def get_params(self):
        return self.params

    # TODO: Get training data https://keras.io/examples/keras_recipes/tfrecord/
    def get_training_data(self):
        return self.train_file_path, self.val_file_path
