import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def _read_tfrecord(path, shape):
    tfrecord_format = ({"image": tf.io.FixedLenFeature([], tf.string)})
    example = tf.io.parse_single_example(path, tfrecord_format)
    image = tf.cast(tf.reshape(tf.io.decode_raw(example['image'], tf.float64), shape),
                    tf.float32)
    return image, image

def _load_dataset(filenames, shape):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order)  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(map_func=lambda x: _read_tfrecord(x, shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def _get_input_fn(filenames, batchsize, epochs, shape, shuffle=1024):
    dataset = _load_dataset(filenames, shape)
    dataset = dataset.shuffle(shuffle)
    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batchsize)
    return dataset


def show_batch(image_batch):
    image_batch = image_batch[0].numpy()
    batchsize = len(image_batch)
    plt.figure(figsize=(15, 15))
    axs = [plt.subplot(int(np.sqrt(batchsize)), int(np.sqrt(batchsize)), i + 1) for i in range(batchsize)]
    for n in range(batchsize):
        axs[n].imshow(image_batch[n], cmap='gray')
    plt.show()


def show_phase_batch(image_batch):
    batchsize = len(image_batch)
    plt.figure(figsize=(10, 10))
    for n in range(25):
        if n < len(image_batch):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(np.angle(image_batch[n]) / 256.0)
            plt.axis("off")
    plt.show(block=False)


def line_plane_intersection(u, N, n, M):
    d = -np.dot(n, M)
    if np.dot(n, u) == 0:
        if np.dot(n, N) + d == 0:
            I = np.array(M)
        else:
            I = None
    else:
        t = - (d + np.dot(n, N)) / np.dot(n, u)
        I = np.array(N) + np.array(u) * t
    return I

def get_phase_factors(system, slm):
    zs = [-system.dz * x for x in np.arange(1, (system.nz - 1) // 2 + 1)][::-1] + [
        system.dz * x for x in np.arange(1, (system.nz - 1) // 2 + 1)]

    Hs = []
    for z in zs:
        x, y = np.meshgrid(np.linspace(-slm.My // 2 + 1, slm.My // 2, slm.My),
                           np.linspace(-slm.Mx // 2 + 1, slm.Mx // 2, slm.Mx))
        fx = x / slm.lp / slm.Mx
        fy = y / slm.lp / slm.My
        exp = np.exp(-1j * np.pi * system.wl * z * (fx ** 2 + fy ** 2))
        Hs.append(np.fft.fftshift(exp.astype(np.complex64)))
    Hs.insert(system.nz // 2, 0)
    return Hs

def propagate(phase, system, slm):
    phi_slm = np.exp(1j * np.squeeze(phase, axis=-1))

    output_list = []
    phase_factors = get_phase_factors(system, slm)
    for i, factor in enumerate(phase_factors):
        if i != len(phase_factors) // 2:
            H = np.broadcast_to(np.expand_dims(factor, axis=0), np.shape(phi_slm))
            phi_slm *= np.fft.fftshift(H, axes=[1, 2])
        fft = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(phi_slm, axes=[1, 2])), axes=[1, 2])
        I = np.square(np.abs(fft))
        output_list.append(np.squeeze(I))
    return np.expand_dims(np.stack(output_list, axis=-1), 0)

def compare_to_gs(obtained_phase, target, system, slm, gs_iter=20,):

    #TODO: Include all images in batch. For now only uses first image
    print(target.shape)
    target = np.expand_dims(target[0], axis=0)
    target_shape = target.shape
    obtained_phase = np.expand_dims(obtained_phase[0, :, :, :], axis=0)

    nn_images = propagate(obtained_phase, system, slm)
    nn_images = normalize_images(nn_images)
    assert nn_images.shape == target.shape, f"Size mismatch: {nn_images.shape} to {target.shape}"

    # GS Algo
    print(f"Running GS algorithm for {gs_iter} iterations ...")
    gs = GS3D(target_shape, system, slm)
    gs_errors = []
    for it in range(1, gs_iter):
        if it % 5 == 0:
            gs_phase = gs.get_phase(np.squeeze(target, axis=0), it)[np.newaxis, ..., np.newaxis].astype(np.float32)
            gs_images = propagate(gs_phase, system, slm)
            gs_images = normalize_images(gs_images)
            gs_errors.append(mean_sq_err(gs_images, target))

    # Results of Network
    mse_cnn = mean_sq_err(nn_images, target)
    ssim_cnn = tf.image.ssim(tf.cast(nn_images, tf.float32), tf.cast(target, tf.float32), max_val=1)

    # Results of GS
    ssim_gs = tf.image.ssim(tf.cast(gs_images, tf.float32), tf.cast(target, tf.float32), max_val=1)

    fig, axs = plt.subplots(2, 3, figsize=(9, 16))
    axs[0, 0].imshow(target[0, :, :, :], cmap='gray')
    axs[0, 1].imshow(gs_images[0, :, :, :], cmap='gray')
    axs[0, 2].imshow(nn_images[0, :, :, :], cmap='gray')
    axs[1, 1].imshow(gs_phase[0, :, :, 0], cmap='gray')
    axs[1, 2].imshow(obtained_phase[0, :, :, 0], cmap='gray')
    plt.show()

    return mse_cnn, ssim_cnn, gs_errors, ssim_gs


def L2(images, targets):
    loss = np.dot(np.abs(images - targets), np.abs(images - targets))
    return loss


def mean_sq_err(images, targets):
    error = ((images - targets) ** 2).mean(axis=None)
    return error


def normalize_images(images):
    x = images - images.min()
    x = images / images.max() if images.max() != 0 else 0
    return x


class GS3D(object):
    '''
    Class for the GS algorithm.
    Inputs:
        batch_size   int, determines the batch size of the prediction
        num_iter   int, determines the number of iterations of the GS algorithm
        input_shape   tuple of shape (height, width)
    Returns:
        Instance of the object
    '''
    def __init__(self, shape, system, slm):
        self.shape = shape
        self.plane_distance = system.dz
        self.wavelength = system.wl
        self.ps = slm.lp
        self.zs = [-self.plane_distance*x for x in np.arange(1, (self.shape[-1]-1)//2+1)][::-1] + [self.plane_distance*x for x in np.arange(1, (self.shape[-1]-1)//2+1)]
        self.Hs = self.__get_H(self.zs, self.shape, self.wavelength, self.ps)

    def __get_H(self, zs, shape, lambda_, ps):
        Hs = []
        for z in zs:
            x, y = np.meshgrid(np.linspace(-shape[1]//2+1, shape[1]//2, shape[1]),
                               np.linspace(-shape[0]//2+1, shape[0]//2, shape[0]))
            fx = x/ps/shape[0]
            fy = y/ps/shape[1]
            exp = np.exp(-1j * np.pi * lambda_ * z * (fx**2 + fy**2))
            Hs.append(np.fft.fftshift(exp.astype(np.complex64)))
        Hs.insert(shape[-1] // 2, 0)
        return Hs

    def __propagate(self, cf, H):
        return np.fft.ifft2(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(cf))*H))

    def __forward(self, cf_slm, Hs, As):
        new_Z = []
        z0 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(cf_slm)))
        for H, A in zip(Hs, As):
            if type(H)!=int:
                new_Z.append(A*np.exp(1j*np.angle(self.__propagate(z0, H))))
            else:
                new_Z.append(A*np.exp(1j*np.angle(z0)))
        return new_Z

    def __backward(self, Zs, Hs):
        slm_cfs = []
        for Z, H in zip(Zs, Hs[::-1]):
            if type(H)!=int:
                slm_cfs.append(np.fft.ifft2(np.fft.ifftshift(self.__propagate(Z, H))))
            else:
                slm_cfs.append(np.fft.ifft2(np.fft.ifftshift(Z)))
        cf_slm = np.exp(1j*np.angle(np.sum(np.array(slm_cfs), axis=0)))
        return cf_slm

    def get_phase(self, As, K):
        As = np.transpose(As, axes=(2, 0, 1))
        cf_slm = np.exp(1j * np.random.rand(*As.shape[1:]))
        for i in range(K):
            new_Zs = self.__forward(cf_slm, self.Hs, As)
            cf_slm = self.__backward(new_Zs, self.Hs)
        return np.angle(cf_slm)