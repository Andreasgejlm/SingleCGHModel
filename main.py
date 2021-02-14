
from SLM.SLM import SLM
from time import time
import numpy as np
import matplotlib.pyplot as plt
from utils import compare_to_gs, _get_input_fn, show_batch
from CGHSystem.CGHSystem import CGHSystem
from NetworkConfig.NetworkConfig import NetworkConfig
from CGHModel.CGHModel import CGHNet
from DataProvider.CGHDataProvider import CGHDataProvider
import tensorflow as tf


slm = SLM(resolution=(1024, 1024), pixel_pitch=0.000015)
system = CGHSystem(dz=0.002, num_z_planes=1, wavelength=1000)
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.45,
    staircase=True)
network_config = NetworkConfig(n_training=32768, batchsize=32, lr=lr_schedule, IF=8, n_validation=64, n_kern_unet_init=128, epochs=30, training_types=("cylinder", "line", "line", "polygon"))

data_provider = CGHDataProvider(slm=slm, system=system, config=network_config)

#train_path, val_path = data_provider.get_training_data()

#train_images = _get_input_fn(tf.io.gfile.glob(train_path + "/file_*.tfrecords"), network_config.batchsize, 1, (512, 512, 3), 1)
#sample = next(iter(train_images))
#show_batch(sample)

model = CGHNet(data_provider)
#model.train_network()
#model.save_model()

gt1 = np.array(plt.imread("hc_1024.jpg"), dtype='float32')
gt2 = np.array(plt.imread("cirkel4.png"))
gt3 = np.array(plt.imread("cirkel0.png"))
g = np.expand_dims(gt1[:, :, 0], axis=[0,-1])
g -= g.min()
g /= g.max()
#g2 = np.expand_dims(np.pad(gt2[:, :, 0], 0), axis=[0])
#g3 = np.expand_dims(np.pad(gt3[:, :, 0], 0), axis=[0])
#g = np.stack([g, g, g], axis=-1)
g = np.repeat(g, 32, axis=0)
#print(g.shape)

slm_phase = model.get_hologram(g)

print(compare_to_gs(slm_phase, g, system, slm))

#t = time()
#model.predict(g)
#t1 = time()
#print('Inference time was {:.2f}ms'.format((t1-t)*1000))


