

class NetworkConfig:
	def __init__(self, n_training=1024, n_validation=256, lr=1E-3, IF=8, training_types=("circle", "line", "sphere", "polygon"), n_kern_unet_init=128, batchsize=32, epochs=10):
		self.nT = n_training
		self.nV = n_validation
		self.IF = IF
		self.lr = lr
		self.n_kern_unet = n_kern_unet_init
		assert all(i in ("line", "cylinder", "circle", "polygon", "sphere") for i in training_types), "Unknown training type."
		self.batchsize = batchsize
		self.epochs = epochs
		self.training_types = training_types
