

class CGHSystem:
	def __init__(self, num_z_planes=3, dz=0.01, wavelength=532):
		self.nz = num_z_planes
		self.dz = dz
		self.wl = wavelength * 10 ** -9