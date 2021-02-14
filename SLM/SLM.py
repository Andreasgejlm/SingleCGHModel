
class SLM:
	def __init__(self, resolution=(512, 512), pixel_pitch=10**-6, wavelength=532):
		self.Mx = resolution[0]
		self.My = resolution[1]
		self.lp = pixel_pitch
		self.l0x = self.Mx * self.lp
		self.l0y = self.My * self.lp
		self.wl = wavelength
