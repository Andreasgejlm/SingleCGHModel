from SLM.SLM import SLM
from CGHSystem.CGHSystem import CGHSystem
from NetworkConfig.NetworkConfig import NetworkConfig
from CGHModel.CGHModel import CGHNet
from DeepCGH.DeepCGH import DeepCGH
from DataProvider.CGHDataProvider import CGHDataProvider

data_provider = CGHDataProvider("DeepCGH_2D_512_3z_1")

model = DeepCGH()
model.create_model(data_provider)
#model.load_model("DeepCGH_2D_512_3z")
model.train_network()
model.test()
model.save_model()

