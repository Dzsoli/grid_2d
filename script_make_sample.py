from BPtools.trainer.bptrainer import BPTrainer
from BPtools.utils.models import *
from BPtools.metrics.criterions import KLD_BCE_loss_2Dvae
from modelv3 import *
import torch
from PIL import Image
from matplotlib import cm

enc = Encoder2D_v3()
dec = Decoder2D_v3()
vae = VAE2D(encoder=enc, decoder=dec).to("cuda")

model_param = torch.load('log_refined_model_v3/model_state_dict_refined_model_v3')
dm = GridVAE_DataModul(path='D:/dataset/grids/31349.0_11.npy', split_ratio=0.2)
dm.prepare_data()
dm.setup()

vae.load_state_dict(model_param)
data = dm.val_dataloader()[-1][:16]
print(data.shape)
kwargs, z = vae(data)
generated = boundary_for_grid(kwargs["output"]).squeeze(1).to("cpu").detach().numpy()
original = boundary_for_grid(data).squeeze(1).to("cpu").detach().numpy()

z = z.squeeze(1).to("cpu").detach().numpy()
print(z.shape)
# i=1
# for gen in generated:
#     # print(gen.shape)
#     # print(np.uint8(gen * 255))
#     im = Image.fromarray(np.uint8(gen * 255))
#     im.save(str(i) + ".png")
#     i+=1
#
# i=1
# for orig in original:
#     # print(gen.shape)
#     # print(np.uint8(gen * 255))
#     im = Image.fromarray(np.uint8(orig * 255))
#     im.save("real" + str(i) + ".png")
#     i+=1
#
i=1
for zz in z:
    # print(zz)
    # print(np.min(zz))
    zz = zz - np.min(zz)
    zz = zz / np.max(zz)
    # print(zz)
    # break
    im = Image.fromarray(np.uint8(zz * 255))
    im.save("latent" + str(i) + ".png")
    i+=1
