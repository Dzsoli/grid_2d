from BPtools.trainer.bptrainer import BPTrainer
from BPtools.utils.models import *
from BPtools.metrics.criterions import KLD_BCE_loss_2Dvae
from modelv3 import *

# import bpoolsból, ha kell
from torchvision.utils import make_grid, save_image
from BPtools.utils.trajectory_plot import boundary_for_grid
from matplotlib import pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision.transforms import Grayscale


model_path = {"vae_gauss": "log_VAE_gauss_lam01_refined_model_v3/model_state_dict_VAE_gauss_lam01_refined_model_v3",
              "vae_bernoulli": "log_VAE_bernoulli_lam01_refined_model_v3_3run/model_state_dict_VAE_bernoulli_lam01_refined_model_v3_3run",
              "aae_gauss": "log_aae_gauss_lindisc_noanneal_refined_model_v3/model_state_dict_aae_gauss_lindisc_noanneal_refined_model_v3",
              "aae_bernoulli": "log_aae_bernoulli_lindisc_noanneal_refined_model_v3/model_state_dict_aae_bernoulli_lindisc_noanneal_refined_model_v3",
              "aae_uniform": "log_ADVAE_uniform_refined_model_v3_2/model_state_dict_ADVAE_uniform_refined_model_v3_2"
              }


# Beolvasni a modelleket
enc = Encoder2D_v3()
enc2 = Type2_Encoder2D_v3()
dec = Decoder2D_v3()
disc = Discriminator2D_Latent3_v3()


vae_gauss = VAE2D(encoder=enc, decoder=dec)
vae_bernoulli = VAE2D_Bernoulli(encoder=enc2, decoder=dec)
aae_uniform = ADVAE2D_Uniform(encoder=enc, decoder=dec, discriminator=disc)
aae_gauss = ADVAE2D_Gauss(encoder=enc, decoder=dec, discriminator=disc)
aae_bernoulli = ADVAE2D_Bernoulli(encoder=enc2, decoder=dec, discriminator=disc)

model_dict = {"aae_gauss": aae_gauss,
              "aae_bernoulli": aae_bernoulli,
              "vae_gauss": vae_gauss,
              "vae_bernoulli": vae_bernoulli,
              }

model_params = {}
for k,v in model_path.items():
    model_params[k] = torch.load(v)

dm = GridVAE_DataModul(path='D:/dataset/grids/31349.0_11.npy', split_ratio=0.2, shuffle=False)
dm.prepare_data()
dm.setup()
N=80
batch = dm.val_dataloader()[-1][N:N+16]

bound = boundary_for_grid(batch)
table = torch.zeros_like(bound).unsqueeze(0)
table[0] = bound

for key, model in model_dict.items():
    model.load_state_dict(model_params[key])
    model.to("cuda")
    kwargs, z = model(batch)
    interp = torch.zeros_like(z)
    # todo bernoulli
    for i in range(16):
        interp[i] = torch.lerp(z[0],z[-1],i/15)
    out = model.decoder(interp)
    table = torch.cat((table, boundary_for_grid(out).unsqueeze(0)), dim=0)
    # print(interp)
    # print(interp.shape)

table = table.transpose(dim0=0, dim1=1)
table = table.reshape((16*5,1,18,130))
img_table = make_grid(table, nrow=5, normalize=True)

conv = Grayscale(num_output_channels=1)
img_table = conv(img_table)
save_image(img_table, "interpol_table.pdf")


np_table = img_table.squeeze(0).cpu().detach().numpy()

print(np_table.shape)#, np_table.dtype, np_table)
np_table = Image.fromarray(np.uint8(np_table * 256), 'L')#.convert("LA")
np_table = ImageOps.invert(np_table)

np_table = ImageOps.expand(np_table,18,255)
draw = ImageDraw.Draw(np_table)
font = ImageFont.truetype("arial.ttf", 15, encoding="unic")
x=130
d=40
draw.text( (d,0), u"Original images", fill='Black', font=font)
draw.text( (d+x,0), u"AdvAE-Gauss", fill='Black', font=font)
draw.text( (d+2*x,0), u"AdvAE-Bernoulli", fill='Black', font=font)
draw.text( (d+3*x,0), u"VAE-Gauss", fill='Black', font=font)
draw.text( (d+4*x,0), u"VAE-Bernoulli", fill='Black', font=font)
np_table.save("np_interpol_table.pdf")