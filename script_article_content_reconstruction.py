from BPtools.trainer.bptrainer import BPTrainer
from BPtools.utils.models import *
from BPtools.metrics.criterions import KLD_BCE_loss_2Dvae
from modelv3 import *

# import bpoolsból, ha kell
from torchvision.utils import make_grid, save_image
from BPtools.utils.trajectory_plot import boundary_for_grid
from matplotlib import pyplot as plt
from PIL import Image, ImageOps, ImageFont, ImageDraw
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

bce = nn.BCELoss()

model_params = {}
for k,v in model_path.items():
    model_params[k] = torch.load(v)


# Beolvasni az adatokat
# Validation stephez hasonlóan kiválasztnai 16 képet
dm = GridVAE_DataModul(path='D:/dataset/grids/31349.0_11.npy', split_ratio=0.2)
dm.prepare_data()
dm.setup()
batch = dm.val_dataloader()[-1][:16]



# vae_gauss = torch.load(model_path["vae_gauss"])
# aae_gauss = torch.load(model_path["aae_gauss"])
# encoder_dict = {k.split("encoder.")[1]: v for k, v in aae_gauss.items() if "encoder" in k}
# decoder_dict = {k.split("decoder.")[1]: v for k, v in aae_gauss.items() if "decoder" in k}

# print(encoder_dict)
# enc.load_state_dict(encoder_dict)
# dec.load_state_dict(decoder_dict)
# print(aae_gauss)


# legyártani a 2*8-ból a valódi, 3 aae, 2 vae képeket
bound = boundary_for_grid(batch)
table = torch.zeros_like(bound).unsqueeze(0)
table[0] = bound
# kiszámolni a loss értékeket a rekonstrukcióra
for key, model in model_dict.items():
    # enc_dict = model.encoder.state_dict()
    # dec_dict = model.decoder.state_dict()
    #
    # enc_param = {k.split("encoder.")[1]: v for k, v in model_params[key].items() if "encoder" in key}
    # dec_param = {k.split("decoder.")[1]: v for k, v in model_params[key].items() if "decoder" in key}
    #
    # enc_dict.update(enc_param)
    # dec_dict.update(dec_param)
    # model.encoder.load_state_dict(enc_dict)
    # model.decoder.load_state_dict(dec_dict)
    model.load_state_dict(model_params[key])

    model.to("cuda")
    # kwargs, z = model(batch)
    # table = torch.cat((table, boundary_for_grid(kwargs["output"]).unsqueeze(0)), dim=0)
    # print(key + "    " + str(bce(kwargs["output"], batch).item()))
    loss_val = 0
    loss_tr = 0
    loss = nn.BCELoss()
    for b in dm.val_dataloader():
        kw, z = model(b)
        l = loss(kw["output"], b)
        loss_val = loss_val + l.item()

    for b in dm.train_dataloader():
        kw, z = model(b)
        l = loss(kw["output"], b)
        loss_tr = loss_tr + l.item()

    print("valid  "+key+"     "+str(loss_val/len(dm.val_dataloader())))
    print("train  "+key+"     "+str(loss_tr/len(dm.train_dataloader())))

    # img_generated = make_grid(boundary_for_grid(kwargs["output"]), normalize=True, nrow=2)
    # save_image(img_generated, fp=key + "_reconstruction.pdf")
    # img_latent_dist_grid = make_grid(boundary_for_grid(z), normalize=True, nrow=2)
    # save_image(img_latent_dist_grid, fp=key + "_latentdist.pdf")
"""
print(table.shape)
# table = table.view((16,6,1,18,130))
table = table.transpose(dim0=0, dim1=1)
print(table.shape)
table = table.reshape((16*5,1,18,130))
img_table = make_grid(table, nrow=5, normalize=True)

conv = Grayscale(num_output_channels=1)
img_table = conv(img_table)
save_image(img_table, "table.pdf")


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
np_table.save("np_table.pdf")

# interpolációs képet is legyártani



###
# legjobb valid loss, epoch train loss

# legjobb train loss, epoch
"""