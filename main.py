from BPtools.trainer.bptrainer import BPTrainer
from BPtools.utils.models import *
from BPtools.utils.datamoduls import *
from BPtools.metrics.criterions import KLD_BCE_loss_2Dvae
from modelv3 import *
from torchvision.models import resnet18

# enc = Type2_Encoder2D_v3()
enc = resnet18(pretrained=False, num_classes=64)
# enc = Encoder2D_v3()
# enc = Encoder2DHigher_v3()

# enc = Type2_Encoder2DLower_v3()
dec = Decoder2D_v3()
disc = Discriminator2D_Latent3_v3()
# vae = VAE2D_Gauss(encoder=enc, decoder=dec)
# aae = ADVAE2D_Bernoulli(encoder=enc, decoder=dec, discriminator=disc)
vae_bernoulli = VAE2D_Bernoulli(encoder=enc, decoder=dec)
aae_uniform = ADVAE2D_Uniform(encoder=enc, decoder=dec, discriminator=disc)

dm = GridVAE_DataModul(path='D:/dataset/grids/31349.0_11.npy', split_ratio=0.2)

trainer = BPTrainer(epochs=20000, criterion=KLD_BCE_loss_2Dvae_bernoulli(lam=0.1), name='VAE2D_Gauss_ResNet18')
trainer.fit(model=vae_bernoulli, datamodule=dm)
