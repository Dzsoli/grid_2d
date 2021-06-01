import torch
import torch.nn as nn
from BPtools.core.bpmodule import BPModule
import torch.optim as optim
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from torchvision.utils import make_grid
from BPtools.utils.trajectory_plot import boundary_for_grid


class Decoder2D_v3(nn.Module):
    def __init__(self, latent_dim=155):
        super(Decoder2D_v3, self).__init__()
        feature = 4
        # [N,  1, 4, 16]

        self.convtr = nn.Sequential(
            nn.ConvTranspose2d(1, feature, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(feature),
            nn.LeakyReLU(0.2),
            # N, 32, 4, 16

            nn.ConvTranspose2d(feature, feature, kernel_size=(3, 3), stride=(1, 1), padding=2),
            nn.BatchNorm2d(feature),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(feature, feature * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.BatchNorm2d(feature * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(feature * 2, feature * 3, kernel_size=(4, 4), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(feature * 3),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(feature * 3, feature * 2, kernel_size=(3, 8), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(feature * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(feature * 2, feature, kernel_size=(3, 8), stride=(2, 2), padding=(1, 2),
                               output_padding=(0, 1)),
            nn.BatchNorm2d(feature),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(feature, 1, kernel_size=(2, 4), stride=1, padding=(1, 2)),
            # nn.AdaptiveAvgPool2d((16, 128)),
            nn.Sigmoid()
            # N, 1, 16, 128

        )

    def forward(self, l):
        # print(l.shape)
        return self.convtr(l)


class Encoder2D_v3(nn.Module):
    def __init__(self, kernel=2):
        super(Encoder2D_v3, self).__init__()

        self.conv = nn.Sequential(
            # input shape: N, 1, 32, 256
            nn.Conv2d(1, 3, kernel_size=(2, 5), stride=1, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 5, kernel_size=(3, 8), stride=1, padding=1),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 8, kernel_size=(3, 8), stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8, 5, kernel_size=1, padding=0),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),
        )
        self.mu = nn.Sequential(

            nn.Conv2d(5, 5, kernel_size=(4, 4), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 1, kernel_size=1, padding=0)
        )
        # [N,  1, 4, 16]
        self.logvar = nn.Sequential(
            nn.Conv2d(5, 5, kernel_size=(4, 4), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 1, kernel_size=1, padding=0)
        )
        # [N,  1, 4, 16]

    def forward(self, x):
        h = self.conv(x)
        return self.mu(h), self.logvar(h)  # .squeeze(1).squeeze(1)


class Type2_Encoder2D_v3(nn.Module):
    def __init__(self, sigmoid=True):
        super(Type2_Encoder2D_v3, self).__init__()
        self.is_sigmoid = sigmoid
        self.sigm = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            # input shape: N, 1, 32, 256
            nn.Conv2d(1, 3, kernel_size=(2, 5), stride=1, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 5, kernel_size=(3, 8), stride=1, padding=1),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 8, kernel_size=(3, 8), stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8, 5, kernel_size=1, padding=0),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(

            nn.Conv2d(5, 5, kernel_size=(4, 4), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 1, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )
        # [N,  1, 4, 16]

    def forward(self, x):
        h = self.conv1(x)
        return self.sigm(self.conv2(h)) if self.is_sigmoid else self.conv2(h)  # .squeeze(1).squeeze(1)


class Type2_Encoder2DHigher_v3(nn.Module):
    def __init__(self, sigmoid=True):
        super(Type2_Encoder2DHigher_v3, self).__init__()
        self.is_sigmoid = sigmoid
        self.sigm = nn.Sigmoid()
        self.conv = nn.Sequential(
            # input shape: N, 1, 32, 256
            nn.Conv2d(1, 3, kernel_size=(3, 5), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 5, kernel_size=(3, 9), padding=(1, 4)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 8, kernel_size=(3, 9), padding=(1, 4)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8, 5, kernel_size=(3, 9), padding=(1, 4)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

        )
        self.conv2 = nn.Sequential(

            nn.Conv2d(5, 5, kernel_size=(4, 29)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 4, kernel_size=(4, 29)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 3, kernel_size=(4, 29)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 1, kernel_size=(4, 29))
        )

    def forward(self, x):
        h = self.conv(x)
        return self.sigm(self.conv2(h)) if self.is_sigmoid else self.conv2(h)  # .squeeze(1).squeeze(1)


class Type2_Encoder2DLower_v3(nn.Module):
    def __init__(self, sigmoid=True):
        super(Type2_Encoder2DLower_v3, self).__init__()
        self.is_sigmoid = sigmoid
        self.sigm = nn.Sigmoid()
        self.conv = nn.Sequential(
            # input shape: N, 1, 32, 256
            nn.Conv2d(1, 3, kernel_size=(3, 5), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 5, kernel_size=(3, 9), stride=1, padding=(1, 4)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 5, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(

            nn.Conv2d(5, 4, kernel_size=(3, 9), stride=(2, 2)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 3, kernel_size=(3, 8)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 1, kernel_size=(2, 6))
        )

    def forward(self, x):
        h = self.conv(x)
        return self.sigm(self.conv2(h)) if self.is_sigmoid else self.conv2(h)  # .squeeze(1).squeeze(1)


class Encoder2DHigher_v3(nn.Module):
    def __init__(self, kernel=2):
        super(Encoder2DHigher_v3, self).__init__()

        self.conv = nn.Sequential(
            # input shape: N, 1, 32, 256
            nn.Conv2d(1, 3, kernel_size=(3, 5), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 5, kernel_size=(3, 9), padding=(1, 4)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 8, kernel_size=(3, 9), padding=(1, 4)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8, 5, kernel_size=(3, 9), padding=(1, 4)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

        )
        self.mu = nn.Sequential(

            nn.Conv2d(5, 5, kernel_size=(4, 29)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 4, kernel_size=(4, 29)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 3, kernel_size=(4, 29)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 1, kernel_size=(4, 29))
        )
        # [N,  1, 4, 16]
        self.logvar = nn.Sequential(
            nn.Conv2d(5, 5, kernel_size=(4, 29)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 4, kernel_size=(4, 29)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 3, kernel_size=(4, 29)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 1, kernel_size=(4, 29))
        )
        # [N,  1, 4, 16]

    def forward(self, x):
        h = self.conv(x)
        return self.mu(h), self.logvar(h)  # .squeeze(1).squeeze(1)


class Discriminator2D_ForGrids_v3(nn.Module):
    def __init__(self):
        super(Discriminator2D_ForGrids_v3, self).__init__()

        self.conv = nn.Sequential(
            # input shape: N, 1, 32, 256
            nn.Conv2d(1, 3, kernel_size=(2, 5), stride=1, padding=2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 5, kernel_size=(3, 8), stride=1, padding=1),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 8, kernel_size=(3, 8), stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(8, 5, kernel_size=1, padding=0),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 5, kernel_size=(4, 4), stride=(1, 2), padding=(1, 2)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 1, kernel_size=(4, 16), padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator2D_Latent_v3(nn.Module):
    def __init__(self):
        super(Discriminator2D_Latent_v3, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(2, 4)),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 5, kernel_size=(2, 4)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 7, kernel_size=(2, 4)),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(0.2),

            nn.Conv2d(7, 4, kernel_size=(1, 4)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 1, kernel_size=(1, 4)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator2D_Latent2_v3(nn.Module):
    def __init__(self):
        super(Discriminator2D_Latent2_v3, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(2, 3)),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),

            nn.Conv2d(3, 5, kernel_size=(2, 3)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),

            nn.Conv2d(5, 7, kernel_size=(2, 3)),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(0.2),

            nn.Conv2d(7, 9, kernel_size=(1, 3)),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(0.2),

            nn.Conv2d(9, 9, kernel_size=1),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(0.2),

            nn.Conv2d(9, 9, kernel_size=1),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(0.2),

            nn.Conv2d(9, 7, kernel_size=(1, 3)),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(0.2),

            nn.Conv2d(7, 4, kernel_size=(1, 3)),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4, 1, kernel_size=(1, 4)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator2D_Latent3_v3(nn.Module):
    def __init__(self):
        super(Discriminator2D_Latent3_v3, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear(x.view(-1, 64))


class ADVAE2D_Bernoulli(BPModule):
    def __init__(self, encoder, decoder, discriminator):
        super(ADVAE2D_Bernoulli, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.bce = nn.BCELoss()
        self.losses_keys = ['disc train', 'generator train', 'disc valid', 'generator valid']

    def sampler(self, p):
        U = torch.rand(p.size()).to(p.device)
        delta = p - U
        ind = delta > 0
        ind = ind.float()
        return (ind - p).detach() + p

    def forward(self, x):
        p = self.encoder(x)
        # TODO: NEm gaussi hanem Bernoulli sampler
        z = self.sampler(p)
        # print(z.shape, mu.shape, logvar.shape)
        pred = self.decoder(z)  # return h
        return {"output": pred, "p": p}, z

    def training_step(self, optim_configuration, step):
        self.train()

        epoch_recon = 0
        epoch_gen = 0
        epoch_disc = 0
        for batch in self.trainer.dataloaders["train"]:
            theta = self.encoder(batch)
            # itt baj lehet, nem  a self(batch) megy

            ### Disc
            z_real = self.sampler(torch.ones_like(theta, dtype=theta.dtype) * 0.5)
            z = self.sampler(theta)
            d_real = self.discriminator(z_real)
            d_fake = self.discriminator(z)

            loss_real = self.bce(d_real, torch.ones_like(d_real))
            loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
            disc_loss = (loss_real + loss_fake) / 2
            disc_loss.backward(retain_graph=True)
            epoch_disc = epoch_disc + disc_loss.item()
            # optim_configuration[0][2].step()
            # optim_configuration[1][2].step()
            # !Annealing
            optim_configuration[2].step()

            self.optimizer_zero_grad(0, 0, optim_configuration, step)

            ### Generator
            # mu, logvar = self.encoder(self.trainer.dataloaders["train"])
            z = self.sampler(theta)
            d_fake = self.discriminator(z)
            gen_loss = self.bce(d_fake, torch.ones_like(d_fake))
            # if step > 2000:
            gen_loss.backward()
            epoch_gen = epoch_gen + gen_loss.item()
            # optim_configuration[0][0].step()
            # optim_configuration[1][0].step()
            # !Annealing
            optim_configuration[0].step()
            self.optimizer_zero_grad(0, 0, optim_configuration, step)

            ### Reconstruction
            kwargs, z = self(batch)
            recon_loss_vae = self.bce(kwargs["output"], batch)
            recon_loss_vae.backward()
            epoch_recon = epoch_recon + recon_loss_vae.item()
            # opt_vae
            # optim_configuration[0][1].step()
            # optim_configuration[1][1].step()
            # !Annealing
            optim_configuration[1].step()
            self.optimizer_zero_grad(0, 0, optim_configuration, step)

        N = len(self.trainer.dataloaders["train"])
        self.trainer.losses["train"].append(epoch_recon / N)
        self.trainer.losses["disc train"].append(epoch_disc / N)
        self.trainer.losses["generator train"].append(epoch_gen / N)

    def validation_step(self, step):
        self.eval()
        self.freeze()
        epoch_recon = 0
        epoch_gen = 0
        epoch_disc = 0
        for batch in self.trainer.dataloaders["valid"]:
            kwargs, z = self(batch)
            ### Disc
            z_real = self.sampler(torch.ones_like(kwargs["p"], dtype=kwargs["p"].dtype) * 0.5)
            d_real = self.discriminator(z_real)
            d_fake = self.discriminator(z)
            loss_real = self.bce(d_real, torch.ones_like(d_real))
            loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
            disc_loss = (loss_real + loss_fake) / 2
            epoch_disc = epoch_disc + disc_loss.item()

            ### Generator
            gen_loss = self.bce(d_fake, torch.ones_like(d_fake))
            # if step > 2000:
            epoch_gen = epoch_gen + gen_loss.item()

            ### Reconstruction
            recon_loss_vae = self.bce(kwargs["output"], batch)
            epoch_recon = epoch_recon + recon_loss_vae.item()

        # var = kwargs["logvar"].exp_()
        # mean_of_var = torch.mean(var)
        # std_of_var = torch.std(var)
        # self.trainer.writer.add_scalars('latent var.', {'std': std_of_var,
        #                                                 'mean of latent var.': mean_of_var}, step)

        N = len(self.trainer.dataloaders["valid"])
        self.trainer.losses["valid"].append(epoch_recon / N)
        self.trainer.losses["disc valid"].append(epoch_disc / N)
        self.trainer.losses["generator valid"].append(epoch_gen / N)

        # Images
        with torch.no_grad():
            # if step % 10 == 1:
            img_fake_grid = make_grid(boundary_for_grid(kwargs["output"][:16]), normalize=True, nrow=2)
            img_real_grid = make_grid(boundary_for_grid(batch[:16]), normalize=True, nrow=2)

            img_latent_grid = make_grid(boundary_for_grid(z[:16]), normalize=True, nrow=2)
            img_theta_dist_grid = make_grid(boundary_for_grid(kwargs["p"][:16]), normalize=True, nrow=2)
            img_prior_grid = make_grid(boundary_for_grid(z_real[:16]), normalize=True, nrow=2)

            self.trainer.writer.add_image("Occupancy Real Images", img_real_grid)
            self.trainer.writer.add_image("Occupancy Fake Images", img_fake_grid, step)

            self.trainer.writer.add_image("Latent Distribution Images", img_theta_dist_grid, step)
            self.trainer.writer.add_image("Latent Vector Images", img_latent_grid, step)
            self.trainer.writer.add_image("Prior Images", img_prior_grid, step)

        self.unfreeze()

    def configure_optimizers(self):
        opt_encoder = optim.Adam(self.encoder.parameters(), lr=0.001)
        opt_vae = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)
        opt_disc = optim.SGD(self.discriminator.parameters(), lr=0.001)

        # sch_enc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_encoder, T_max=1500)
        # sch_vae = torch.optim.lr_scheduler.MultiStepLR(opt_vae, milestones=[8000, 80000, 120000, 170000], gamma=0.8)
        # sch_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=1000)
        return [opt_encoder, opt_vae, opt_disc]  # , [sch_enc, sch_vae, sch_disc]

    def optimizer_zero_grad(
            self, epoch: int, batch_idx: int, optimizer: Union[optim.Optimizer, List], optimizer_idx: int):
        for opt in optimizer:  # [0]:
            opt.zero_grad()


class ADVAE2D_Gauss(BPModule):
    def __init__(self, encoder, decoder, discriminator):
        super(ADVAE2D_Gauss, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.bce = nn.BCELoss()
        self.losses_keys = ['disc train', 'generator train', 'disc valid', 'generator valid']

    def sampler(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        # TODO: f(k, p) miatt q, és p generálása
        # TODO: NEm gaussi hanem Bernoulli sampler
        z = self.sampler(mu, logvar)
        # print(z.shape, mu.shape, logvar.shape)
        pred = self.decoder(z)  # return h
        return {"output": pred, "mu": mu, "logvar": logvar}, z

    def training_step(self, optim_configuration, step):
        self.train()

        epoch_recon = 0
        epoch_gen = 0
        epoch_disc = 0
        for batch in self.trainer.dataloaders["train"]:
            mu, logvar = self.encoder(batch)
            # itt baj lehet, nem  a self(batch) megy

            ### Disc
            z_real = torch.FloatTensor(mu.size()).normal_().to(mu.device)
            z = self.sampler(mu, logvar)
            d_real = self.discriminator(z_real)
            d_fake = self.discriminator(z)

            loss_real = self.bce(d_real, torch.ones_like(d_real))
            loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
            disc_loss = (loss_real + loss_fake) / 2
            disc_loss.backward(retain_graph=True)
            epoch_disc = epoch_disc + disc_loss.item()
            # optim_configuration[0][2].step()
            # optim_configuration[1][2].step()
            # !Annealing
            optim_configuration[2].step()

            self.optimizer_zero_grad(0, 0, optim_configuration, step)

            ### Generator
            # mu, logvar = self.encoder(self.trainer.dataloaders["train"])
            z = self.sampler(mu, logvar)
            d_fake = self.discriminator(z)
            gen_loss = self.bce(d_fake, torch.ones_like(d_fake))
            # if step > 2000:
            gen_loss.backward()
            epoch_gen = epoch_gen + gen_loss.item()
            # optim_configuration[0][0].step()
            # optim_configuration[1][0].step()
            # !Annealing
            optim_configuration[0].step()
            self.optimizer_zero_grad(0, 0, optim_configuration, step)

            ### Reconstruction
            kwargs, z = self(batch)
            recon_loss_vae = self.bce(kwargs["output"], batch)
            recon_loss_vae.backward()
            epoch_recon = epoch_recon + recon_loss_vae.item()
            # opt_vae
            # optim_configuration[0][1].step()
            # optim_configuration[1][1].step()
            # !Annealing
            optim_configuration[1].step()
            self.optimizer_zero_grad(0, 0, optim_configuration, step)

        N = len(self.trainer.dataloaders["train"])
        self.trainer.losses["train"].append(epoch_recon / N)
        self.trainer.losses["disc train"].append(epoch_disc / N)
        self.trainer.losses["generator train"].append(epoch_gen / N)

    def validation_step(self, step):
        self.eval()
        self.freeze()
        epoch_recon = 0
        epoch_gen = 0
        epoch_disc = 0
        for batch in self.trainer.dataloaders["valid"]:
            kwargs, z = self(batch)
            ### Disc
            z_real = torch.FloatTensor(z.size()).normal_().to(z.device)
            d_real = self.discriminator(z_real)
            d_fake = self.discriminator(z)
            loss_real = self.bce(d_real, torch.ones_like(d_real))
            loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
            disc_loss = (loss_real + loss_fake) / 2
            epoch_disc = epoch_disc + disc_loss.item()

            ### Generator
            gen_loss = self.bce(d_fake, torch.ones_like(d_fake))
            # if step > 2000:
            epoch_gen = epoch_gen + gen_loss.item()

            ### Reconstruction
            recon_loss_vae = self.bce(kwargs["output"], batch)
            epoch_recon = epoch_recon + recon_loss_vae.item()

        var = kwargs["logvar"].exp_()
        mean_of_var = torch.mean(var)
        std_of_var = torch.std(var)
        self.trainer.writer.add_scalars('latent var.', {'std': std_of_var,
                                                        'mean of latent var.': mean_of_var}, step)

        N = len(self.trainer.dataloaders["valid"])
        self.trainer.losses["valid"].append(epoch_recon / N)
        self.trainer.losses["disc valid"].append(epoch_disc / N)
        self.trainer.losses["generator valid"].append(epoch_gen / N)

        # Images
        with torch.no_grad():
            # if step % 10 == 1:
            img_fake_grid = make_grid(boundary_for_grid(kwargs["output"][:16]), normalize=True, nrow=2)
            img_real_grid = make_grid(boundary_for_grid(batch[:16]), normalize=True, nrow=2)

            img_latent_dist_grid = make_grid(boundary_for_grid(z[:16]), normalize=True, nrow=2)
            img_prior_dist_grid = make_grid(boundary_for_grid(z_real[:16]), normalize=True, nrow=2)

            self.trainer.writer.add_image("Occupancy Real Images", img_real_grid)
            self.trainer.writer.add_image("Occupancy Fake Images", img_fake_grid, step)

            self.trainer.writer.add_image("Latent Distribution Images", img_latent_dist_grid, step)
            self.trainer.writer.add_image("Prior Distribution Images", img_prior_dist_grid, step)

        self.unfreeze()

    def configure_optimizers(self):
        opt_encoder = optim.Adam(self.encoder.parameters(), lr=0.001)
        opt_vae = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)
        opt_disc = optim.SGD(self.discriminator.parameters(), lr=0.001)

        # sch_enc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_encoder, T_max=1500)
        # sch_vae = torch.optim.lr_scheduler.MultiStepLR(opt_vae, milestones=[8000, 80000, 120000, 170000], gamma=0.8)
        # sch_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=1000)
        return [opt_encoder, opt_vae, opt_disc]  # , [sch_enc, sch_vae, sch_disc]

    def optimizer_zero_grad(
            self, epoch: int, batch_idx: int, optimizer: Union[optim.Optimizer, List], optimizer_idx: int):
        for opt in optimizer:  # [0]:
            opt.zero_grad()


class VAE2D_Bernoulli(BPModule):
    def __init__(self, encoder, decoder):
        super(VAE2D_Bernoulli, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.losses_keys = ['train', 'valid']

    def sampler(self, p):
        U = torch.rand(p.size()).to(p.device)
        delta = p - U
        ind = delta > 0
        ind = ind.float()
        return (ind - p).detach() + p

    def forward(self, x):
        p = self.encoder(x)
        z = self.sampler(p)
        # print(z.shape, mu.shape, logvar.shape)
        pred = self.decoder(z)  # return h
        return {"output": pred, "p": p}, z

    def training_step(self, optim_configuration, step):
        self.train()
        self.optimizer_zero_grad(0, 0, optim_configuration, 0)
        epoch_loss = 0
        for batch in self.trainer.dataloaders["train"]:
            kwargs, z = self(batch)
            loss = self.trainer.criterion(**kwargs, target=batch)
            loss.backward()
            optim_configuration.step()
            epoch_loss = epoch_loss + loss.item()
        self.trainer.losses["train"].append(epoch_loss / len(self.trainer.dataloaders["train"]))
        # self.trainer.writer.add_scalar('train loss', loss.item(), step)

    def validation_step(self, step):
        self.eval()
        self.freeze()
        epoch_loss = 0
        for batch in self.trainer.dataloaders["valid"]:
            kwargs, z = self(batch)
            loss = self.trainer.criterion(**kwargs, target=batch)
            epoch_loss = epoch_loss + loss.item()

        # var = kwargs["logvar"].exp_()
        # mean_of_var = torch.mean(var)
        # std_of_var = torch.std(var)
        # self.trainer.writer.add_scalars('latent var.', {'std': std_of_var,
        #                                                 'mean of latent var.': mean_of_var}, step)

        mean = torch.mean(kwargs["p"])
        std = torch.mean(torch.std(kwargs["p"], dim=(2, 3)))
        self.trainer.writer.add_scalars('Latent vector mean and std', {'mean': mean,
                                                                       "std": std}, step)

        # Images
        with torch.no_grad():
            # if step % 10 == 1:
            img_fake_grid = make_grid(boundary_for_grid(kwargs["output"][:16]), normalize=True, nrow=2)
            img_real_grid = make_grid(boundary_for_grid(batch[:16]), normalize=True, nrow=2)

            self.trainer.writer.add_image("Occupancy Real Images", img_real_grid)
            self.trainer.writer.add_image("Occupancy Fake Images", img_fake_grid, step)

        self.trainer.losses["valid"].append(epoch_loss / len(self.trainer.dataloaders["valid"]))
        self.unfreeze()
        # self.trainer.writer.add_scalar('valid loss', loss.item(), step)

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters())


class ADVAE2D_Uniform(BPModule):
    def __init__(self, encoder, decoder, discriminator):
        super(ADVAE2D_Uniform, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.bce = nn.BCELoss()
        self.losses_keys = ['disc train', 'generator train', 'disc valid', 'generator valid']

    def sampler(self, a, b):
        U = torch.rand(a.size()).to(a.device)
        return U * (b - a) + a

    def forward(self, x):
        a, b = self.encoder(x)
        z = self.sampler(a, b)
        pred = self.decoder(z)  # return h
        return {"output": pred, "a": a, "b": b}, z

    def training_step(self, optim_configuration, step):
        self.train()

        epoch_recon = 0
        epoch_gen = 0
        epoch_disc = 0
        for batch in self.trainer.dataloaders["train"]:
            a, b = self.encoder(batch)

            ### Disc
            z_real = torch.rand(a.size()).to(a.device)
            z = self.sampler(a, b)
            d_real = self.discriminator(z_real)
            d_fake = self.discriminator(z)

            loss_real = self.bce(d_real, torch.ones_like(d_real))
            loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
            disc_loss = (loss_real + loss_fake) / 2
            disc_loss.backward(retain_graph=True)
            epoch_disc = epoch_disc + disc_loss.item()
            # optim_configuration[0][2].step()
            # optim_configuration[1][2].step()
            # !Annealing
            optim_configuration[2].step()

            self.optimizer_zero_grad(0, 0, optim_configuration, step)

            ### Generator
            # mu, logvar = self.encoder(self.trainer.dataloaders["train"])
            z = self.sampler(a, b)
            d_fake = self.discriminator(z)
            gen_loss = self.bce(d_fake, torch.ones_like(d_fake))
            # if step > 2000:
            gen_loss.backward()
            epoch_gen = epoch_gen + gen_loss.item()
            # optim_configuration[0][0].step()
            # optim_configuration[1][0].step()
            # !Annealing
            optim_configuration[0].step()
            self.optimizer_zero_grad(0, 0, optim_configuration, step)

            ### Reconstruction
            kwargs, z = self(batch)
            recon_loss_vae = self.bce(kwargs["output"], batch)
            recon_loss_vae.backward()
            epoch_recon = epoch_recon + recon_loss_vae.item()
            # opt_vae
            # optim_configuration[0][1].step()
            # optim_configuration[1][1].step()
            # !Annealing
            optim_configuration[1].step()
            self.optimizer_zero_grad(0, 0, optim_configuration, step)

        N = len(self.trainer.dataloaders["train"])
        self.trainer.losses["train"].append(epoch_recon / N)
        self.trainer.losses["disc train"].append(epoch_disc / N)
        self.trainer.losses["generator train"].append(epoch_gen / N)

    def validation_step(self, step):
        self.eval()
        self.freeze()
        epoch_recon = 0
        epoch_gen = 0
        epoch_disc = 0
        for batch in self.trainer.dataloaders["valid"]:
            kwargs, z = self(batch)
            ### Disc
            z_real = torch.FloatTensor(z.size()).normal_().to(z.device)
            d_real = self.discriminator(z_real)
            d_fake = self.discriminator(z)
            loss_real = self.bce(d_real, torch.ones_like(d_real))
            loss_fake = self.bce(d_fake, torch.zeros_like(d_fake))
            disc_loss = (loss_real + loss_fake) / 2
            epoch_disc = epoch_disc + disc_loss.item()

            ### Generator
            gen_loss = self.bce(d_fake, torch.ones_like(d_fake))
            # if step > 2000:
            epoch_gen = epoch_gen + gen_loss.item()

            ### Reconstruction
            recon_loss_vae = self.bce(kwargs["output"], batch)
            epoch_recon = epoch_recon + recon_loss_vae.item()

        A = torch.mean(kwargs["a"])
        B = torch.std(kwargs["b"])
        self.trainer.writer.add_scalars('latent Uniform', {'A': A,
                                                           'B': B}, step)

        N = len(self.trainer.dataloaders["valid"])
        self.trainer.losses["valid"].append(epoch_recon / N)
        self.trainer.losses["disc valid"].append(epoch_disc / N)
        self.trainer.losses["generator valid"].append(epoch_gen / N)

        # Images
        with torch.no_grad():
            # if step % 10 == 1:
            img_fake_grid = make_grid(boundary_for_grid(kwargs["output"][:16]), normalize=True, nrow=2)
            img_real_grid = make_grid(boundary_for_grid(batch[:16]), normalize=True, nrow=2)

            img_latent_dist_grid = make_grid(boundary_for_grid(z[:16]), normalize=True, nrow=2)
            img_prior_dist_grid = make_grid(boundary_for_grid(z_real[:16]), normalize=True, nrow=2)

            self.trainer.writer.add_image("Occupancy Real Images", img_real_grid)
            self.trainer.writer.add_image("Occupancy Fake Images", img_fake_grid, step)

            self.trainer.writer.add_image("Latent Distribution Images", img_latent_dist_grid, step)
            self.trainer.writer.add_image("Prior Distribution Images", img_prior_dist_grid, step)

        self.unfreeze()

    def configure_optimizers(self):
        opt_encoder = optim.Adam(self.encoder.parameters(), lr=0.001)
        opt_vae = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)
        opt_disc = optim.SGD(self.discriminator.parameters(), lr=0.001)

        # sch_enc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_encoder, T_max=1500)
        # sch_vae = torch.optim.lr_scheduler.MultiStepLR(opt_vae, milestones=[8000, 80000, 120000, 170000], gamma=0.8)
        # sch_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=1000)
        return [opt_encoder, opt_vae, opt_disc]  # , [sch_enc, sch_vae, sch_disc]

    def optimizer_zero_grad(
            self, epoch: int, batch_idx: int, optimizer: Union[optim.Optimizer, List], optimizer_idx: int):
        for opt in optimizer:  # [0]:
            opt.zero_grad()


class KLD_BCE_loss_2Dvae_bernoulli(nn.Module):
    def __init__(self, lam):
        super(KLD_BCE_loss_2Dvae_bernoulli, self).__init__()
        self.bce = nn.BCELoss()
        self.lam = lam

    def forward(self, output, p, target):
        loss = self.bce(output, target)
        KL = p * torch.log(2 * p) + (1 - p) * torch.log(2 * (1 - p))
        KLD = self.lam * torch.mean(KL)
        return loss + KLD
