from modelv3 import *

common = [nn.Conv2d(1, 3, kernel_size=(2, 5), stride=1, padding=2),
          nn.Conv2d(3, 5, kernel_size=(3, 8), stride=1, padding=1),
          nn.Conv2d(5, 8, kernel_size=(3, 8), stride=1, padding=0),
          nn.Conv2d(8, 5, kernel_size=1, padding=0),
          nn.Conv2d(5, 5, kernel_size=(4, 4), stride=(1, 2), padding=(1, 2)),
          nn.Conv2d(5, 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
          nn.Conv2d(4, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
          nn.Conv2d(3, 1, kernel_size=1, padding=0)
          ]

feature = 4
decoder = [nn.ConvTranspose2d(1, feature, kernel_size=(3, 3), stride=1, padding=0),
           nn.ConvTranspose2d(feature, feature, kernel_size=(3, 3), stride=(1, 1), padding=2),
           nn.ConvTranspose2d(feature, feature * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
           nn.ConvTranspose2d(feature * 2, feature * 3, kernel_size=(4, 4), stride=(1, 2), padding=(1, 2)),
           nn.ConvTranspose2d(feature * 3, feature * 2, kernel_size=(3, 8), stride=(1, 1), padding=(1, 1)),
           nn.ConvTranspose2d(feature * 2, feature, kernel_size=(3, 8), stride=(2, 2), padding=(1, 2),
                           output_padding=(0, 1)),
           nn.ConvTranspose2d(feature, 1, kernel_size=(2, 4), stride=1, padding=(1, 2))]
           # N, 1, 16, 128]


t = torch.randn(17,1,16,128)
print(t.shape)
print("Encoder")
for layer in common:
    t=layer(t)
    print(t.shape)

encoder = Encoder2D_v3()
print(sum(p.numel() for p in encoder.parameters() if p.requires_grad))


print("Discriminator")
disc = Discriminator2D_Latent3_v3()
print(disc(t).shape)
print(sum(p.numel() for p in disc.parameters() if p.requires_grad))
# print(disc(t))

print("Decoder")
for layer in decoder:
    t=layer(t)
    print(t.shape)

