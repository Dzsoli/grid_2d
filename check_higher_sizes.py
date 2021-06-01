from modelv3 import *

common = [nn.Conv2d(1, 3, kernel_size=(3, 5), padding=(1,2)),
          nn.Conv2d(3, 5, kernel_size=(3, 9), stride=1, padding=(1,4)),
          # nn.Conv2d(5, 5, kernel_size=(3, 9), stride=1, padding=(1,4)),
          nn.Conv2d(5, 5, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),

          # nn.Conv2d(5, 5, kernel_size=(3, 5), stride=(2, 2)),
          nn.Conv2d(5, 4, kernel_size=(3, 9), stride=(2, 2)),
          nn.Conv2d(4, 3, kernel_size=(3, 8)),
          nn.Conv2d(3, 1, kernel_size=(2, 6))
          ]
common2 = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=(3, 5), padding=(1, 2)),
    nn.Conv2d(3, 5, kernel_size=(3, 9), stride=1, padding=(1, 4)),
    nn.Conv2d(5, 5, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),

    nn.Conv2d(5, 4, kernel_size=(3, 9), stride=(2, 2)),
    nn.Conv2d(4, 3, kernel_size=(3, 8)),
    nn.Conv2d(3, 1, kernel_size=(2, 6))
)

print(sum(p.numel() for p in common2.parameters() if p.requires_grad))

t = torch.randn(17,1,16,128)
print(t.shape)
print("Encoder")
for layer in common:
    t=layer(t)
    print(t.shape)

encoder = Encoder2DHigher_v3()
print(sum(p.numel() for p in encoder.parameters() if p.requires_grad))

encoder = Encoder2D_v3()
print(sum(p.numel() for p in encoder.parameters() if p.requires_grad))
