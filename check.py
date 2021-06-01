from modelv3 import *


t = torch.randn(17,1,16,128)
enc = Encoder2D_v3()
dec = Decoder2D_v3()
z, l = enc(t)
print("encoder")
print(sum(p.numel() for p in enc.parameters() if p.requires_grad))

print(z.shape, z.shape[2]*z.shape[3])

tt = dec(z)
print("decoder")
print(sum(p.numel() for p in dec.parameters() if p.requires_grad))

print(tt.shape)
var=torch.mean(z)
print(var, var.shape)
