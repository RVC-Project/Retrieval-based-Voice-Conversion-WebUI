import pdb

import torch

# a=torch.load(r"E:\codes\py39\vits_vc_gpu_train\logs\ft-mi-suc\G_1000.pth")["model"]#sim_nsf#
# a=torch.load(r"E:\codes\py39\vits_vc_gpu_train\logs\ft-mi-freeze-vocoder-flow-enc_q\G_1000.pth")["model"]#sim_nsf#
# a=torch.load(r"E:\codes\py39\vits_vc_gpu_train\logs\ft-mi-freeze-vocoder\G_1000.pth")["model"]#sim_nsf#
# a=torch.load(r"E:\codes\py39\vits_vc_gpu_train\logs\ft-mi-test\G_1000.pth")["model"]#sim_nsf#
a = torch.load(
    r"E:\codes\py39\vits_vc_gpu_train\logs\ft-mi-no_opt-no_dropout\G_1000.pth"
)[
    "model"
]  # sim_nsf#
for key in a.keys():
    a[key] = a[key].half()
# torch.save(a,"ft-mi-freeze-vocoder_true_1k.pt")#
# torch.save(a,"ft-mi-sim1k.pt")#
torch.save(a, "ft-mi-no_opt-no_dropout.pt")  #
