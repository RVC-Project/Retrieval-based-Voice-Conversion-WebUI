############离线VC参数
inp_root=r"白鹭霜华长条"#对输入目录下所有音频进行转换，别放非音频文件
opt_root=r"opt"#输出目录
f0_up_key=0#升降调，整数，男转女12，女转男-12
person=r"weights\洛天依v3.pt"#目前只有洛天依v3
############硬件参数
device = "cuda:0"#填写cuda:x或cpu，x指代第几张卡，只支持N卡加速
is_half=True#9-10-20-30-40系显卡无脑True，不影响质量，>=20显卡开启有加速
n_cpu=0#默认0用上所有线程，写数字限制CPU资源使用
############下头别动
import torch
if(torch.cuda.is_available()==False):
    print("没有发现支持的N卡，使用CPU进行推理")
    device="cpu"
    is_half=False
if(device!="cpu"):
    gpu_name=torch.cuda.get_device_name(int(device.split(":")[-1]))
    if("16"in gpu_name):
        print("16系显卡强制单精度")
        is_half=False
from multiprocessing import cpu_count
if(n_cpu==0):n_cpu=cpu_count()
if(is_half==True):
    #6G显存配置
    x_pad       =   3
    x_query     =   10
    x_center    =   60
    x_max       =   65
else:
    #5G显存配置
    x_pad       =   1
    # x_query     =   6
    # x_center    =   30
    # x_max       =   32
    #6G显存配置
    x_query     =   6
    x_center    =   38
    x_max       =   41