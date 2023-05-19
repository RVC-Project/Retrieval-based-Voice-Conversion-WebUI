import torch, os, traceback, sys, warnings, shutil, numpy as np
from tqdm import tqdm
import argparse
import threading
from time import sleep
from subprocess import Popen
import faiss
from random import shuffle
from i18n import I18nAuto
from config_cli import Config

config = Config()

weight_root = "weights"
index_root = "logs"

i18n = I18nAuto()

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if (not torch.cuda.is_available()) or ngpu == 0:
    if_gpu_ok = False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if ("10" in gpu_name or "16" in gpu_name or "20" in gpu_name
                or "30" in gpu_name or "40" in gpu_name
                or "A2" in gpu_name.upper() or "A3" in gpu_name.upper()
                or "A4" in gpu_name.upper() or "P4" in gpu_name.upper()
                or "A50" in gpu_name.upper() or "70" in gpu_name
                or "80" in gpu_name or "90" in gpu_name
                or "M4" in gpu_name.upper() or "T4" in gpu_name.upper()
                or "TITAN"
                in gpu_name.upper()):  # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory / 1024 /
                    1024 / 1024 + 0.4))
if if_gpu_ok == True and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


def if_done(done, p):
    while 1:
        if p.poll() == None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() == None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


now_dir = os.getcwd()
sys.path.append(now_dir)


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    cmd = (config.python_cmd +
           " trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s " %
           (trainset_dir, sr, n_p, now_dir, exp_dir) + str(config.noparallel))
    print(cmd)
    p = Popen(cmd,
              shell=True)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    ##煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19):
    print(gpus, n_p, f0method, if_f0, exp_dir, version19)
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        cmd = config.python_cmd + " extract_f0_print_cli.py %s/logs/%s %s %s" % (
            now_dir,
            exp_dir,
            n_p,
            f0method,
        )
        print(cmd)
        p = Popen(cmd, shell=True,
                  cwd=now_dir)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
        ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
        done = [False]
        threading.Thread(
            target=if_done,
            args=(
                done,
                p,
            ),
        ).start()
    ####对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (config.python_cmd +
               " extract_feature_print_cli.py %s %s %s %s %s/logs/%s %s" % (
                   config.device,
                   leng,
                   idx,
                   n_g,
                   now_dir,
                   exp_dir,
                   version19,
               ))
        print(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = ("%s/3_feature256" %
                   (exp_dir) if version19 == "v1" else "%s/3_feature768" %
                   (exp_dir))
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)]))
    else:
        names = set([
            name.split(".")[0] for name in os.listdir(gt_wavs_dir)
        ]) & set([name.split(".")[0] for name in os.listdir(feature_dir)])
    opt = []
    print('正在划分数据集 生成filelist')
    for name in tqdm(names):
        if if_f0_3:
            opt.append("%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s" % (
                gt_wavs_dir.replace("\\", "\\\\"),
                name,
                feature_dir.replace("\\", "\\\\"),
                name,
                f0_dir.replace("\\", "\\\\"),
                name,
                f0nsf_dir.replace("\\", "\\\\"),
                name,
                spk_id5,
            ))
        else:
            opt.append("%s/%s.wav|%s/%s.npy|%s" % (
                gt_wavs_dir.replace("\\", "\\\\"),
                name,
                feature_dir.replace("\\", "\\\\"),
                name,
                spk_id5,
            ))
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5))
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5))
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:", gpus16)
    if gpus16:
        cmd = (
            config.python_cmd +
            " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s -pg %s -pd %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                pretrained_G14,
                pretrained_D15,
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            ))
    else:
        cmd = (
            config.python_cmd +
            " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s -pg %s -pd %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                pretrained_G14,
                pretrained_D15,
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            ))
    print(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = ("%s/3_feature256" %
                   (exp_dir) if version19 == "v1" else "%s/3_feature768" %
                   (exp_dir))
    if os.path.exists(feature_dir) == False:
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos = []
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768,
                                "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s.index" %
        (exp_dir, n_ivf, index_ivf.nprobe, version19),
    )
    # faiss.write_index(index, '%s/trained_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in tqdm(range(0, big_npy.shape[0], batch_size_add)):
        index.add(big_npy[i:i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s.index" %
        (exp_dir, n_ivf, index_ivf.nprobe, version19),
    )
    infos.append("成功构建索引，added_IVF%s_Flat_nprobe_%s_%s.index" %
                 (n_ivf, index_ivf.nprobe, version19))
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    model_log_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    preprocess_log_path = "%s/preprocess.log" % model_log_dir
    extract_f0_feature_log_path = "%s/extract_f0_feature.log" % model_log_dir
    gt_wavs_dir = "%s/0_gt_wavs" % model_log_dir
    feature_dir = ("%s/3_feature256" %
                   model_log_dir if version19 == "v1" else "%s/3_feature768" %
                   model_log_dir)

    os.makedirs(model_log_dir, exist_ok=True)
    #########step1:处理数据
    open(preprocess_log_path, "w").close()
    cmd = (config.python_cmd +
           " trainset_preprocess_pipeline_print.py %s %s %s %s " %
           (trainset_dir4, sr_dict[sr2], np7, model_log_dir) +
           str(config.noparallel))
    yield get_info_str(i18n("step1:正在处理数据"))
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True)
    p.wait()
    with open(preprocess_log_path, "r") as f:
        print(f.read())
    #########step2a:提取音高
    open(extract_f0_feature_log_path, "w")
    if if_f0_3:
        yield get_info_str("step2a:正在提取音高")
        cmd = config.python_cmd + " extract_f0_print.py %s %s %s" % (
            model_log_dir,
            np7,
            f0method8,
        )
        yield get_info_str(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        p.wait()
        with open(extract_f0_feature_log_path, "r") as f:
            print(f.read())
    else:
        yield get_info_str(i18n("step2a:无需提取音高"))
    #######step2b:提取特征
    yield get_info_str(i18n("step2b:正在提取特征"))
    gpus = gpus16.split("-")
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = config.python_cmd + " extract_feature_print.py %s %s %s %s %s %s" % (
            config.device,
            leng,
            idx,
            n_g,
            model_log_dir,
            version19,
        )
        yield get_info_str(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    for p in ps:
        p.wait()
    with open(extract_f0_feature_log_path, "r") as f:
        print(f.read())
    #######step3a:训练模型
    yield get_info_str(i18n("step3a:正在训练模型"))
    # 生成filelist
    if if_f0_3:
        f0_dir = "%s/2a_f0" % model_log_dir
        f0nsf_dir = "%s/2b-f0nsf" % model_log_dir
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)]))
    else:
        names = set([
            name.split(".")[0] for name in os.listdir(gt_wavs_dir)
        ]) & set([name.split(".")[0] for name in os.listdir(feature_dir)])
    opt = []
    for name in names:
        if if_f0_3:
            opt.append("%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s" % (
                gt_wavs_dir.replace("\\", "\\\\"),
                name,
                feature_dir.replace("\\", "\\\\"),
                name,
                f0_dir.replace("\\", "\\\\"),
                name,
                f0nsf_dir.replace("\\", "\\\\"),
                name,
                spk_id5,
            ))
        else:
            opt.append("%s/%s.wav|%s/%s.npy|%s" % (
                gt_wavs_dir.replace("\\", "\\\\"),
                name,
                feature_dir.replace("\\", "\\\\"),
                name,
                spk_id5,
            ))
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5))
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5))
    shuffle(opt)
    with open("%s/filelist.txt" % model_log_dir, "w") as f:
        f.write("\n".join(opt))
    yield get_info_str("write filelist done")
    if gpus16:
        cmd = (
            config.python_cmd +
            " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s -pg %s -pd %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                pretrained_G14,
                pretrained_D15,
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            ))
    else:
        cmd = (
            config.python_cmd +
            " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s -pg %s -pd %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                pretrained_G14,
                pretrained_D15,
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            ))
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    yield get_info_str(i18n("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"))
    #######step3b:训练索引
    npys = []
    listdir_res = list(os.listdir(feature_dir))
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % model_log_dir, big_npy)

    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    yield get_info_str("%s,%s" % (big_npy.shape, n_ivf))
    index = faiss.index_factory(256 if version19 == "v1" else 768,
                                "IVF%s,Flat" % n_ivf)
    yield get_info_str("training index")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s.index" %
        (model_log_dir, n_ivf, index_ivf.nprobe, version19),
    )
    yield get_info_str("adding index")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i:i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s.index" %
        (model_log_dir, n_ivf, index_ivf.nprobe, version19),
    )
    yield get_info_str("成功构建索引, added_IVF%s_Flat_nprobe_%s_%s.index" %
                       (n_ivf, index_ivf.nprobe, version19))
    yield get_info_str(i18n("全流程结束！"))


par = argparse.ArgumentParser()

# preprocess
par.add_argument('dataset_path')
par.add_argument('exp_dir')
par.add_argument('-sr',
                 '--sample_rate',
                 help='可选值32k 40k 48k',
                 type=str,
                 default='48k')
par.add_argument('-np',
                 '--n_p',
                 help='number of process cpus',
                 default=config.n_cpu)

# extract_f0_feature
par.add_argument('-gpus',
                 '--gpus',
                 type=str,
                 help='显卡选择 格式：0-1-2 表示使用卡0和卡1和卡2',
                 default='0')
par.add_argument('-if',
                 '--if_f0',
                 type=bool,
                 help='模型是否带音高指导(唱歌一定要, 语音可以不要)',
                 default=True)
par.add_argument('-v',
                 '--version',
                 type=str,
                 help='版本(目前仅40k支持了v2)',
                 default='v1',
                 choices=['v1', 'v2'])
par.add_argument('-f',
                 '--f0method',
                 type=str,
                 help='选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢',
                 choices=["pm", "harvest", "dio"],
                 default='harvest')

# train
par.add_argument('-s', '--spk_id', type=str, help='speaker id', default='0')
par.add_argument('-se', '--save_epoch', type=int, help='保存频率', default=5)
par.add_argument('-e', '--total_epoch', type=int, help='总训练轮数total_epoch', default=40 )
par.add_argument('-bs',
                 '--batch_size',
                 type=int,
                 help="每张显卡的batch_size",
                 default=default_batch_size)
par.add_argument('-sl',
                 '--if_save_latest',
                 type=bool,
                 help='是否仅保存最新的ckpt文件以节省硬盘空间',
                 default=True)
par.add_argument('-cg',
                 '--if_cache_gpu',
                 type=bool,
                 help='是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速',
                 default=False)
par.add_argument('-sew',
                 '--if_save_every_weights',
                 type=bool,
                 help="是否在每次保存时间点将最终小模型保存至weights文件夹",
                 default=True)
par.add_argument('-pg', '--pretrained_g', type=str, help='加载预训练底模G路径', default="pretrained/f0G48k.pth")
par.add_argument('-pd', '--pretrained_d', type=str, help='加载预训练底模D路径', default="pretrained/f0D48k.pth")

args = par.parse_args()

# preprocess
dataset_path = args.dataset_path
exp_dir = args.exp_dir
sr = args.sample_rate
n_p = args.n_p

# extract_f0_feature
gpus = args.gpus
if_f0 = args.if_f0
version = args.version
f0method = args.f0method

# train

spk_id = args.spk_id
save_epoch = args.save_epoch
total_epoch = args.total_epoch
batch_size = args.batch_size
if_save_latest = args.if_save_latest
if_cache_gpu = args.if_cache_gpu
if_save_every_weights =args.if_save_every_weights
pretrained_G = args.pretrained_g
pretrained_D = args.pretrained_d
preprocess_dataset(dataset_path, exp_dir, sr, n_p)

extract_f0_feature(gpus, n_p,f0method, if_f0,exp_dir, version)

click_train(
    exp_dir,
    sr,
    if_f0,
    spk_id,
    save_epoch,
    total_epoch,
    batch_size,
    if_save_latest,
    pretrained_G,
    pretrained_D,
    gpus,
    if_cache_gpu,
    if_save_every_weights,
    version,
)