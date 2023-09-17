import os

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    index_paths= [
            os.path.join(root, name)
            for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
            for name in files
            if name.endswith(".index") and "trained" not in name
        ]
    
    def multisplit(str, sp=".-_/\\"):
        ret = []
        cur = ""
        for i in str:
            if i in sp:
                ret.append(cur.strip())
                cur = ""
            else:
                cur += i
        ret.append(cur.strip())
        return ret

    sel_index_path = ""
    # name = os.path.join("logs", sid.split(".")[0], "")
    names=multisplit(sid.split(".")[0])
    # print(name)
    mx=0
    for f in index_paths:
        fs=multisplit(f)
        fs=list(filter(lambda x:x not in ['logs','1','Flat','nprobe','added','index'],fs))
        cnt=0
        for i in names:
            if i in fs:
                cnt+=1
        if cnt>=2 and cnt>mx:
            # print("selected index path:", f)
            sel_index_path = f
            mx=cnt
    return sel_index_path


def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
