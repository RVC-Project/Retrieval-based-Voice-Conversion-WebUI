from io import BytesIO
import pickle
import time
import torch
from tqdm import tqdm


def load_inputs(path,device,is_half=False):
    parm=torch.load(path,map_location=torch.device("cpu"))
    for key in parm.keys():
        parm[key] = parm[key].to(device)
        if is_half and parm[key].dtype == torch.float32:
            parm[key] = parm[key].half()  
        elif not is_half and parm[key].dtype == torch.float16:
             parm[key] = parm[key].float()
    return parm

def benchmark(model,inputs_path,device=torch.device("cpu"),epoch=1000,is_half=False):
    parm=load_inputs(inputs_path,device,is_half)
    total_ts = 0.0
    bar=tqdm(range(epoch))
    for i in bar:
        start_time=time.perf_counter()
        o=model(**parm)
        total_ts+=time.perf_counter()-start_time
    print(f"num_epoch: {epoch} | avg time(ms): {(total_ts*1000)/epoch}")

def jit_warm_up(model,inputs_path,device=torch.device("cpu"),epoch=5,is_half=False):
    benchmark(model,inputs_path,device,epoch=epoch,is_half=is_half)

def to_jit_model(model_path,model_type:str,inputs_path:str,device=torch.device("cpu"),is_half=False):
    model=None
    if model_type.lower()=="synthesizer":
        from tools.jit_export.get_synthesizer import get_synthesizer
        model,_=get_synthesizer(model_path,device)
        model.forward = model.infer
    elif model_type.lower()=="rmvpe":
        from tools.jit_export.get_rmvpe import get_rmvpe
        model=get_rmvpe(model_path,device)
    elif model_type.lower()=="hubert":
        from tools.jit_export.get_hubert import get_hubert_model
        model=get_hubert_model(model_path,device)
        model.forward = model.infer
    else:
        raise ValueError(f"No model type named {model_type}")
    model = model.eval()
    model = model.half() if is_half else model.float()
    inputs = load_inputs(inputs_path,device,is_half)
    model_jit=torch.jit.trace(model,example_kwarg_inputs=inputs)
    model_jit.to(device)
    model_jit = model_jit.half() if is_half else model_jit.float()
    # model = model.half() if is_half else model.float()
    return (model,model_jit)


def synthesizer_jit_export(model_path:str,inputs_path:str,save_path:str=None,device=torch.device("cpu"),is_half=False):
    if not save_path:
        save_path=model_path.rstrip(".pth")
        save_path+=".half.jit" if is_half else ".jit"
    from tools.jit_export.get_synthesizer import get_synthesizer
    model,cpt=get_synthesizer(model_path,device)
    model = model.half() if is_half else model.float()
    model.eval()
    model.forward = model.infer
    inputs =load_inputs(inputs_path,device,is_half)
    model_jit=torch.jit.trace(model,example_kwarg_inputs=inputs)
    model_jit.to(device)
    model_jit = model_jit.half() if is_half else model_jit.float()
    model_jit.infer = model_jit.forward
    buffer = BytesIO()
    model_jit=model_jit.cpu()
    torch.jit.save(model_jit,buffer)
    assert isinstance(cpt,dict)
    cpt.pop("weight")
    cpt["model"] = buffer.getvalue()
    with open(save_path,"wb") as f:   
        pickle.dump(cpt,f)
    del model_jit
    return cpt
