

import torch
# from infer.lib.rmvpe import RMVPE

from tools.jit_export import benchmark, jit_warm_up, to_jit_model
# RMVPE
# get_rmvpe()


if __name__ =="__main__":
    model_path ="C:\\Users\\14404\\Project\\Retrieval-based-Voice-Conversion-WebUI\\assets\\weights\\kikiV1.pth"
    epoch = 4000
    is_half = False
    device=torch.device("cuda")


    # print(f"Process-> rmvpe")
    # model,model_jit=to_jit_model("assets/rmvpe/rmvpe.pt",
    #                              "rmvpe","assets/rmvpe_inputs.pth",
    #                              device,is_half)
    # print("Benchmark-> rmvpe")
    # benchmark(model,"assets/rmvpe_inputs.pth",device,epoch,is_half)
    # print("Warm up-> rmvpe.jit")
    # jit_warm_up(model_jit,"assets/rmvpe_inputs.pth",device,5,is_half)
    # print("Benchmark-> rmvpe.jit")
    # benchmark(model_jit,"assets/rmvpe_inputs.pth",device,epoch,is_half)
    # print("Save-> rmvpe.jit")
    # torch.jit.save(model_jit.cpu(),"assets/rmvpe/rmvpe.jit")

    print(f"Process-> synthesizer")
    model,model_jit=to_jit_model("assets/weights/kikiV1.pth",
                                "synthesizer","assets/Synthesizer_inputs.pth",
                                device,is_half)
    print("Benchmark-> synthesizer")
    benchmark(model,"assets/Synthesizer_inputs.pth",device,epoch,is_half)
    print("Warm up-> synthesizer.jit")
    jit_warm_up(model_jit,"assets/Synthesizer_inputs.pth",device,5,is_half)
    print("Benchmark-> synthesizer.jit")
    benchmark(model_jit,"assets/Synthesizer_inputs.pth",device,epoch,is_half)
    # print("Save-> synthesizer.jit")
    # torch.jit.save(model_jit.cpu(),"assets/weights/kikiV1.jit")

    # print(f"Process-> hubert")
    # model,model_jit=to_jit_model("assets/hubert/hubert_base.pt",
    #                             "hubert","assets/hubert_inputs.pth",
    #                             device,is_half)
    # print("Benchmark-> hubert")
    # benchmark(model,"assets/hubert_inputs.pth",device,epoch,is_half)
    # print("Warm up-> hubert.jit")
    # jit_warm_up(model_jit,"assets/hubert_inputs.pth",device,5,is_half)
    # print("Benchmark-> hubert.jit")
    # benchmark(model_jit,"assets/hubert_inputs.pth",device,epoch,is_half)
    # print("Save-> hubert.jit")
    # torch.jit.save(model_jit.cpu(),"assets/hubert/hubert_base.jit.pt")