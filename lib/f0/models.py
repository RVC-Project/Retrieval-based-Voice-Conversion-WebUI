import torch


def get_rmvpe(
    model_path="assets/rmvpe/rmvpe.pt", device=torch.device("cpu"), is_half=False
):
    from lib.f0.e2e import E2E

    model = E2E(4, 1, (2, 2))
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    del ckpt
    model.eval()
    if is_half:
        model = model.half()
    model = model.to(device)
    return model
