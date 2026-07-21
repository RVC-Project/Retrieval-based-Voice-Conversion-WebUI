import numpy as np
import torch
import torch.nn.functional as F
from tools.cuda_graph import clear_cuda_graph_cache, run_cuda_graph
from tqdm import tqdm


def make_padding(width, cropsize, offset):
    left = offset
    roi_size = cropsize - left * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


def _execute_torch_windows(
    X_mag_pad,
    roi_size,
    n_window,
    device,
    model,
    aggressiveness,
    data,
    batch_size,
):
    windows = X_mag_pad.unfold(
        2,
        data["window_size"],
        roi_size,
    )[:, :, :n_window, :]
    model_dtype = next(model.parameters()).dtype
    predictions = None
    write_offset = 0
    with torch.inference_mode():
        for start in tqdm(range(0, n_window, batch_size)):
            end = min(start + batch_size, n_window)
            batch = (
                windows[:, :, start:end, :]
                .permute(2, 0, 1, 3)
                .contiguous()
                .to(device=device, dtype=model_dtype)
            )
            prediction = run_cuda_graph(
                model,
                "uvr-vr-%s" % repr(aggressiveness),
                lambda window: model.predict(window, aggressiveness),
                batch,
            )
            prediction = prediction.float().permute(1, 2, 0, 3).reshape(
                prediction.shape[1], prediction.shape[2], -1
            )
            if predictions is None:
                predictions = torch.empty(
                    prediction.shape[0],
                    prediction.shape[1],
                    n_window * roi_size,
                    device=prediction.device,
                    dtype=torch.float32,
                )
            end_offset = write_offset + prediction.shape[2]
            predictions[:, :, write_offset:end_offset].copy_(prediction)
            write_offset = end_offset
    return predictions[:, :, :write_offset]


def _torch_batch_size(device):
    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_gb = free_bytes / (1024**3)
    if free_gb > 20:
        return 8
    if free_gb > 12:
        return 4
    if free_gb > 8:
        return 2
    return 1


def _inference_torch(X_spec, device, model, aggressiveness, data):
    X_spec = X_spec.to(device)
    X_mag = torch.abs(X_spec)
    coef = X_mag.max().clamp_min(1e-8)
    X_mag_pre = X_mag / coef
    n_frame = X_mag_pre.shape[2]
    pad_l, pad_r, roi_size = make_padding(
        n_frame, data["window_size"], model.offset
    )
    n_window = int(np.ceil(n_frame / roi_size))

    def execute(pad_left, pad_right, windows_count):
        padded = F.pad(X_mag_pre, (pad_left, pad_right))
        batch_size = _torch_batch_size(device)
        while True:
            try:
                return _execute_torch_windows(
                    padded,
                    roi_size,
                    windows_count,
                    device,
                    model,
                    aggressiveness,
                    data,
                    batch_size,
                )
            except torch.cuda.OutOfMemoryError:
                clear_cuda_graph_cache(model)
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)

    pred = execute(pad_l, pad_r, n_window)[:, :, :n_frame]
    if data["tta"]:
        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1
        pred_tta = execute(pad_l, pad_r, n_window)
        pred_tta = pred_tta[:, :, roi_size // 2 :][:, :, :n_frame]
        pred = (pred + pred_tta) * 0.5
    return pred * coef, X_mag, None


def inference(X_spec, device, model, aggressiveness, data):
    """
    data : dic configs
    """

    if torch.is_tensor(X_spec) and X_spec.device.type == "cuda":
        return _inference_torch(X_spec, device, model, aggressiveness, data)

    def _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half=True):
        model.eval()
        with torch.no_grad():
            preds = []

            iterations = [n_window]

            total_iterations = sum(iterations)
            for i in tqdm(range(n_window)):
                start = i * roi_size
                X_mag_window = X_mag_pad[None, :, :, start : start + data["window_size"]]
                X_mag_window = torch.from_numpy(X_mag_window)
                if is_half:
                    X_mag_window = X_mag_window.half()
                X_mag_window = X_mag_window.to(device)

                pred = run_cuda_graph(
                    model,
                    "uvr-vr-%s" % repr(aggressiveness),
                    lambda window: model.predict(window, aggressiveness),
                    X_mag_window,
                )

                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])

            pred = np.concatenate(preds, axis=2)
        return pred

    def preprocess(X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    X_mag, X_phase = preprocess(X_spec)

    coef = X_mag.max()
    X_mag_pre = X_mag / coef

    n_frame = X_mag_pre.shape[2]
    pad_l, pad_r, roi_size = make_padding(n_frame, data["window_size"], model.offset)
    n_window = int(np.ceil(n_frame / roi_size))

    X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

    if list(model.state_dict().values())[0].dtype == torch.float16:
        is_half = True
    else:
        is_half = False
    pred = _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half)
    pred = pred[:, :, :n_frame]

    if data["tta"]:
        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

        pred_tta = _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half)
        pred_tta = pred_tta[:, :, roi_size // 2 :]
        pred_tta = pred_tta[:, :, :n_frame]

        return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.0j * X_phase)
    else:
        return pred * coef, X_mag, np.exp(1.0j * X_phase)
