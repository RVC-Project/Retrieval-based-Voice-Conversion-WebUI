import os
from pathlib import Path
import requests
import time

RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"

BASE_DIR = Path(__file__).resolve().parent.parent


def dl_model(link, model_name, dir_name, max_retries=5):
    """下载模型文件，支持自动重试"""
    file_path = dir_name / model_name

    # 如果文件已存在，跳过下载
    if file_path.exists():
        print(f"✓ {model_name} 已存在，跳过下载")
        return True

    # 创建目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 尝试下载，支持重试
    for attempt in range(max_retries):
        try:
            print(f"正在下载 {model_name}... (尝试 {attempt + 1}/{max_retries})")

            # 设置超时时间
            with requests.get(f"{link}{model_name}", timeout=300, stream=True) as r:
                r.raise_for_status()

                # 获取文件大小
                total_size = int(r.headers.get('content-length', 0))
                downloaded_size = 0

                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)

                            # 显示进度
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                print(f"\r  进度: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end="")

                print(f"\n✓ {model_name} 下载成功！")
                return True

        except Exception as e:
            print(f"\n✗ 下载失败: {str(e)}")

            # 如果文件部分下载，删除它
            if file_path.exists():
                file_path.unlink()

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 指数退避：2秒、4秒、6秒...
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"✗ {model_name} 下载失败，已达到最大重试次数")
                return False

    return False


if __name__ == "__main__":
    print("=" * 60)
    print("RVC 预训练模型下载工具（支持断点续传）")
    print("=" * 60)
    print()

    failed_downloads = []

    # 1. 下载 hubert_base.pt
    print("\n[1/5] 下载 hubert_base.pt...")
    if not dl_model(RVC_DOWNLOAD_LINK, "hubert_base.pt", BASE_DIR / "assets/hubert"):
        failed_downloads.append("hubert_base.pt")

    # 2. 下载 rmvpe.pt
    print("\n[2/5] 下载 rmvpe.pt...")
    if not dl_model(RVC_DOWNLOAD_LINK, "rmvpe.pt", BASE_DIR / "assets/rmvpe"):
        failed_downloads.append("rmvpe.pt")

    # 3. 下载 vocals.onnx
    print("\n[3/5] 下载 vocals.onnx...")
    if not dl_model(
        RVC_DOWNLOAD_LINK + "uvr5_weights/onnx_dereverb_By_FoxJoy/",
        "vocals.onnx",
        BASE_DIR / "assets/uvr5_weights/onnx_dereverb_By_FoxJoy",
    ):
        failed_downloads.append("vocals.onnx")

    # 4. 下载 pretrained 模型
    print("\n[4/5] 下载 pretrained 模型...")
    rvc_models_dir = BASE_DIR / "assets/pretrained"

    model_names = [
        "D32k.pth",
        "D40k.pth",
        "D48k.pth",
        "G32k.pth",
        "G40k.pth",
        "G48k.pth",
        "f0D32k.pth",
        "f0D40k.pth",
        "f0D48k.pth",
        "f0G32k.pth",
        "f0G40k.pth",
        "f0G48k.pth",
    ]

    for i, model in enumerate(model_names, 1):
        print(f"\n  [{i}/{len(model_names)}] {model}")
        if not dl_model(RVC_DOWNLOAD_LINK + "pretrained/", model, rvc_models_dir):
            failed_downloads.append(f"pretrained/{model}")

    # 5. 下载 pretrained_v2 模型
    print("\n[5/5] 下载 pretrained_v2 模型...")
    rvc_models_dir = BASE_DIR / "assets/pretrained_v2"

    for i, model in enumerate(model_names, 1):
        print(f"\n  [{i}/{len(model_names)}] {model}")
        if not dl_model(RVC_DOWNLOAD_LINK + "pretrained_v2/", model, rvc_models_dir):
            failed_downloads.append(f"pretrained_v2/{model}")

    # 6. 下载 uvr5_weights
    print("\n[6/6] 下载 uvr5_weights...")
    rvc_models_dir = BASE_DIR / "assets/uvr5_weights"

    uvr5_models = [
        "HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth",
        "HP2_all_vocals.pth",
        "HP3_all_vocals.pth",
        "HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoNormal.pth",
    ]

    for i, model in enumerate(uvr5_models, 1):
        print(f"\n  [{i}/{len(uvr5_models)}] {model}")
        if not dl_model(RVC_DOWNLOAD_LINK + "uvr5_weights/", model, rvc_models_dir):
            failed_downloads.append(f"uvr5_weights/{model}")

    # 总结
    print("\n" + "=" * 60)
    if failed_downloads:
        print("⚠ 下载完成，但有部分文件失败：")
        for file in failed_downloads:
            print(f"  ✗ {file}")
        print("\n请重新运行此脚本继续下载失败的文件")
    else:
        print("✓ 所有模型下载完成！")
    print("=" * 60)
