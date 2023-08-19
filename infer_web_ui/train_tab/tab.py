import gradio as gr
import numpy as np

from infer_web_ui.train_tab.functions import preprocess_dataset, change_sr2, change_version19, change_f0, \
    extract_f0_feature, click_train, train_index, train1key
from infer_web_ui.utils import check_gpu, i18n, config


def load_train_tab():
    # Check GPU
    gpu_info, default_batch_size, gpus = check_gpu(i18n)

    # Initialize vars
    f0_gpu_visible = not config.dml

    def change_f0_method(method):
        if method == "rmvpe_gpu":
            visible = f0_gpu_visible
        else:
            visible = False
        return {"visible": visible, "__type__": "update"}

    # Create tab
    gr.Markdown(
        value=i18n(
            "step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. "
        )
    )
    with gr.Row():
        exp_dir1 = gr.Textbox(label=i18n("输入实验名"), value="mi-test")
        sr2 = gr.Radio(
            label=i18n("目标采样率"),
            choices=["40k", "48k"],
            value="40k",
            interactive=True,
        )
        if_f0_3 = gr.Radio(
            label=i18n("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
            choices=[True, False],
            value=True,
            interactive=True,
        )
        version19 = gr.Radio(
            label=i18n("版本"),
            choices=["v1", "v2"],
            value="v2",
            interactive=True,
            visible=True,
        )
        np7 = gr.Slider(
            minimum=0,
            maximum=config.n_cpu,
            step=1,
            label=i18n("提取音高和处理数据使用的CPU进程数"),
            value=int(np.ceil(config.n_cpu / 1.5)),
            interactive=True,
        )
    with gr.Group():  # 暂时单人的, 后面支持最多4人的#数据处理
        gr.Markdown(
            value=i18n(
                "step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. "
            )
        )
        with gr.Row():
            trainset_dir4 = gr.Textbox(
                label=i18n("输入训练文件夹路径"), value="E:\\语音音频+标注\\米津玄师\\src"
            )
            spk_id5 = gr.Slider(
                minimum=0,
                maximum=4,
                step=1,
                label=i18n("请指定说话人id"),
                value=0,
                interactive=True,
            )
            but1 = gr.Button(i18n("处理数据"), variant="primary")
            info1 = gr.Textbox(label=i18n("输出信息"), value="")
            but1.click(
                preprocess_dataset,
                [trainset_dir4, exp_dir1, sr2, np7],
                [info1],
                api_name="train_preprocess",
            )
    with gr.Group():
        gr.Markdown(value=i18n("step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)"))
        with gr.Row():
            with gr.Column():
                gpus6 = gr.Textbox(
                    label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                    value=gpus,
                    interactive=True,
                    visible=f0_gpu_visible,
                )
                gpu_info9 = gr.Textbox(
                    label=i18n("显卡信息"), value=gpu_info, visible=f0_gpu_visible
                )
            with gr.Column():
                f0method8 = gr.Radio(
                    label=i18n(
                        "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢,rmvpe效果最好且微吃CPU/GPU"
                    ),
                    choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                    value="rmvpe_gpu",
                    interactive=True,
                )
                gpus_rmvpe = gr.Textbox(
                    label=i18n(
                        "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                    ),
                    value="%s-%s" % (gpus, gpus),
                    interactive=True,
                    visible=f0_gpu_visible,
                )
            but2 = gr.Button(i18n("特征提取"), variant="primary")
            info2 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
            f0method8.change(
                fn=change_f0_method,
                inputs=[f0method8],
                outputs=[gpus_rmvpe],
            )
            but2.click(
                extract_f0_feature,
                [
                    gpus6,
                    np7,
                    f0method8,
                    if_f0_3,
                    exp_dir1,
                    version19,
                    gpus_rmvpe,
                ],
                [info2],
                api_name="train_extract_f0_feature",
            )
    with gr.Group():
        gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
        with gr.Row():
            save_epoch10 = gr.Slider(
                minimum=1,
                maximum=50,
                step=1,
                label=i18n("保存频率save_every_epoch"),
                value=5,
                interactive=True,
            )
            total_epoch11 = gr.Slider(
                minimum=2,
                maximum=1000,
                step=1,
                label=i18n("总训练轮数total_epoch"),
                value=20,
                interactive=True,
            )
            batch_size12 = gr.Slider(
                minimum=1,
                maximum=40,
                step=1,
                label=i18n("每张显卡的batch_size"),
                value=default_batch_size,
                interactive=True,
            )
            if_save_latest13 = gr.Radio(
                label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),
                choices=[i18n("是"), i18n("否")],
                value=i18n("否"),
                interactive=True,
            )
            if_cache_gpu17 = gr.Radio(
                label=i18n(
                    "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                ),
                choices=[i18n("是"), i18n("否")],
                value=i18n("否"),
                interactive=True,
            )
            if_save_every_weights18 = gr.Radio(
                label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                choices=[i18n("是"), i18n("否")],
                value=i18n("否"),
                interactive=True,
            )
        with gr.Row():
            pretrained_g14 = gr.Textbox(
                label=i18n("加载预训练底模G路径"),
                value="pretrained_v2/f0G40k.pth",
                interactive=True,
            )
            pretrained_d15 = gr.Textbox(
                label=i18n("加载预训练底模D路径"),
                value="pretrained_v2/f0D40k.pth",
                interactive=True,
            )
            sr2.change(
                change_sr2,
                [sr2, if_f0_3, version19],
                [pretrained_g14, pretrained_d15],
            )
            version19.change(
                change_version19,
                [sr2, if_f0_3, version19],
                [pretrained_g14, pretrained_d15, sr2],
            )
            if_f0_3.change(
                change_f0,
                [if_f0_3, sr2, version19],
                [f0method8, pretrained_g14, pretrained_d15],
            )
            gpus16 = gr.Textbox(
                label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                value=gpus,
                interactive=True,
            )
            but3 = gr.Button(i18n("训练模型"), variant="primary")
            but4 = gr.Button(i18n("训练特征索引"), variant="primary")
            but5 = gr.Button(i18n("一键训练"), variant="primary")
            info3 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=10)
            but3.click(
                click_train,
                [
                    exp_dir1,
                    sr2,
                    if_f0_3,
                    spk_id5,
                    save_epoch10,
                    total_epoch11,
                    batch_size12,
                    if_save_latest13,
                    pretrained_g14,
                    pretrained_d15,
                    gpus16,
                    if_cache_gpu17,
                    if_save_every_weights18,
                    version19,
                ],
                info3,
                api_name="train_start",
            )
            but4.click(train_index, [exp_dir1, version19], info3)
            but5.click(
                train1key,
                [
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
                    pretrained_g14,
                    pretrained_d15,
                    gpus16,
                    if_cache_gpu17,
                    if_save_every_weights18,
                    version19,
                    gpus_rmvpe,
                ],
                info3,
                api_name="train_start_all",
            )
