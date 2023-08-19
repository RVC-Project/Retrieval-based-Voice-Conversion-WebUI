import gradio as gr

from infer_web_ui.gradio_tools import clean
from infer_web_ui.inference_tab.functions import change_choices, vc_single, vc_multi, get_vc
from infer_web_ui.utils import i18n, names, config, index_paths


def load_inference_tab():
    with gr.Row():
        sid0 = gr.Dropdown(label=i18n("推理音色"), choices=sorted(names))
        refresh_button = gr.Button(i18n("刷新音色列表和索引路径"), variant="primary")
        clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
        spk_item = gr.Slider(
            minimum=0,
            maximum=2333,
            step=1,
            label=i18n("请选择说话人id"),
            value=0,
            visible=False,
            interactive=True,
        )
        clean_button.click(
            fn=clean, inputs=[], outputs=[sid0], api_name="infer_clean"
        )
    with gr.Group():
        gr.Markdown(
            value=i18n("男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. ")
        )
        with gr.Row():
            with gr.Column():
                vc_transform0 = gr.Number(
                    label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
                )
                input_audio0 = gr.Textbox(
                    label=i18n("输入待处理音频文件路径(默认是正确格式示例)"),
                    value="E:\\codes\\py39\\test-20230416b\\todo-songs\\clip1.wav",
                )
                f0method0 = gr.Radio(
                    label=i18n(
                        "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                    ),
                    choices=["pm", "harvest", "crepe", "rmvpe"]
                    if config.dml == False
                    else ["pm", "harvest", "rmvpe"],
                    value="pm",
                    interactive=True,
                )
                filter_radius0 = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                    value=3,
                    step=1,
                    interactive=True,
                )
            with gr.Column():
                file_index1 = gr.Textbox(
                    label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                    value="",
                    interactive=True,
                )
                file_index2 = gr.Dropdown(
                    label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                    choices=sorted(index_paths),
                    interactive=True,
                )
                refresh_button.click(
                    fn=change_choices,
                    inputs=[],
                    outputs=[sid0, file_index2],
                    api_name="infer_refresh",
                )
                index_rate1 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("检索特征占比"),
                    value=0.75,
                    interactive=True,
                )
            with gr.Column():
                resample_sr0 = gr.Slider(
                    minimum=0,
                    maximum=48000,
                    label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                    value=0,
                    step=1,
                    interactive=True,
                )
                rms_mix_rate0 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                    value=0.25,
                    interactive=True,
                )
                protect0 = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n(
                        "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                    ),
                    value=0.33,
                    step=0.01,
                    interactive=True,
                )
            f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"))
            but0 = gr.Button(i18n("转换"), variant="primary")
            with gr.Row():
                vc_output1 = gr.Textbox(label=i18n("输出信息"))
                vc_output2 = gr.Audio(label=i18n("输出音频(右下角三个点,点了可以下载)"))
            but0.click(
                vc_single,
                [
                    spk_item,
                    input_audio0,
                    vc_transform0,
                    f0_file,
                    f0method0,
                    file_index1,
                    file_index2,
                    # file_big_npy1,
                    index_rate1,
                    filter_radius0,
                    resample_sr0,
                    rms_mix_rate0,
                    protect0,
                ],
                [vc_output1, vc_output2],
                api_name="infer_convert",
            )
    with gr.Group():
        gr.Markdown(
            value=i18n(
                "批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. ")
        )
        with gr.Row():
            with gr.Column():
                vc_transform1 = gr.Number(
                    label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
                )
                opt_input = gr.Textbox(label=i18n("指定输出文件夹"), value="opt")
                f0method1 = gr.Radio(
                    label=i18n(
                        "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                    ),
                    choices=["pm", "harvest", "crepe", "rmvpe"]
                    if not config.dml
                    else ["pm", "harvest", "rmvpe"],
                    value="pm",
                    interactive=True,
                )
                filter_radius1 = gr.Slider(
                    minimum=0,
                    maximum=7,
                    label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                    value=3,
                    step=1,
                    interactive=True,
                )
            with gr.Column():
                file_index3 = gr.Textbox(
                    label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                    value="",
                    interactive=True,
                )
                file_index4 = gr.Dropdown(
                    label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                    choices=sorted(index_paths),
                    interactive=True,
                )
                refresh_button.click(
                    fn=lambda: change_choices()[1],
                    inputs=[],
                    outputs=file_index4,
                    api_name="infer_refresh_batch",
                )
                index_rate2 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("检索特征占比"),
                    value=1,
                    interactive=True,
                )
            with gr.Column():
                resample_sr1 = gr.Slider(
                    minimum=0,
                    maximum=48000,
                    label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                    value=0,
                    step=1,
                    interactive=True,
                )
                rms_mix_rate1 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                    value=1,
                    interactive=True,
                )
                protect1 = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n(
                        "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                    ),
                    value=0.33,
                    step=0.01,
                    interactive=True,
                )
            with gr.Column():
                dir_input = gr.Textbox(
                    label=i18n("输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"),
                    value=r"E:\codes\py39\\test-20230416b\\todo-songs",
                )
                inputs = gr.File(
                    file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
                )
            with gr.Row():
                format1 = gr.Radio(
                    label=i18n("导出文件格式"),
                    choices=["wav", "flac", "mp3", "m4a"],
                    value="flac",
                    interactive=True,
                )
                but1 = gr.Button(i18n("转换"), variant="primary")
                vc_output3 = gr.Textbox(label=i18n("输出信息"))
            but1.click(
                vc_multi,
                [
                    spk_item,
                    dir_input,
                    opt_input,
                    inputs,
                    vc_transform1,
                    f0method1,
                    file_index3,
                    file_index4,
                    # file_big_npy2,
                    index_rate2,
                    filter_radius1,
                    resample_sr1,
                    rms_mix_rate1,
                    protect1,
                    format1,
                ],
                [vc_output3],
                api_name="infer_convert_batch",
            )
    sid0.change(
        fn=get_vc,
        inputs=[sid0, protect0, protect1],
        outputs=[spk_item, protect0, protect1, file_index2, file_index4],
    )
