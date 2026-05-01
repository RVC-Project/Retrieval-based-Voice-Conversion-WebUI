import logging
from pathlib import Path
import time
from typing import List, Optional, Tuple
import gradio as gr
import numpy as np

import shared
from lib.f0 import PITCH_METHODS, PitchMethod
from shared import i18n


def clean():
    return {"value": "", "__type__": "update"}


def change_choices():
    names = []
    for entry in shared.weight_root.iterdir():
        if entry.suffix == ".pth":
            names.append(entry.name)
    index_paths = [""]
    for index_file in shared.index_root.rglob("*.index"):
        if "trained" not in index_file.name:
            index_paths.append(str(index_file))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def get_pitch_methods() -> List[PitchMethod]:
    return PITCH_METHODS


def get_model_list() -> List[str]:
    print(f"Models: {shared.names}")
    return sorted(shared.names)


def get_index_paths() -> List[str]:
    return sorted(shared.index_paths)


def create_inference_tab(app: gr.Blocks):

    with gr.TabItem(i18n("Inference")):
        gr.api(
            get_model_list,
            api_name="get_model_list",
        )
        gr.api(get_pitch_methods, api_name="get_pitch_methods")
        gr.api(get_index_paths, api_name="get_index_paths")
        with gr.Row():
            with gr.Column():
                model_list = sorted(shared.names)
                if not model_list:
                    # If no models are found, display a Textbox with a message
                    gr.Textbox(
                        label=i18n("Model"),
                        value=i18n("No models found."),
                        interactive=False,
                        visible=True,
                    )
                    model_dropdown = gr.Dropdown(
                        label=i18n("Model"), choices=[], visible=False
                    )
                else:
                    # If models are found, display the Dropdown
                    model_dropdown = gr.Dropdown(
                        label=i18n("Model"), choices=model_list, visible=True
                    )

                with gr.Column():
                    refresh_btn = gr.Button(i18n("Refresh"), variant="primary")
                with gr.TabItem(i18n("Basic")):
                    audio_input = gr.Audio(
                        label=i18n("Input Audio"),
                        type="numpy",
                    )
                    convert_btn = gr.Button(i18n("Convert"), variant="primary")
                    autoplay_checkbox = gr.Checkbox(label=i18n("Autoplay"), value=False)

                    vc_file_output = gr.Audio(
                        label=i18n("Output Audio"),
                    )

                    def set_autoplay(x: bool):
                        print(f"Set auto play: {x}")
                        return {"autoplay": x, "__type__": "update"}

                    autoplay_checkbox.input(
                        set_autoplay,
                        [autoplay_checkbox],
                        [vc_file_output],
                    )

                with gr.TabItem(i18n("Real Time (WIP)")):
                    import sounddevice as sd

                    lag_backlog = 0.0

                    def realtime_vc_generator(
                        audio_chunk: Optional[Tuple[int, np.ndarray]],
                        f0_up_key,
                        f0_method,
                        file_index,
                        index_rate,
                        resample_sr,
                        rms_mix_rate,
                        protect,
                    ):
                        nonlocal lag_backlog

                        if audio_chunk is None:
                            return None

                        # Gradio streams (sr, np.ndarray)
                        if isinstance(audio_chunk, tuple):
                            sr, audio_data = audio_chunk
                        else:
                            # Some versions just pass the raw np.ndarray
                            sr, audio_data = 16000, audio_chunk

                            # Calculate duration of incoming chunk
                        chunk_duration = len(audio_data) / sr

                        # If backlog is already too large, skip without even processing
                        if lag_backlog > chunk_duration:
                            print(
                                f"⏭️ Skipping chunk ({chunk_duration:.4f}s) | "
                                f"Backlog: {lag_backlog:.3f}s"
                            )
                            lag_backlog -= chunk_duration  # reduce backlog
                            if lag_backlog < 0:
                                lag_backlog = 0
                            return None

                        start_time = time.time()

                        # Process audio
                        result = shared.vc.vc_realtime(
                            (sr, audio_data),
                            f0_up_key=f0_up_key,
                            f0_method=f0_method,
                            file_index=file_index if file_index else None,
                            index_rate=index_rate,
                            resample_sr=resample_sr,
                            rms_mix_rate=rms_mix_rate,
                            protect=protect,
                        )

                        end_time = time.time()
                        processing_time = end_time - start_time

                        # Compare and print results
                        print(
                            f"⏱️ Chunk Duration: {chunk_duration:.4f}s | "
                            f"Processing Time: {processing_time:.4f}s | "
                            f"Backlog: {lag_backlog:.3f}s"
                        )

                        if processing_time > chunk_duration:
                            lag_ms = (processing_time - chunk_duration) * 1000
                            lag_backlog += processing_time - chunk_duration
                            print(
                                f"⚠️ Behind real-time! Added lag: {lag_ms:.2f} ms "
                                f"(Total backlog: {lag_backlog:.3f}s)"
                            )
                        else:
                            # If we are faster than real time, reduce backlog
                            lag_backlog -= chunk_duration - processing_time
                            if lag_backlog < 0:
                                lag_backlog = 0
                            print("✅ Keeping up!")

                        if result is None:
                            return None

                        tgt_sr, audio_out = result
                        # Play directly on server
                        sd.play(audio_out, samplerate=tgt_sr)
                        logging.info("outputting")

                    audio_stream_in = gr.Audio(
                        label="Mic Input",
                        type="numpy",
                        streaming=True,
                        sources=["microphone"],
                    )

                    # Define UI controls for params instead of passing None/values directly
                    pitch_offset_rt = gr.Number(value=0, label="Pitch Offset")
                    f0method_rt = gr.Dropdown(
                        ["rmvpe", "crepe"], value="rmvpe", label="F0 Method"
                    )
                    file_index_rt = gr.Textbox(value="", label="Index File (optional)")
                    index_rate_rt = gr.Slider(
                        0, 1, value=0.5, step=0.01, label="Index Rate"
                    )
                    resample_sr_rt = gr.Number(value=16000, label="Resample SR")
                    rms_mix_rate_rt = gr.Slider(
                        0, 1, value=1.0, step=0.01, label="RMS Mix Rate"
                    )
                    protect_rt = gr.Slider(0, 1, value=0.5, step=0.01, label="Protect")

                    audio_stream_in.stream(
                        fn=realtime_vc_generator,
                        inputs=[
                            audio_stream_in,
                            pitch_offset_rt,
                            f0method_rt,
                            file_index_rt,
                            index_rate_rt,
                            resample_sr_rt,
                            rms_mix_rate_rt,
                            protect_rt,
                        ],
                        outputs=[],
                        api_name="infer_realtime",
                    )

            with gr.Column():
                pitch_offset = gr.Slider(
                    label="Pitch Offset",
                    minimum=-24,
                    maximum=24,
                    step=1,
                    value=0,
                )
                resample_sr0 = gr.Slider(
                    minimum=0,
                    maximum=48000,
                    label=i18n("Resample Rate (Skip if it is 0)"),
                    value=0,
                    step=1,
                    interactive=True,
                )
                rms_mix_rate0 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n(
                        # "Fusion ratio of replacing input source volume envelope with output volume envelope, closer to 1 uses output envelope more"
                        "RMS Mix Rate"
                    ),
                    value=0.25,
                    interactive=True,
                )
                protect0 = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n(
                        # "Protect voiceless consonants and breath sounds, preventing artifacts like tearing of electronic music. Maxing out to 0.5 turns it off, lowering it increases protection but might reduce the index effect"
                        "Protect 0 (Reduce Artifact)"
                    ),
                    value=0.33,
                    step=0.01,
                    interactive=True,
                )
                index_rate1 = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Search feature ratio"),
                    value=0.75,
                    interactive=True,
                )
            with gr.Column():
                file_index2 = gr.Dropdown(
                    label=i18n("Index File"),
                    choices=sorted(shared.index_paths),
                    interactive=True,
                    allow_custom_value=True,
                    value="",
                )
                f0method0 = gr.Radio(
                    label=i18n("Pitch Method"),
                    choices=get_pitch_methods(),
                    value="rmvpe",
                    interactive=True,
                )
                vc_log_output = gr.Textbox(label=i18n("Log info"))

        convert_btn.click(
            shared.vc.vc_single,
            [
                audio_input,
                pitch_offset,
                f0method0,
                file_index2,
                index_rate1,
                resample_sr0,
                rms_mix_rate0,
                protect0,
            ],
            [vc_log_output, vc_file_output],
            api_name="infer_convert",
        )
        refresh_btn.click(
            fn=change_choices,
            inputs=[],
            outputs=[model_dropdown, file_index2],
            api_name="infer_refresh",
        )
        model_dropdown.change(
            fn=shared.vc.get_vc,
            inputs=[
                model_dropdown,
                protect0,
            ],  # Use protect0 and protect1 from Basic/Batch tab
            outputs=[protect0, file_index2],
            api_name="infer_change_voice",
        )
        app.load(
            fn=shared.vc.get_vc,
            inputs=[
                model_dropdown,
                protect0,
            ],  # Use the components themselves to get their initial values
            outputs=[protect0, file_index2],
        )
