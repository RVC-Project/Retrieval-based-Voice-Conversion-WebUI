"""Shared sounddevice device-query helpers used by gui_v1 and the realtime engine."""

import sounddevice as sd

from tools.realtime.dsp import printt


def query_devices(hostapi_name=None):
    """Re-scan audio devices.

    Returns (hostapis, input_devices, output_devices,
    input_devices_indices, output_devices_indices) for the given hostapi
    (falls back to the first hostapi if the name is unknown).
    """
    sd._terminate()
    sd._initialize()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for hostapi in hostapis:
        for device_idx in hostapi["devices"]:
            devices[device_idx]["hostapi_name"] = hostapi["name"]
    hostapi_names = [hostapi["name"] for hostapi in hostapis]
    if hostapi_name not in hostapi_names:
        hostapi_name = hostapi_names[0]
    input_devices = [
        d["name"]
        for d in devices
        if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]
    output_devices = [
        d["name"]
        for d in devices
        if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]
    input_devices_indices = [
        d["index"] if "index" in d else d["name"]
        for d in devices
        if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]
    output_devices_indices = [
        d["index"] if "index" in d else d["name"]
        for d in devices
        if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
    ]
    return (
        hostapi_names,
        input_devices,
        output_devices,
        input_devices_indices,
        output_devices_indices,
    )


def set_devices(
    input_devices,
    input_devices_indices,
    output_devices,
    output_devices_indices,
    input_device,
    output_device,
):
    sd.default.device[0] = input_devices_indices[input_devices.index(input_device)]
    sd.default.device[1] = output_devices_indices[output_devices.index(output_device)]
    printt("Input device: %s:%s", str(sd.default.device[0]), input_device)
    printt("Output device: %s:%s", str(sd.default.device[1]), output_device)


def get_device_samplerate():
    return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])


def get_device_channels():
    max_input_channels = sd.query_devices(device=sd.default.device[0])[
        "max_input_channels"
    ]
    max_output_channels = sd.query_devices(device=sd.default.device[1])[
        "max_output_channels"
    ]
    return min(max_input_channels, max_output_channels, 2)
