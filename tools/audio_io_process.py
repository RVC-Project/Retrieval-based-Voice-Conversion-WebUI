import numpy as np
from multiprocessing import Process, Value, Event
from multiprocessing.shared_memory import SharedMemory
import sounddevice as sd
import signal


class AudioIoProcess(Process):
    def __init__(self,
                 input_device,
                 output_device,
                 input_audio_block_size: int,
                 sample_rate: int,
                 channel_num: int = 2,
                 is_device_combined: bool = True,
                 is_input_wasapi_exclusive: bool = False,
                 is_output_wasapi_exclusive: bool = False
                 ):
        super().__init__()
        self.in_dev = input_device
        self.out_dev = output_device
        self.block_size: int = input_audio_block_size
        self.buf_size: int = self.block_size << 1  # 双缓冲
        self.sample_rate: int = sample_rate
        self.channels: int = channel_num
        self.is_device_combined: bool = is_device_combined
        self.is_input_wasapi_exclusive: bool = is_input_wasapi_exclusive
        self.is_output_wasapi_exclusive: bool = is_output_wasapi_exclusive

        self.__rec_ptr = 0
        self.in_ptr = Value('i', 0)  # 当收满一个block时由本进程设置
        self.out_ptr = Value('i', 0)  # 由主进程设置，指示下一次预期写入位置
        self.play_ptr = Value('i', 0)  # 由本进程设置，指示当前音频已经播放到哪里
        self.in_evt = Event()  # 当收满一个block时由本进程设置
        self.stop_evt = Event()  # 当主进程停止音频活动时由主进程设置

        self.latency = Value('d', 114514.1919810)

        self.buf_shape: tuple = (self.buf_size, self.channels)
        self.buf_dtype: np.dtype = np.float32
        self.buf_nbytes: int = int(
            np.prod(self.buf_shape) * np.dtype(self.buf_dtype).itemsize)

        self.in_mem = SharedMemory(create=True, size=self.buf_nbytes)
        self.out_mem = SharedMemory(create=True, size=self.buf_nbytes)
        self.in_mem_name: str = self.in_mem.name
        self.out_mem_name: str = self.out_mem.name

        self.in_buf = None
        self.out_buf = None

    def get_in_mem_name(self) -> str:
        return self.in_mem_name

    def get_out_mem_name(self) -> str:
        return self.out_mem_name

    def get_np_shape(self) -> tuple:
        return self.buf_shape

    def get_np_dtype(self) -> np.dtype:
        return self.buf_dtype

    def get_ptrs_and_events(self):
        return self.in_ptr, \
            self.out_ptr,\
            self.play_ptr,\
            self.in_evt, \
            self.stop_evt\

    def get_latency(self) -> float:
        return self.latency.value

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        in_mem = SharedMemory(name=self.in_mem_name)
        self.in_buf = np.ndarray(
            self.buf_shape, dtype=self.buf_dtype, buffer=in_mem.buf, order='C')
        self.in_buf.fill(0.0)

        out_mem = SharedMemory(name=self.out_mem_name)
        self.out_buf = np.ndarray(
            self.buf_shape, dtype=self.buf_dtype, buffer=out_mem.buf, order='C')
        self.out_buf.fill(0.0)

        exclusive_settings = sd.WasapiSettings(exclusive=True)

        sd.default.device = (self.in_dev, self.out_dev)

        def output_callback(outdata, frames, time_info, status):
            play_ptr = self.play_ptr.value
            end_ptr = play_ptr + frames

            if end_ptr <= self.buf_size:
                outdata[:] = self.out_buf[play_ptr:end_ptr]
            else:
                first = self.buf_size - play_ptr
                second = end_ptr - self.buf_size
                outdata[:first] = self.out_buf[play_ptr:]
                outdata[first:] = self.out_buf[:second]

            self.play_ptr.value = end_ptr % self.buf_size

        def input_callback(indata, frames, time_info, status):
            # 收录输入数据
            end_ptr = self.__rec_ptr + frames
            if end_ptr <= self.buf_size:  # 整块拷贝
                self.in_buf[self.__rec_ptr:end_ptr] = indata
            else:  # 处理回绕
                first = self.buf_size - self.__rec_ptr
                second = end_ptr - self.buf_size
                self.in_buf[self.__rec_ptr:] = indata[:first]
                self.in_buf[:second] = indata[first:]
            write_pos = self.__rec_ptr
            self.__rec_ptr = end_ptr % self.buf_size

            # 设置信号
            if write_pos < self.block_size and self.__rec_ptr >= self.block_size:
                self.in_ptr.value = 0
                self.in_evt.set()  # 通知主线程来取甲缓冲
            elif write_pos < self.buf_size and self.__rec_ptr < write_pos:
                self.in_ptr.value = self.block_size
                self.in_evt.set()  # 通知主线程来取乙缓冲

        def combined_callback(indata, outdata, frames, time_info, status):
            output_callback(outdata, frames, time_info, status)  # 优先出声
            input_callback(indata, frames, time_info, status)

        if self.is_device_combined:
            with sd.Stream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.buf_dtype,
                latency='low',
                extra_settings=exclusive_settings if
                        self.is_input_wasapi_exclusive and
                        self.is_output_wasapi_exclusive else None,
                callback=combined_callback
            ) as s:
                self.latency.value = s.latency[-1]
                self.stop_evt.wait()
                self.out_buf.fill(0.0)
        else:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.buf_dtype,
                latency='low',
                extra_settings=exclusive_settings if self.is_input_wasapi_exclusive else None,
                callback=input_callback
            ) as si, sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.buf_dtype,
                latency='low',
                extra_settings=exclusive_settings if self.is_output_wasapi_exclusive else None,
                callback=output_callback
            ) as so:
                self.latency.value = si.latency[-1] + so.latency[-1]
                self.stop_evt.wait()
                self.out_buf.fill(0.0)

        # 清理共享内存
        in_mem.close()
        out_mem.close()
        in_mem.unlink()
        out_mem.unlink()
