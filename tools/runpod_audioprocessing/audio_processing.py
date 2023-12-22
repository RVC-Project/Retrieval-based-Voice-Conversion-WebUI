import os
import re
import subprocess
import tempfile

import ffmpeg

from tools.runpod_audioprocessing.utils import AudioProcessingUtils


class AudioProcessing:
    SILENCE_DETECT_PARAMS = 'silencedetect=noise=-50dB:d=0.5'
    LOUDNESS_NORMALIZATION_PARAMS = 'loudnorm=I=-20:dual_mono=false:TP=-1:LRA=11:print_format=summary'
    SPEECH_SAFETY_PAD = 0.2

    def __init__(self, input_audio_path: str):
        if not os.path.isfile(input_audio_path):
            raise FileNotFoundError(f"Input audio file '{input_audio_path}' does not exist.")

        self._speech_start = None
        self._speech_end = None
        self._speech_segments = []
        self._input_audio_path = input_audio_path
        self._input_audio_duration = float(ffmpeg.probe(self._input_audio_path)["format"]["duration"])
        self._joined_speeches_audio_path = AudioProcessingUtils.get_temp_file_path('joined_speeches.wav')

    @property
    def joined_speeches_audio_path(self):
        return self._joined_speeches_audio_path

    def _prepare_speech_segments_from_time(self, silence_time: float, silence_time_type: str) -> None:
        if silence_time_type == 'end':
            # speech start is silence end, should not fall off below 0
            self._speech_start = max((silence_time - self.SPEECH_SAFETY_PAD), 0)
        elif silence_time_type == 'start' and silence_time != 0.0:
            # speech end is silence start, should not be greater than input audio duration
            self._speech_end = min((silence_time + self.SPEECH_SAFETY_PAD), self._input_audio_duration)
        else:
            return

        # add speech segment if we know speech start and end
        # or if audio starts with sound, we must add 1st speech as segment from 0 to first silence start
        if ((self._speech_start and self._speech_end)
                or (not self._speech_start and not self._speech_segments and silence_time != 0.0)):
            self._speech_segments.append({
                'start': self._speech_start if self._speech_start else 0.0,
                'end': self._speech_end,
                'duration': (self._speech_end - (self._speech_start if self._speech_start else 0.0))}
            )
            self._speech_start = None
            self._speech_end = None
            return

    def _extract_speech_segments(self, contents: str) -> list:
        matches = re.findall(r'silence_(start|end): (\d+(\.\d+)?)', contents)
        for match in matches:
            time_string = match[1]
            time_type = match[0]  # start or end
            if bool(re.match(r'^\d+(\.\d+)?$', time_string)):
                self._prepare_speech_segments_from_time(float(time_string), time_type)

        # special case if no silence gaps were detected by ffmpeg
        if not self._speech_segments:
            self._speech_segments.append({
                'start': 0.0,
                'end': self._input_audio_duration,
                'duration': self._input_audio_duration}
            )

        # special case if no silence was detected at the end of audio
        if self._speech_start and not self._speech_end:
            self._speech_segments.append({
                'start': self._speech_start,
                'end': self._input_audio_duration,
                'duration': self._input_audio_duration - self._speech_start}
            )
        return self._speech_segments

    def extract_speeches(self) -> None:
        try:
            out, err = (ffmpeg
                        .input(self._input_audio_path)
                        .output('-', format='null', af=self.SILENCE_DETECT_PARAMS)
                        .run(capture_stdout=True, capture_stderr=True)
                        )
            # The ffmpeg command prints the output of the -af silencedetect option to stderr and not to stdout.
            # This is part of its design and not due to an error.
            ffmpeg_output = err.decode()
            self._extract_speech_segments(ffmpeg_output)
        except ffmpeg.Error as e:
            print('ffmpeg error')
            print(e.stderr.decode())

    def join_speech_segments(self) -> None:
        temp_file_list = AudioProcessingUtils.get_temp_file_path('temp_file_list.txt')

        try:
            for segment in self._speech_segments:
                temp_segment_file = AudioProcessingUtils.get_temp_file_path("temp_" + str(segment['start']) + ".wav")
                (
                    ffmpeg
                    .input(self._input_audio_path, ss=segment['start'], to=segment['end'])
                    .output(
                        temp_segment_file,
                        af=self.LOUDNESS_NORMALIZATION_PARAMS
                    )
                    .run(overwrite_output=True)
                )
                AudioProcessingUtils.append_to_temp_file(temp_file_list, f"file '{temp_segment_file}'\n")

            # Concatenate all segments
            (
                ffmpeg
                .input(temp_file_list, format='concat', safe=0)
                .output(self._joined_speeches_audio_path, c='copy')
                .run(overwrite_output=True)
            )

            # Clean up temporary files
            AudioProcessingUtils.remove_temp_files('temp_*.wav')
            os.remove(temp_file_list)
        except ffmpeg.Error as e:
            print('ffmpeg error')
            print(e.stderr.decode())

    def get_auto_pitch_correction(self, model_f0m: float) -> int:
        input_audio_f0m = AudioProcessingUtils.get_fundamental_frequency(self._joined_speeches_audio_path)
        return AudioProcessingUtils.semitone_distance(model_f0m, input_audio_f0m)

    def restore_speeches_timing(self, converted_file_no_silence: str, output_file: str) -> None:
        # Initialize variables
        current_position_in_original = 0.0
        current_position_in_joined = 0.0
        file_list = AudioProcessingUtils.get_temp_file_path('file_list.txt')
        if os.path.isfile(file_list):
            os.remove(file_list)

        try:
            for segment in self._speech_segments:
                # Calculate duration of silence before the speech segment
                silence_duration = float(segment['start']) - current_position_in_original

                # Create silent audio segment
                if silence_duration > 0:
                    silence_file = AudioProcessingUtils.get_temp_file_path("silence_" + str(segment['start']) + ".wav")
                    cmd = [
                        "ffmpeg",
                        "-f", "lavfi",
                        "-i", "anullsrc=r=48000:cl=mono",
                        "-t", str(silence_duration),
                        "-q:a", "9",
                        silence_file
                    ]
                    subprocess.run(cmd, check=True)
                    AudioProcessingUtils.append_to_temp_file(file_list, f"file '{silence_file}'\n")

                # Extract the speech segment from the joined file
                speech_file = AudioProcessingUtils.get_temp_file_path("speech_" + str(segment['start']) + ".wav")
                (
                    ffmpeg
                    .input(converted_file_no_silence, ss=str(current_position_in_joined), t=str(segment['duration']))
                    .output(speech_file, ac=1, ar='48000')
                    .run(overwrite_output=True)
                )
                AudioProcessingUtils.append_to_temp_file(file_list, f"file '{speech_file}'\n")

                # Update the current positions
                current_position_in_original = float(segment['end'])
                current_position_in_joined += float(segment['duration'])

            # Concatenate all segments
            (
                ffmpeg
                .input(file_list, format='concat', safe=0)
                .output(output_file, c='copy')
                .run(overwrite_output=True)
            )

            # Clean up temporary files
            for file in os.listdir(tempfile.gettempdir()):
                if (file.startswith('silence_') or file.startswith('speech_')) and file.endswith('.wav'):
                    os.remove(os.path.join(tempfile.gettempdir(), file))
            os.remove(file_list)
        except ffmpeg.Error as e:
            print('ffmpeg error')
            print(e.stderr.decode())
