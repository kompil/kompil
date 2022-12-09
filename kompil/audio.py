import os

import kompil.utils.ffmpeg as ffmpeg


def extract_audio(src_path: str, output_path: str):
    if not output_path:
        folder, file_name = os.path.split(src_path)
        name, _ = os.path.splitext(file_name)
        mp3 = name + ".mp3"
        output_path = os.path.join(folder, mp3)

    ffmpeg.extract_audio(src_path, output_path)

    print(f"Audio extract at {output_path}")
