import os


def get_video(video_name: str, as_pk: bool = False) -> str:
    """
    Get video path and download it if necessary.
    """
    # Check the name is a path instead
    if os.path.exists(video_name):
        print(f"Video {video_name} is read from the hard drive.")
        return video_name


def get_pytorch_model(pymod_name: str, as_pk: bool = False):
    """
    Get model path and download it if necessary.
    """
    # Check the name is a path instead
    if os.path.exists(pymod_name):
        print(f"Model {pymod_name} is read from the hard drive.")
        return pymod_name
