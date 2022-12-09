import os
import shutil
import fnmatch

PATH_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH_RESOURCES = os.path.join(PATH_ROOT, "res")
PATH_VIDEOS = os.path.join(PATH_RESOURCES, "videos")
PATH_MODELS = os.path.join(PATH_RESOURCES, "models")
PATH_BUILD = os.path.join(PATH_ROOT, "build")
PATH_LOGS = os.path.join(PATH_ROOT, "logs")
PATH_CONFIG = os.path.expanduser("~/.kompil")
PATH_CONFIG_HW = os.path.join(PATH_CONFIG, "hw.json")


def clear_dir(path: str, filter_str: str = None):
    if not path:
        return

    if os.path.exists(path):
        if filter_str:
            files = fnmatch.filter(os.listdir(path), filter_str)
        else:
            files = os.listdir(path)

        for filename in files:
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


def make_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def make_logs_dir():
    make_dir(PATH_LOGS)


def zip_dir(dir_path: str, zipfile_path: str) -> bool:
    if not os.path.exists(dir_path):
        return False

    try:
        if os.path.exists(zipfile_path):
            os.unlink(zipfile_path)

        if zipfile_path.endswith(".zip"):
            zipfile_path = os.path.splitext(zipfile_path)[0]

        shutil.make_archive(zipfile_path, "zip", dir_path)

        return True
    except Exception as e:
        print(f"Failed zip {zipfile_path}. Reason: {e}")
        return False
