import subprocess
from kompil.utils.paths import PATH_ROOT


def _simple_shell(cmd: str) -> str:
    # Ensure to work in the good repo
    cmd = f"cd {PATH_ROOT}; {cmd}"
    # Run the command line
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8")
    # Strip the output
    return output.rstrip()


def current_commit():
    return _simple_shell("git rev-parse HEAD")


def current_branch():
    return _simple_shell("git branch --show-current")


def find_closest_main_commit(from_commit: [str, None] = None):
    if from_commit is None:
        from_commit = current_commit()

    return _simple_shell(f"git merge-base origin/master {from_commit}")


def diff(base_commit: [str, None] = None, end_commit: [str, None] = None):
    if not base_commit and not end_commit:
        return _simple_shell("git diff HEAD")

    if not base_commit:
        return _simple_shell(f"git diff HEAD {end_commit}")

    if not end_commit:
        return _simple_shell(f"git diff {base_commit}")

    return _simple_shell(f"git diff {base_commit} {end_commit}")


def get_list_untracked() -> list:
    text = _simple_shell("git ls-files --others --exclude-standard")

    return text.splitlines()
