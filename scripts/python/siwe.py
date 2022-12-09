import os
import sys
import torch
import json

from kompil.utils.resources import get_video
from kompil.utils.time import now_str

_SIWE_BUILD_DIR = os.path.join(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "build/siwe/"
    ),
    now_str(),
)

_KOMPIL_PY_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "kompil.py"
)

_DEFAULT_SIWE_BATCH_SIZE = 16
_DEFAULT_SIWE_LR = "primus"
_DEFAULT_SIWE_SCHEDULER = "tycho"
_DEFAULT_SIWE_MAX_EPOCH = 3000
_DEFAULT_SIWE_AROUSE_ZERO = [500, 1500, 100, -0.1, 0.1]  # /!\ arouse can make unstable training
_DEFAULT_SIWE_RESOLUTION = "320p"
_DEFAULT_SIWE_TOPOLOGY = "donnager_section_walloc"

_DEFAULT_SIWE_SECTIONIZE_METHOD = "cluster"
_DEFAULT_SIWE_NB_SECTION = 5

_DEFAULT_SIWE_WALLOC_CONFIRM = False
_DEFAULT_SIWE_WALLOC_SHIFT_TOLERANCE = 1

_DEFAULT_SIWE_ITERATION = 5
_DEFAULT_SIWE_CONFIRM_ITERATION = False


def _run_kompil_cmd(args: str):
    assert args

    print()
    print("Running cmd : kompil", args)

    os.system(f"python3 {_KOMPIL_PY_SCRIPT} {args}")


def _build_switch(video_fpath: str, sections_fpath: str, final_model_fpath: str):
    assert os.path.exists(video_fpath)
    assert os.path.exists(sections_fpath)

    nb_frame = len(torch.load(sections_fpath))

    _run_kompil_cmd(
        f"topology init {nb_frame} auto_switch_cluster {final_model_fpath} --model-extra {sections_fpath} {_SIWE_BUILD_DIR}/section-"
    )


def _eval(
    model_fpath: str,
    video_fpath: str,
    sections_fpath: str = None,
    section_idx: int = None,
):
    assert os.path.exists(model_fpath)
    assert os.path.exists(video_fpath)

    output_fpath = os.path.join(_SIWE_BUILD_DIR, "eval.json")

    section_option = f"--cluster {sections_fpath} {section_idx}" if sections_fpath else ""

    _run_kompil_cmd(
        f"eval {model_fpath} {video_fpath} --output-file {output_fpath} {section_option}"
    )

    with open(output_fpath) as file:
        result_data = json.load(file)

    mean_ssim = result_data["ssim"]["mean"]
    psnr_ssim = result_data["psnr"]["mean"]

    return mean_ssim * psnr_ssim


def _encode(
    video_fpath: str,
    sections_fpath: str,
    section_idx: int,
    resolution: str,
    walloc_fpath: str,
    topology: str,
    max_epoch: int,
    arouse_zero: list,
    learning_rate: str,
    scheduler: str,
    batch_size: int,
):
    assert os.path.exists(sections_fpath)
    assert os.path.exists(video_fpath)
    assert os.path.exists(walloc_fpath)

    cmd = f"encode {video_fpath} "
    cmd += f"--name section-{section_idx} "
    cmd += f"--output-folder {_SIWE_BUILD_DIR} "
    cmd += f"--resolution {resolution} "
    cmd += f"--topology-builder {topology} "
    cmd += f"--cluster {sections_fpath} {section_idx} "
    cmd += f"--model-extra {walloc_fpath} {section_idx} "
    cmd += f"--autoquit-epoch {max_epoch} "
    cmd += f"--arouse-zero {' '.join(str(i) for i in arouse_zero)} " if arouse_zero else ""
    cmd += f"--scheduler {scheduler} "
    cmd += f"--learning-rate {learning_rate} "
    cmd += f"--batch-size {batch_size} "

    cmd += "--precision 16 "
    cmd += "--loading-mode full-ram "

    _run_kompil_cmd(cmd)

    return _eval(
        f"{_SIWE_BUILD_DIR}/section-{section_idx}.pth",
        video_fpath,
        sections_fpath,
        section_idx,
    )


def _walloc(
    sections_fpath: str,
    total_mp: float,
    prev_weights_allocation: torch.Tensor = None,
    prev_results: torch.Tensor = None,
    force: list = None,
):
    assert os.path.exists(sections_fpath)

    force_str = ""

    # Force weight assignation, only happens at initialization basically
    if force:
        force_str = "-f " + " ".join(str(i) for i in force)
    # Use encoding results to rewalloc parameters
    elif prev_weights_allocation is not None and prev_results is not None:
        force_str = "-f "
        mean_results = prev_results.mean()
        rewalloc = torch.zeros_like(prev_weights_allocation)
        nb_section = len(prev_results)
        nb_best = int(nb_section / 2) + 1
        _, all_best_results_idx = torch.topk(prev_results, nb_best)

        # Count of not good enough / just mean results
        nb_not_good = torch.count_nonzero(prev_results < mean_results).item()

        # Check over all sections
        for sect_idx in range(nb_section):
            sect_encoding_result = prev_results[sect_idx]
            rewalloc[sect_idx] += prev_weights_allocation[sect_idx]

            # If result is not good enough...
            if sect_encoding_result < mean_results:
                # Then iterate over only good results
                for one_best_idx in all_best_results_idx:
                    one_best_sect_encoding_result = prev_results[one_best_idx]

                    # Don't take mp from not good enough section
                    if one_best_sect_encoding_result <= mean_results:
                        break  # Only worst further

                    # Compute the swap amount
                    one_best_sect_prev_walloc = prev_weights_allocation[one_best_idx]
                    nb_params_shift = (
                        (1 - (mean_results / one_best_sect_encoding_result))
                        * one_best_sect_prev_walloc
                    ) / nb_not_good

                    # Then swap some walloc between them
                    rewalloc[one_best_idx] -= nb_params_shift
                    rewalloc[sect_idx] += nb_params_shift

        rewalloc /= 1e6

        for sect_idx in range(len(rewalloc)):
            force_str += f"{sect_idx} {rewalloc[sect_idx]} "

    walloc_fpath = os.path.join(_SIWE_BUILD_DIR, "walloc.pth")

    _run_kompil_cmd(f"walloc {sections_fpath} {total_mp} {walloc_fpath} {force_str}")

    return walloc_fpath


def _sectionize(video_fpath: str, nb_section: int, sect_method: str) -> str:
    assert os.path.exists(video_fpath)

    sections_fpath = os.path.join(_SIWE_BUILD_DIR, "sections_idx.pth")

    _run_kompil_cmd(f"sectionize {video_fpath} {nb_section} {sections_fpath} -m {sect_method}")

    return sections_fpath


def siwe(config_fpath: str):
    # Get config
    with open(config_fpath) as file:
        config_data = json.load(file)

        # Global config
        video_name = config_data["video_name"]
        video_fpath = get_video(video_name)
        model_fpath = os.path.join(_SIWE_BUILD_DIR, config_data["model_name"])
        max_iteration = int(config_data.get("max_iteration", _DEFAULT_SIWE_ITERATION))
        confirm_new_iteration = config_data.get(
            "confirm_new_iteration", _DEFAULT_SIWE_CONFIRM_ITERATION
        )

        # Walloc config
        walloc_data = config_data.get("walloc", {})
        walloc_total_mp = float(  # If no provided mp, deduced from input file
            walloc_data.get("initial_total_mp", os.path.getsize(video_fpath) / 1024 / 1024)
        )
        walloc_initial_mp = walloc_data.get("initial_mp", None)
        walloc_confirm = walloc_data.get("confirm", _DEFAULT_SIWE_WALLOC_CONFIRM)
        walloc_shift_tolerance = float(
            walloc_data.get("shift_tolerance", _DEFAULT_SIWE_WALLOC_SHIFT_TOLERANCE)
        )

        # Sectionize config
        sectionize_data = config_data.get("sectionize", {})
        section_method = sectionize_data.get("method", _DEFAULT_SIWE_SECTIONIZE_METHOD)
        nb_section = int(sectionize_data.get("nb_section", _DEFAULT_SIWE_NB_SECTION))

        # Encode config
        encode_data = config_data.get("encode", {})
        resolution = encode_data.get("resolution", _DEFAULT_SIWE_RESOLUTION)
        learning_rate = encode_data.get("learning_rate", _DEFAULT_SIWE_LR)
        scheduler = encode_data.get("scheduler", _DEFAULT_SIWE_SCHEDULER)
        topology = encode_data.get("topology", _DEFAULT_SIWE_TOPOLOGY)
        arouse_zero = encode_data.get("arouse_zero", _DEFAULT_SIWE_AROUSE_ZERO)
        max_epoch = int(encode_data.get("max_epoch", _DEFAULT_SIWE_MAX_EPOCH))
        batch_size = int(encode_data.get("batch_size", _DEFAULT_SIWE_BATCH_SIZE))

    os.makedirs(_SIWE_BUILD_DIR, exist_ok=True)

    encoding_results = torch.zeros(nb_section)

    # 1. Sectionize the whole video according method
    sections_fpath = _sectionize(video_fpath, nb_section, section_method)

    # 2. Initial weight allocation according nb frame in each section
    walloc_fpath = _walloc(sections_fpath, walloc_total_mp, force=walloc_initial_mp)
    weights_allocation = torch.load(walloc_fpath)
    updated_walloc = torch.BoolTensor(nb_section).fill_(True)

    # 3. Iterative encoding according specified walloc
    for ite in range(1, max_iteration + 1):
        print()
        print(f"Iteration {ite} walloc : {weights_allocation.tolist()}")

        # 3.1. Encode each section
        for section_idx in range(nb_section):
            print()
            print(
                "*********************************",
                f"Iteration {ite} - Section {section_idx}",
                "*********************************",
            )
            # Encode only if weights updated seen last walloc
            if updated_walloc[section_idx]:
                encoding_results[section_idx] = _encode(
                    video_fpath,
                    sections_fpath,
                    section_idx,
                    resolution,
                    walloc_fpath,
                    topology,
                    max_epoch,
                    arouse_zero,
                    learning_rate,
                    scheduler,
                    batch_size,
                )
            else:
                print(f"No need to re-encode section {section_idx}")

        print()
        print(f"Iteration {ite} walloc : {weights_allocation.tolist()}")
        print(
            f"Iteration {ite} results : {encoding_results.tolist()}, mean : {encoding_results.mean().item()}"
        )

        if confirm_new_iteration:
            user_input_confirm_next = input(f"Confirm next iteration ({ite + 1}) ? [y/n]")

            if user_input_confirm_next in ["n", "no"]:
                break

        # 3.2 Re-walloc according encoding results
        walloc_fpath = _walloc(
            sections_fpath, walloc_total_mp, weights_allocation, encoding_results
        )
        new_weights_allocation = torch.load(walloc_fpath)

        if walloc_confirm:
            user_input_confirm = input(
                f"Generated walloc : {new_weights_allocation.tolist()}, confirm for next iteration ({ite + 1}) ? [y/n]"
            )
            if user_input_confirm in ["n", "no", "modify"]:
                user_input_force = input(
                    f"{user_input_confirm} -> Please insert your walloc assignation in EXACT format sect_idx_x mp_x sect_idx_y mp_y... :"
                )
                if user_input_force:
                    # Re-walloc according user inputs
                    walloc_fpath = _walloc(
                        sections_fpath,
                        walloc_total_mp,
                        weights_allocation,
                        encoding_results,
                        force=user_input_force,
                    )
                    new_weights_allocation = torch.load(walloc_fpath)
                else:
                    print(f"{user_input_force} -> Bad user entrance, keep walloc as it")
            else:
                print(
                    f"{user_input_confirm} -> Keep walloc as it :", new_weights_allocation.tolist()
                )

        # Check which section has been re-walloc
        updated_walloc = new_weights_allocation != weights_allocation
        # Compute distance beetween previous and current walloc
        walloc_shift = (
            torch.sqrt(torch.sum(torch.square(new_weights_allocation - weights_allocation) + 1e-8))
            / 1e6
        )
        weights_allocation = new_weights_allocation

        print()
        print(f"Iteration {ite} shift : {walloc_shift.item()}")

        # If shift is small, means the current walloc is good enough
        if walloc_shift < walloc_shift_tolerance:
            print("Shift close enough : Stopping SIWE")
            break

    print()
    print("********************************* END *********************************")

    # 4. Build up the final topology with all data among
    _build_switch(video_fpath, sections_fpath, model_fpath)

    # 5. Final evaluation
    score = _eval(model_fpath, video_fpath)

    print("Final score :", score)


if __name__ == "__main__":
    args = sys.argv[1:]

    assert len(args) == 1, "Must provide SIWE configuration file"

    siwe(args[0])
