import os

from kompil.nn.models.model import VideoNet, model_save, checkpoint_save
from kompil.profile.report import EncodingReport


def save_local_model_and_report(model: VideoNet, report: EncodingReport, output_folder):
    try:
        print(f"Local backup...")

        if model:
            model_path = os.path.join(output_folder, f"{model.name}.pth")
            print(f"Saving model as {model_path}")
            model_save(model, model_path)

        if report:
            print(f"Saving report in {output_folder} directory...")
            report.save_in(output_folder)

        if model and report:
            checkpoints_folder = os.path.join(output_folder, "checkpoints")
            if not os.path.exists(checkpoints_folder):
                print(f"Creating checkpoint folder...")
                os.makedirs(checkpoints_folder)
            checkpoint_path = os.path.join(checkpoints_folder, f"{model.name}.checkpoint.pth")
            print(f"Saving checkpoint as {checkpoint_path}")
            checkpoint_save(model, checkpoint_path, epoch=report.epochs)

        return True
    except Exception as e:
        print(f"Failed saving local report. Reason: {e}")

        return False
