import torch
import os
import cv2
import shutil
import time
import gc

from torch.autograd import Variable
from torch import optim

# don't forget set PYTHON PATH
from kompil.utils.paths import make_dir, clear_dir, PATH_BUILD
from kompil.metrics import compute_psnr
from kompil.train.loss.base import factory as loss_factory
from kompil.utils.time import now_str

BENCH_DIR = os.path.join(PATH_BUILD, f"benchmarks/losses-{now_str()}")


class Image:
    def __init__(self, path=None, data=None, name=None):
        self.best_loss = None
        self.best_eval = 0
        self.best_loss_time = 0

        if path:
            print(f"Creating image from {path}...")

            assert os.path.exists(path)

            res_frame = cv2.imread(path, cv2.IMREAD_COLOR)
            res_frame = res_frame.transpose(2, 0, 1)

            self.data = res_frame

            self.name = os.path.basename(path)
        elif data:
            self.data = data

        if name:
            self.name = name

    @staticmethod
    def write(path, img):
        img = img.transpose(1, 2, 0)
        cv2.imwrite(path, img)


class BenchLosses:
    def __init__(self, epochs: int = 100, learning_rate: float = 0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate

    def start(self):
        imgs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "res/imgs"
        )
        imgs = self.load_images(imgs_dir)

        clear_dir(BENCH_DIR)
        make_dir(BENCH_DIR)

        for img in imgs:
            print()
            print(f"    ***Processing on {img.name}***")
            for loss_name, loss_fn in loss_factory().items():
                res_img, res_eval = self.converge_to_image(img, loss_fn)

                if res_eval > img.best_eval:
                    img.best_eval = res_eval
                    img.best_loss = loss_name

                Image.write(f"{BENCH_DIR}/generated-{loss_name}-{img.name}", res_img)
                gc.collect()

        print()
        print(f"    ***Results***")
        print(f"Epochs : {self.epochs}")
        print(f"Learning rate : {self.learning_rate}")

        for img in imgs:
            print(f"===> Best loss for image {img.name} : {img.best_loss} with {img.best_eval}")

    def load_images(self, folder_path) -> list:
        assert os.path.exists(folder_path)

        imgs = []

        for img_file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, img_file)
            imgs.append(Image(full_path))

        return imgs

    def converge_to_image(self, img_ref, loss_fn):
        print(f"Converging with {loss_fn}")

        img_ref_tensor = torch.from_numpy(img_ref.data).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_gen_tensor = torch.rand(img_ref_tensor.size())

        if torch.cuda.is_available():
            img_ref_tensor = img_ref_tensor.cuda()
            img_gen_tensor = img_gen_tensor.cuda()

        img_ref_tensor = Variable(img_ref_tensor, requires_grad=False)
        img_gen_tensor = Variable(img_gen_tensor, requires_grad=True)

        optimizer = optim.Adam([img_gen_tensor], lr=self.learning_rate)
        start_time = time.time()
        psnr_res = 0
        params = {
            "y_pred": img_gen_tensor,
            "y_ref": img_ref_tensor,
            "chw_size": img_ref_tensor.nelement(),
        }

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            loss_ite, _ = loss_fn(**params)
            loss_ite.backward()
            optimizer.step()
            psnr_res = compute_psnr(img_ref_tensor, img_gen_tensor)
            print(
                f"    Epoch {epoch + 1}/{self.epochs} with PSNR score : {psnr_res}",
                end="\r",
            )

        print()
        compute_time = time.time() - start_time
        print(f"    Total time : {compute_time}")
        print(f"    Mean time : {compute_time / self.epochs}")

        img_gen_tensor = (img_gen_tensor * 255.0).squeeze()
        img_res = img_gen_tensor.detach().cpu().type(torch.uint8).numpy()

        return img_res, psnr_res


if __name__ == "__main__":
    BenchLosses(epochs=50, learning_rate=0.1).start()
