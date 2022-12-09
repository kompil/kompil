import unittest
import torch

from kompil.utils.video import discretize


class TestDiscretize(unittest.TestCase):
    def test_0(self):
        fake_input = torch.Tensor(3, 2000, 3000).uniform_(0, 1)
        output_255 = fake_input * 255.0  # up to [0; 255] range like byte-storage version
        floor_255 = torch.floor(output_255)  # considere floor because of float to byte conversion
        disc_output = floor_255 / 255.0  # back to [0; 1] range

        ouput_255_uint8 = output_255.byte()  # Normal [0; 255] upscale
        disc_output_uint8 = (disc_output * 255.0).byte()  # Output with discretization

        self.assertTrue(torch.all(torch.eq(ouput_255_uint8, disc_output_uint8)))

    def test_1(self):
        fake_input = torch.Tensor(3, 2000, 3000).uniform_(0, 1)
        disc_output = discretize(fake_input)
        disc_output_uint8 = (disc_output * 255.0).byte()  # Output with discretization

        output_255 = fake_input * 255.0
        ouput_255_uint8 = output_255.byte()  # Normal [0; 255] upscale

        self.assertTrue(torch.all(torch.eq(ouput_255_uint8, disc_output_uint8)))


if __name__ == "__main__":
    unittest.main()
