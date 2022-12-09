import unittest
import torch
import math


def subblock(x, factor):
    b, c_i, h_i, w_i = x.shape
    factor_sqr = factor**2

    c_o = int(c_i / factor_sqr)
    h_o = int(h_i * factor)
    w_o = int(w_i * factor)

    full = torch.Tensor(b, c_o, h_o, w_o).to(x.device)

    for c_i_idx in range(c_i):
        c_i_idx_rel = c_i_idx % factor_sqr

        c_o_idx = int(c_i_idx / factor_sqr)
        h_o_idx = int((c_i_idx_rel * h_i) / h_o) * h_i
        w_o_idx = (c_i_idx_rel * w_i) % w_o

        h_o_end = h_o_idx + h_i
        w_o_end = w_o_idx + w_i

        full[:, c_o_idx, h_o_idx:h_o_end, w_o_idx:w_o_end] = x[:, c_i_idx]

    return full


class TestSubBlock(unittest.TestCase):
    def test_0(self):
        h = 5
        w = 10
        x = torch.Tensor(1, 16, h, w).to("cuda")

        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                x[b, c] = torch.ones(h, w) * (c + 1)

        y = subblock(x, 4)  # 20, 40

        k = 1
        for b in range(y.shape[0]):
            for c in range(y.shape[1]):
                for hh in range(0, y.shape[2], h):
                    for ww in range(0, y.shape[3], w):
                        vals = torch.ones(h, w).to("cuda") * k
                        pick = y[b, c, hh : (hh + h), ww : (ww + w)]

                        self.assertTrue(torch.all(torch.eq(pick, vals)))
                        k += 1


if __name__ == "__main__":
    unittest.main()
