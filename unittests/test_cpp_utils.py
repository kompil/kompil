import unittest
import torch

from kompil.utils.cpp_utils import _PythonImplem, _CppImplem, get_gc_nodes


class TestAdjacent2d(unittest.TestCase):
    def setUp(self):
        pass

    def test_consistency(self):
        test_sample = 100
        # Test gray code
        for i in range(test_sample):
            cpp_gc = _CppImplem.graycode(i)
            python_gc = _PythonImplem.graycode(i)
            self.assertEqual(cpp_gc, python_gc)

        # Test layer binary
        t_val = torch.arange(0, test_sample).view(-1, 1)
        nodes = get_gc_nodes(test_sample)
        cpp_out = torch.empty(t_val.shape[0], nodes)
        python_out = torch.empty(t_val.shape[0], nodes)
        _CppImplem.layer_binary(t_val, cpp_out)
        _PythonImplem.layer_binary(t_val, python_out)
        self.assertTrue(torch.equal(cpp_out, python_out))


if __name__ == "__main__":
    unittest.main()
