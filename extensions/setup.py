from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

ext_modules = []

ext_modules.append(CUDAExtension(
    name='kompil_adjacent',
    sources=['adjacent.cpp', 'adjacent_2d.cu', 'adjacent_1d.cu'],
    extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
))

ext_modules.append(CUDAExtension(
    name='kompil_activations',
    sources=['activations.cpp', 'activations_cuda.cu'],
    extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2', '--extended-lambda']},
))

ext_modules.append(CppExtension(
    name="kompil_vmaf",
    sources=["vmaf.cpp"],
    libraries=["vmaf"],
))

ext_modules.append(CppExtension(
    name="kompil_utils",
    sources=["utils.cpp"],
))

setup(
    name='kompil_extensions',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)