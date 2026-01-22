from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='reorder_kernel',
    ext_modules=[
        CUDAExtension(
            name='reorder_kernel',
            sources=['reorder_kernel.cu']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
