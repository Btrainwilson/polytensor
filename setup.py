import os
from setuptools import setup, find_packages


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file = open(filepath, "r")
    return file.read()


setup(
    name="polytensor",
    version="0.0.3",
    author="Blake Wilson",
    author_email="wilso692@purdue.edu",
    description="",
    long_description=read("README.rst"),
    long_description_content_type="text/markdown",
    license="MIT License",
    keywords=[],
    url="",
    packages=find_packages(),
    scripts=[],
    install_requires=[
        "beartype>=0.16.4",
        "filelock>=3.13.1",
        "fsspec>=2023.12.2",
        "Jinja2>=3.1.2",
        "MarkupSafe>=2.1.3",
        "mpmath>=1.3.0",
        "networkx>=3.2.1",
        "numpy>=1.26.3",
        "nvidia-cublas-cu12>=12.1.3.1",
        "nvidia-cuda-cupti-cu12>=12.1.105",
        "nvidia-cuda-nvrtc-cu12>=12.1.105",
        "nvidia-cuda-runtime-cu12>=12.1.105",
        "nvidia-cudnn-cu12>=8.9.2.26",
        "nvidia-cufft-cu12>=11.0.2.54",
        "nvidia-curand-cu12>=10.3.2.106",
        "nvidia-cusolver-cu12>=11.4.5.107",
        "nvidia-cusparse-cu12>=12.1.0.106",
        "nvidia-nccl-cu12>=2.18.1",
        "nvidia-nvjitlink-cu12>=12.3.101",
        "nvidia-nvtx-cu12>=12.1.105",
        "sympy>=1.12",
        "torch>=2.1.2",
        "triton>=2.1.0",
        "typing_extensions>=4.9.0",
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
