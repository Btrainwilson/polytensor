import os
from setuptools import setup, find_packages


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file = open(filepath, "r")
    return file.read()


setup(
    name="polytensor",
    version="0.0.1",
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
    install_requires=[line for line in read("requirements.txt").split("\n") if line],
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
