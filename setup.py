from __future__ import absolute_import

from glob import glob
import os
import sys
import setuptools


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


def read_version():
    return read("VERSION").strip()


packages = setuptools.find_packages(where="src")

required_packages = [
    "pip",
    "psutil>=5.6.7",
]

setuptools.setup(
    name="aws_topology",
    version=read_version(),
    description="Library for launching topology-aware multi-spine jobs on Amazon SageMaker.",
    packages=packages,
    package_dir={"aws_topology": "src/aws_topology"},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Amazon Web Services",
    url="https://github.com/aws/sagemaker-training-toolkit/",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=required_packages,
    data_files=[('bin', ['src/aws_topology/bin/multispine_latency_calculator'])],
    include_package_data=True,
)