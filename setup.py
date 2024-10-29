# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the package."""

import os
import re
from io import open

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
PACKAGE_NAME = "oat"

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    readme = f.read()


def _read_version():
    VERSION_FILE = f"{PACKAGE_NAME}/__about__.py"
    ver_str_line = open(VERSION_FILE, "rt").read()
    version_re = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(version_re, ver_str_line, re.M)
    if mo:
        version = mo.group(1)
    else:
        raise RuntimeError(f"Unable to find version string in {VERSION_FILE}.")
    return version


setup(
    name=PACKAGE_NAME,
    version=_read_version(),
    author="Zichen Liu",
    author_email="lkevinzc@gmail.com",
    description="Online AlignmenT (OAT) for LLMs.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/sail-sg/oat",
    license="Apache-2.0",
    packages=find_packages(exclude=["examples*", "k8s*", "benchmark*", "test*"]),
    include_package_data=True,
    python_requires=">=3.8, <3.11",
    setup_requires=["setuptools_scm>=7.0"],
    zip_safe=False,
)
