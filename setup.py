import io
import re

from setuptools import find_packages, setup

with io.open("acvp/__init__.py", "rt", encoding="utf8") as f:
    VERSION = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)


setup(
    name="rainy-acvp",
    version=VERSION,
    url="https://github.com/kngwyu/rainy-acvp",
    project_urls={
        "Code": "https://github.com/kngwyu/rainy-acvp",
        "Issue tracker": "https://github.com/kngwyu/rainy-acvp/issues",
    },
    author="Yuji Kanagawa",
    author_email="yuji.kngw.80s.revive@gmail.com",
    description="A collection of DRL algorithms with intrinsic rewards",
    packages=find_packages(),
    python_requires=">=3.6",
)
