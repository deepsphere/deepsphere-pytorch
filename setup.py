import codecs
import re
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), "r") as fh:
    long_description = fh.read()


def read(*parts):
    with codecs.open(path.join(this_directory, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_requirements(file_name):
    """Strip a requirement files of all comments

    Args:
        file_name (string): File which contains requirements

    Returns:
        list: list of requirements
    """

    with open(path.join(this_directory, "{}.txt".format(file_name)), "r") as file:
        reqs = []

        for req in file.readlines():
            if not req.startswith("#"):
                #if req.startswith("git+"):
                    #split_name = req.split("#")
                    #start = split_name[-1].replace("egg=", "").strip()
                #    end = split_name[0].replace("git+","").strip()
                #    reqs.append(start + " @ " + end)
                #else:
                #    reqs.append(req)
                reqs.append(req)

        print(reqs)
        return reqs


INSTALL_REQUIRES = get_requirements("requirements")
TESTS_REQUIRES = get_requirements("requirements-tests")

EXTRA_REQUIRE = {"tests": TESTS_REQUIRES}

setup(
    name="deepsphere",
    version=find_version("deepsphere", "__init__.py"),
    description="Deep Sphere package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Arcanite",
    author_email="contact@arcanite.ch",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRE,
    packages=find_packages(),
)
