from os import path

from setuptools import setup

with open(path.join(path.abspath(path.dirname(__file__)), "README.md"), "r") as fh:
    long_description = fh.read()


def get_requirements(file_name):
    """Strip a requirement files of all comments

    Args:
        file_name (string): File which contains requirements

    Returns:
        list: list of requirements
    """
    with open(path.join(path.abspath(path.dirname(__file__)), "{}.txt".format(file_name)), "r") as file:
        reqs = []

        for req in file.readlines():
            if not req.startswith("#"):
                if req.startswith("git+"):
                    name = req.split("#")[-1].replace("egg=", "").strip()
                    req.replace("git+", "")
                    reqs.append(f"{name} @ {req}")
                else:
                    reqs.append(req)

        return reqs


INSTALL_REQUIRES = get_requirements("requirements")
TESTS_REQUIRES = get_requirements("requirements-tests")

EXTRA_REQUIRE = {"tests": TESTS_REQUIRES}

setup(
    name="deepsphere",
    version="0.1",
    description="Deep Sphere package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Arcanite",
    author_email="contact@arcanite.ch",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRE,
    packages=["deepsphere"],
)
