# deepsphere

Pytorch implementation of DeepSphere.


## Tests
The different tests of this package can be run with the command:
```bash
python -m unittest discover -v
```
It will automatically run all the tests found in the repo and print more verbosity (the doc of the tests) with the `-v` option


## Optional configurations
### Code formatting
In order to make the code of the repo nice and clear, it use several external tools in order to properly format the code.


#### Black
[Black](https://pypi.org/project/black/) is a code formatter made to be uncompromising. Therefore no time is spent in order to determine how things
should be formatted since there is no choice to make.
It also format the code in order to produce the smallest diff possible.

To run black on a folder or a file, simply run
```bash
black <path>
```

The configuration of black can be found in the file `pyproject.toml`


#### Isort
[Isort](https://pypi.org/project/isort/) is a external tool that sort and format all the import of a file.

To command to format a file is:
```bash
isort <path_file>
```

or to run over a folder
```bash
isort -rc <path_folder>
```

The configuration is also stored in the file `pyproject.toml`


#### pylint
[Pylint](https://www.pylint.org/) is another external tool that can check the coding standard such as line length, common mistake, error, ...
It is used in the CI with the combination of [pylint-fail-under](https://pypi.org/project/pylint-fail-under/) a wrapper.

The command to check the output over a folder or a file of pylint is:
```bash
pylint --rcfile=setup.cfg <path>
```

The configuration is stored in the file `setup.cfg`


### Pre Commit hook
[pre-commit](https://pre-commit.com/) is a tool that will apply some hooks before every commit you do. This will be used in order to run the different
formatting tools.


#### Installation
In order to be able to use the pre-commit hook, you first need to install pre-commit thanks to the file `requirements-tests.txt`
Then run the command
```bash
pre-commit install
```
It will automatically create the hook and run it every time you do a commit.


#### Function
The pre-commit hook does 4 things:
1. Adding blank line at the end of file if it's missing
2. Removing trailing white spaces
3. Applying Black
4. Applying Isort


#### Configuration
The configuration of pre-commit can be found in the file .pre-commit-config.yaml

### Documentation
In order to setup some documentation of the project, [Sphinx](http://www.sphinx-doc.org/en/master/) is used. It's a library that will generate a nice
documentation from the different comments from the code.

In order for it to work, the docstring should be of the google style format (some example can be found [here](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) - from the google styleguide and [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) - the plugin used by Sphinx in order to parse google docstring)
To generate the documentation, follow the different steps:

1. Install Sphinx via the `requirements-tests.txt` file.
2. If you added new .py files in the project, go inside the folder `docs/` (Important) and run the command `sphinx-apidoc -f -o source ../`
3. You can customize the files by editing the .rst files in the folder `docs/source/`
4. Generate the doc with `make html` while being inside the folder `docs/`
5. Open the file `docs/bild/html/index.html` in a navigator to see the documentation
