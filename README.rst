scikit-surgerytorch
===============================

.. image:: https://github.com/UCL/scikit-surgerytorch/raw/master/project-icon.png
   :height: 128px
   :width: 128px
   :target: https://github.com/UCL/scikit-surgerytorch
   :alt: Logo

.. image:: https://github.com/UCL/scikit-surgerytorch/badges/master/build.svg
   :target: https://github.com/UCL/scikit-surgerytorch/pipelines
   :alt: GitLab-CI test status

.. image:: https://github.com/UCL/scikit-surgerytorch/badges/master/coverage.svg
    :target: https://github.com/UCL/scikit-surgerytorch/commits/master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/scikit-surgerytorch/badge/?version=latest
    :target: http://scikit-surgerytorch.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



Author: Thomas Dowrick

scikit-surgerytorch is part of the `scikit-surgery`_ software project, developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, part of `University College London (UCL)`_.

The aim of scikit-surgerytorch is to provide a home for various pytorch examples and
utilities and to show best practice. It's NOT meant to be a layer on-top of pytorch
or provide a new kind-of platform. The aim is that researchers can learn from examples,
and importantly, learn how to deliver an algorithm that can be used by other people
out of the box, with just a ```pip install```, rather than a new user having to
re-implement stuff, or struggle to get someone else's code running. Researchers
can commit their research to this repository, or use the `PythonTemplate`_ to
generate their own project as a home for their new world-beating algorithm!

Features
----------

Each project herein should provide the following:

* Code that passes pylint.
* Unit testing, as appropriate. In all likelihood, testing will cover individual functions, not large training cycles.
* Sufficient logging, including date, time, software (git) version, runtime folder, machine name.
* A main class containing a network that can be run separately in train/test mode.
* Visualisation with TensorBoard.
* Saving of learned network weights at the end of training.
* Loading of pre-train weights, initialising the network ready for inference.
* The ability to be run repeatedly for hyper-parameter tuning via python scripting, not bash.
* The ability to be callable from within a Jupyter Notebook, and thereby amenable to weekly writup's for supervisions.
* One or more command line programs that are pip-installable, enabling a subsequent user to train and test your algorithm with almost-zero faff.
* Visualisation for debugging purposes, such as printing example image thumbnails etc. should be done in Jupyter notebooks, or in tensorboard, not in the same class as your algorithm.

Optional features could include:

* Small test projects that train quickly to completion won't need checkpointing, but large ones will.

Developing
----------

Cloning
^^^^^^^

You can clone the repository using the following command:

::

    git clone https://github.com/UCL/scikit-surgerytorch


Running tests
^^^^^^^^^^^^^
Pytest is used for running unit tests:
::

    pip install pytest
    python -m pytest


Linting
^^^^^^^

This code conforms to the PEP8 standard. Pylint can be used to analyse the code:

::

    pip install pylint
    pylint --rcfile=tests/pylintrc sksurgerytorch


Installing
----------

You can pip install directly from the repository as follows:

::

    pip install git+https://github.com/UCL/scikit-surgerytorch



Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines`_.


Useful links
^^^^^^^^^^^^

* `Source code repository`_
* `Documentation`_


Licensing and copyright
-----------------------

Copyright 2020 University College London.
scikit-surgerytorch is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`source code repository`: https://github.com/UCL/scikit-surgerytorch
.. _`Documentation`: https://scikit-surgerytorch.readthedocs.io
.. _`scikit-surgery`: https://github.com/UCL/scikit-surgery/wiki
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://github.com/UCL/scikit-surgerytorch/blob/master/CONTRIBUTING.rst
.. _`license file`: https://github.com/UCL/scikit-surgerytorch/blob/master/LICENSE
.. _`PythonTemplate`: https://github.com/UCL/PythonTemplate

