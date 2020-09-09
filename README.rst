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

.. features-start

The aim of scikit-surgery torch is to provide a home for various pytorch models/examples/utilities that may be useful for Image Guided Surgery.

Features
--------
Implemented models:

* `High Resolution Stereo network <https://github.com/gengshan-y/high-res-stereo>`_ Inference only, see author's repo for pre trained weights. As at commit `aae0b9b <https://github.com/gengshan-y/high-res-stereo/tree/aae0b9b86c4ab007f83ed0f583f9ed7ff4b032ea>`_.
* `Volume2SurfaceCNN <https://gitlab.com/nct_tso_public/Volume2SurfaceCNN>`_ Inferencece only, see author's repo for pre trained weights. As at commit `5a656381 <https://gitlab.com/nct_tso_public/Volume2SurfaceCNN/-/tree/5a656381a162b5b37ef2eeb7b715d1e3cfbb9bf4>`_.
* Models can run on GPU or CPU.
* Example usage in `tests/`.

.. features-end

scikit-surgerytorch is NOT meant to be a layer on-top of pytorch
or provide a new kind-of platform. The aim is that researchers can learn from examples,
and importantly, learn how to deliver an algorithm that can be used by other people
out of the box, with just a ```pip install```, rather than a new user having to
re-implement stuff, or struggle to get someone else's code running.



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

