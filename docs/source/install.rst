Installing
==========

You can install pop-tools with ``pip``, ``conda``, or by installing from source.

Pip
---

Pip can be used to install pop-tools::

   pip install pop-tools

Conda
-----

To install the latest version of pop-tools from the
`conda-forge <https://conda-forge.github.io/>`_ repository using
`conda <https://www.anaconda.com/downloads>`_::

    conda install -c conda-forge pop-tools

Install from Source
-------------------

To install pop-tools from source, clone the repository from `github
<https://github.com/NCAR/pop-tools>`_::

    git clone https://github.com/NCAR/pop-tools.git
    cd pop-tools
    pip install -e .

You can also install directly from git master branch::

    pip install git+https://github.com/NCAR/pop-tools


Test
----

To run pop-tools's tests with ``pytest``::

    git clone https://github.com/NCAR/pop-tools.git
    cd pop-tools
    pytest - v
