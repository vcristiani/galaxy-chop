==========================
Installation
==========================

You can also `install Galaxychop from PyPI`_ using pip:

.. code:: bash

   $ pip install galaxychop

Finally, you can also install the latest development version of
Galaxychop `directly from GitHub`_:

.. code:: bash

   $ pip install git+https://github.com/vcristiani/galaxy-chop.git

This is useful if there is some feature that you want to try, but we did
not release it yet as a stable version. Although you might find some
unpolished details, these development installations should work without
problems. If you find any, please open an issue in the `issue tracker`_.

.. warning::

   It is recommended that you
   **never ever use sudo** with distutils, pip, setuptools and friends in Linux
   because you might seriously break your system
   [`1 <http://wiki.python.org/moin/CheeseShopTutorial#Distutils_Installation>`_]
   [`2 <http://stackoverflow.com/questions/4314376/how-can-i-install-a-python-egg-file/4314446#comment4690673_4314446>`_]
   [`3 <http://workaround.org/easy-install-debian>`_]
   [`4 <http://matplotlib.1069221.n5.nabble.com/Why-is-pip-not-mentioned-in-the-Installation-Documentation-tp39779p39812.html)>`_].
   Use `virtual environments <https://docs.python.org/3/library/venv.html>`_ instead.

.. _conda: https://conda.io/docs/
.. _mamba: https://mamba.readthedocs.io/
.. _issue tracker: https://github.com/vcristiani/galaxy-chop/issues
.. _install GalaxyChop from PyPI: https://pypi.python.org/pypi/galaxychop/
.. _directly from GitHub: https://github.com/vcristiani/galaxy-chop


If you don't have Python
-------------------------

If you don't already have a python installation with numpy and scipy, we
recommend to install either via your package manager or via a python bundle.
These come with numpy, scipy, matplotlib and many other helpful
scientific and data processing libraries.

`Canopy
<https://www.enthought.com/products/canopy>`_ and `Anaconda
<https://www.continuum.io/downloads>`_ both ship a recent
version of Python, in addition to a large set of scientific python
library for Windows, Mac OSX and Linux.