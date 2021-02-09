Requirements
============

You need Python 3.7 or 3.8 to run GalaxyChop.


Installation
============


This is the recommended way to install GalaxyChop.

Installing  with pip
^^^^^^^^^^^^^^^^^^^^

Make sure that the Python interpreter can load GalaxyChop code.
The most convenient way to do this is to use virtualenv, virtualenvwrapper, and pip.

After setting up and activating the virtualenv, run the following command:

.. code-block:: console

   $ pip install galaxychop

That should be it all.



Installing the development version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you’d like to be able to update your GalaxyChop code occasionally with the
latest improvements and bug fixes, follow these instructions:

Make sure that you have Git installed and that you can run its commands from a shell.
(Enter *git help* at a shell prompt to test this.)

Check out GalaxyChop main development branch like so:

.. code-block:: console

   $ git clone https://github.com/vcristiani/galaxy-chop.git
   

This will create a directory *galaxy-chop* in your current directory.

Then you can proceed to install with the commands

.. code-block:: console

   $ cd GalaxyChop
   $ pip install -e .
