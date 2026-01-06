pycoretools
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


`pycoretools` is a lightweight collection of **generic, low-level
utilities** that are reused across multiple Python projects.  It is
designed to provide commonly needed building blocks without
introducing heavy dependencies or project-specific assumptions.

This package exists primarily to **break dependency cycles** between
higher-level libraries and to centralise small but widely useful
tools.


Installation
------------

Installation is easy with pip::

  pip install pycoretools

alternatively the package can be cloned from github at https://github.com/GDeLaurentis/pycoretools.


Quick start
-----------

.. code-block:: python
   :caption: Parallelise map over multiple cores.
   :linenos:
   
   from pycoretools import mapThreads

   result = mapThreads(some_function, some_iterable)


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. Hidden TOCs

.. toctree::
   :caption: Modules Documentation
   :maxdepth: 2

   modules
