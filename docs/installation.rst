Installation
============

Quick Installation
-----------------

To install CamTools, simply do:

.. code-block:: bash

   pip install camtools

Installation from Source
-----------------------

Alternatively, you can install CamTools from source with one of the following
methods:

.. code-block:: bash

   git clone https://github.com/yxlao/camtools.git
   cd camtools

   # Installation mode, if you want to use camtools only.
   pip install .

   # Editable mode, if you want to modify camtools on the fly.
   pip install -e .

   # Editable mode and dev dependencies.
   pip install -e .[dev]

   # Help VSCode resolve imports when installed with editable mode.
   # https://stackoverflow.com/a/76897706/1255535
   pip install -e .[dev] --config-settings editable_mode=strict

   # Enable torch-related features (e.g. computing image metrics)
   pip install camtools[torch]

   # Enable torch-related features in editable mode
   pip install -e .[torch]
