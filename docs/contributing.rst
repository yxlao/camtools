Contributing
===========

Contributing Guidelines
----------------------

- Follow `Angular's commit message convention <https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format>`_ for PRs.
  - This applies to PR's title and ultimately the commit messages in ``main``.
  - The prefix shall be one of ``build``, ``ci``, ``docs``, ``feat``, ``fix``, ``perf``, ``refactor``, ``test``.
  - Use lowercase.
- Format your code with `black <https://github.com/psf/black>`_. This will be enforced by the CI.

Building Documentation
---------------------

To build and view the documentation locally:

.. code-block:: bash

   # Build the documentation
   cd docs
   make html

   # Start a local server to view the documentation
   python -m http.server 8000 --directory _build/html

Then open your browser and navigate to ``http://localhost:8000`` to view the documentation.

Build with CamTools
-------------------

If you use CamTools in your project, consider adding one of the following
badges to your project.

.. raw:: html

   <p>
   <a href="https://github.com/yxlao/camtools"><img alt="Built with CamTools" src="https://raw.githubusercontent.com/yxlao/camtools/main/camtools/assets/built_with_camtools_dark.svg" width=240></a>
   <a href="https://github.com/yxlao/camtools"><img alt="Built with CamTools" src="https://raw.githubusercontent.com/yxlao/camtools/main/camtools/assets/built_with_camtools_light.svg" width=240></a>
   </p>
