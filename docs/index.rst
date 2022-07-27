fast-tsp
============================

An SLS (Stochastic Local Search) solver for TSP (Travelling Salesman Problem).

.. image:: https://readthedocs.org/projects/fast-tsp/badge/?version=latest
   :target: https://fast-tsp.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/shmulvad/fast-tsp/workflows/Pip/badge.svg
   :target: https://github.com/shmulvad/fast-tsp/actions?query=workflow%3APip
   :alt: Pip Actions Status

.. image:: https://img.shields.io/pypi/v/fast-tsp
   :target: https://pypi.org/project/fast-tsp/
   :alt: PyPI - Version

.. image:: https://img.shields.io/badge/license-BSD-success?style=flat-square
   :target: https://github.com/shmulvad/fast-tsp/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/issues/shmulvad/fast-tsp?style=flat-square
   :target: https://github.com/shmulvad/fast-tsp/issues
   :alt: Issues


Quickstart:
#############

Install the module:

.. code-block:: bash

   $ pip install fast-tsp


Create a distance matrix and run the solver:

.. code-block:: python

   import fast_tsp
   dists = [
      [ 0, 63, 72, 70],
      [63,  0, 57, 53],
      [72, 57,  0,  4],
      [70, 53,  4,  0],
   ]
   tour = fast_tsp.find_tour(dists)
   print(tour)
   # [0, 1, 3, 2]

The returned tour is the list of indices of the nodes in the order they are visited.

Contents:
##########

.. toctree::
   :maxdepth: 2

   about
   usage
   fast_tsp
   comparisons
