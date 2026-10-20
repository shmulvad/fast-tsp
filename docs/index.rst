.. role:: pyt(code)
   :language: python

fast-tsp
============================

A fast local-search solver for the TSP (Travelling Salesman Problem), written in Rust with Python bindings.
Give it a symmetric integer distance matrix and a time budget, and it returns a near-optimal tour:
optimal on most TSPLIB instances below 1000 nodes within a second or two, and within a few tenths
of a percent of the optimum on instances of several thousand nodes.

.. image:: https://readthedocs.org/projects/fast-tsp/badge/?version=latest
   :target: https://fast-tsp.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/shmulvad/fast-tsp/workflows/Pip/badge.svg
   :target: https://github.com/shmulvad/fast-tsp/actions?query=workflow%3APip
   :alt: Pip Actions Status

.. image:: https://img.shields.io/pypi/v/fast-tsp
   :target: https://pypi.org/project/fast-tsp/
   :alt: PyPI - Version

.. image:: https://img.shields.io/badge/license-MIT-success?style=flat-square
   :target: https://github.com/shmulvad/fast-tsp/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/issues/shmulvad/fast-tsp?style=flat-square
   :target: https://github.com/shmulvad/fast-tsp/issues
   :alt: Issues


Quickstart
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
:pyt:`find_tour` takes a time budget (``duration_seconds``, default 2), a thread count
(``max_threads``) and two tuning options; see :doc:`usage`.

Contents
##########

.. toctree::
   :maxdepth: 2

   about
   usage
   fast_tsp
   comparisons
