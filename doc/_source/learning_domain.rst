The Domain of the simulation
##############################

With pyLBM, the numerical simulations can be performed in a domain
with a complex geometry.
The creation of the geometry from a dictionary is explained `here <learning_geometry.html>`_.
All the informations needed to build the domain are defined through a dictionary
and put in a object of the class :py:class:`Domain <pyLBM.domain.Domain>`.

The domain is built from three types of informations:

* a geometry (class :py:class:`Geometry <pyLBM.geometry.Geometry>`),
* a stencil (class :py:class:`Stencil <pyLBM.geometry.Stencil>`),
* a space step (a float for the grid step of the simulation).

The domain is a uniform cartesian discretization of the geometry with a grid step
:math:`dx`. The whole box is discretized even if some elements are added to reduce
the domain of the computation.
The stencil is necessary in order to know the maximal velocity in each direction
so that the corresponding number of phantom cells are added at the borders of
the domain (for the treatment of the boundary conditions).

Several examples of domains can be found in
demo/examples/domain/

Examples in 1D
******************************

:download:`script<codes/domain_1D_segment.py>`

The segment :math:`[0, 1]`
==============================

.. literalinclude:: codes/domain_1D_segment.py
    :lines: 11-

.. image:: /images/domain_1D_segment.png

The segment :math:`[0,1]` is created by the dictionary with the key ``box``.
The stencil is composed by the velocity :math:`v_0=0`, :math:`v_1=1`, and
:math:`v_2=-1`. One phantom cell is then added at the left and at the right of
the domain.
The space step :math:`dx` is taken to :math:`0.1` to allow the visualization.
The result is then visualized with the distance of the boundary points
by using the method
:py:meth:`visualize<pyLBM.geometry.Geometry.visualize>`.


Examples in 2D
******************************

The square :math:`[0,1]^2`
==============================

:download:`script<codes/geometry_2D_square_label.py>`

.. literalinclude:: codes/geometry_2D_square.py
    :lines: 11-

The square :math:`[0,1]^2` is created by the dictionary with the key ``box``.
The result is then visualized by using the method
:py:meth:`visualize <pyLBM.geometry.Geometry.visualize>`.

We then add the labels on each edge of the square
through a list of integers with the conventions:

.. hlist::
  :columns: 2

  * first for the left (:math:`x=x_{\operatorname{min}}`)
  * third for the bottom (:math:`y=y_{\operatorname{min}}`)
  * second for the right (:math:`x=x_{\operatorname{max}}`)
  * fourth for the top (:math:`y=y_{\operatorname{max}}`)

.. literalinclude:: codes/geometry_2D_square_label.py
    :lines: 11-

If all the labels have the same value, a shorter solution is to
give only the integer value of the label instead of the list.
If no labels are given in the dictionary, the default value is -1.

A square with a hole
==============================

:download:`script 1<codes/geometry_2D_square_hole.py>`
:download:`script 2<codes/geometry_2D_square_triangle.py>`
:download:`script 3<codes/geometry_2D_square_parallelogram.py>`

The unit square :math:`[0,1]^2` can be holed with a circle (script 1)
or with a triangular or with a parallelogram (script 3)

In the first example,
a solid disc lies in the fluid domain defined by
a :py:class:`circle <pyLBM.elements.Circle>`
with a center of (0.5, 0.5) and a radius of 0.125

.. literalinclude:: codes/geometry_2D_square_hole.py
    :lines: 11-

The dictionary of the geometry then contains an additional key ``elements``
that is a list of elements.
In this example, the circle is labelized by 1 while the edges of the square by 0.

The element can be also a :py:class:`triangle <pyLBM.elements.Triangle>`

.. literalinclude:: codes/geometry_2D_square_triangle.py
    :lines: 11-

or a :py:class:`parallelogram <pyLBM.elements.Parallelogram>`

.. literalinclude:: codes/geometry_2D_square_parallelogram.py
    :lines: 11-

A complex cavity
==============================

:download:`script <codes/geometry_2D_cavity.py>`

A complex geometry can be build by using a list of elements. In this example,
the box is fixed to the unit square :math:`[0,1]^2`. A square hole is added with the
argument ``isfluid=False``. A strip and a circle are then added with the argument
``isfluid=True``. Finally, a square hole is put. The value of ``elements``
contains the list of all the previous elements. Note that the order of
the elements in the list is relevant.

.. literalinclude:: codes/geometry_2D_cavity.py
    :lines: 11-19

.. image:: /images/geometry_2D_cavity_1.png

Once the geometry is built, it can be modified by adding or deleting
other elements. For instance, the four corners of the cavity can be rounded
in this way.

.. literalinclude:: codes/geometry_2D_cavity.py
    :lines: 21-

.. image:: /images/geometry_2D_cavity_2.png


Examples in 3D
******************************

The cube :math:`[0,1]^3`
==============================

:download:`script<codes/geometry_3D_cube.py>`

.. literalinclude:: codes/geometry_3D_cube.py
    :lines: 11-

The cube :math:`[0,1]^3` is created by the dictionary with the key ``box``.
The result is then visualized by using the method
:py:meth:`visualize <pyLBM.geometry.Geometry.visualize>`.

We then add the labels on each edge of the square
through a list of integers with the conventions:

.. hlist::
  :columns: 2

  * first for the left (:math:`x=x_{\operatorname{min}}`)
  * third for the bottom (:math:`y=y_{\operatorname{min}}`)
  * fifth for the front (:math:`z=z_{\operatorname{min}}`)
  * second for the right (:math:`x=x_{\operatorname{max}}`)
  * fourth for the top (:math:`y=y_{\operatorname{max}}`)
  * sixth for the back (:math:`z=z_{\operatorname{max}}`)

If all the labels have the same value, a shorter solution is to
give only the integer value of the label instead of the list.
If no labels are given in the dictionary, the default value is -1.

.. image:: /images/geometry_3D_cube.png