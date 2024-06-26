# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
pylbm: tests for the geometry
"""

import pytest
import pylbm

ELEMENTS = [
    [2, pylbm.Circle([0, 0], 1)],
    [2, pylbm.Ellipse([0, 0], [1, 0], [0, 1])],
    [2, pylbm.Triangle([-1, -1], [0, 2], [2, 0])],
    [2, pylbm.Parallelogram([-1, -1], [0, 2], [2, 0])],
    [3, pylbm.CylinderCircle([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])],
    [3, pylbm.CylinderEllipse([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])],
    [3, pylbm.CylinderTriangle([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])],
    [3, pylbm.Parallelepiped([-1, -1, -1], [2, 0, 0], [0, 2, 0], [0, 0, 2])],
    [3, pylbm.Ellipsoid([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])],
    [3, pylbm.Sphere([0, 0, 0], 1)],
]


@pytest.fixture(params=ELEMENTS, ids=[elem[1].__class__.__name__ for elem in ELEMENTS])
def get_element(request):
    """get one element of the geometry"""
    return request.param


BOX = [
    (1, {"x": [-2, 2], "label": 3}),
    (2, {"x": [-2, 2], "y": [-2, 2], "label": 3}),
    (3, {"x": [-2, 2], "y": [-2, 2], "z": [-2, 2], "label": 3}),
]


@pytest.fixture(params=BOX, ids=["box1d", "box2d", "box3d"])
def get_box(request):
    """get one box"""
    return request.param


class TestGeometry:
    """
    test the geometry with pytest
    """

    def test_box_label(self, get_box):
        dim_box, box = get_box
        dico = {"box": box}
        geom = pylbm.Geometry(dico)
        assert geom.list_of_labels() == [3]

    def test_elem_label(self, get_box, get_element):
        dim_element, element = get_element
        dim_box, box = get_box
        dico = {"box": box, "elements": [element]}

        if dim_element != dim_box:
            with pytest.raises(ValueError):
                geom = pylbm.Geometry(dico)
        else:
            geom = pylbm.Geometry(dico)
            assert geom.list_of_labels() == pytest.approx([0, 3])
            assert geom.list_of_elements_labels() == pytest.approx([0])

    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_visualize(self, get_box, get_element):
        dim_element, element = get_element
        dim_box, box = get_box
        dico = {"box": box, "elements": [element]}

        if dim_element == dim_box:
            geom = pylbm.Geometry(dico)
            views = geom.visualize(viewlabel=False)
            return views.fig
        else:
            pytest.skip("incompatible dimension")
