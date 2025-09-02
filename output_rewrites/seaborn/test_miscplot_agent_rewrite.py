import matplotlib.pyplot as plt

from seaborn import miscplot as misc
from seaborn.palettes import color_palette
from .test_utils import _network


class TestPalPlot:
    """Test the function that visualizes a color palette."""
    def test_palplot_size(self):

        # Edge case: empty palette
        pal_empty = []
        misc.palplot(pal_empty)
        size_empty = plt.gcf().get_size_inches()
        assert tuple(size_empty) == (0, 1)

        # Edge case: large palette
        pal_large = color_palette("husl", 20)
        misc.palplot(pal_large)
        size_large = plt.gcf().get_size_inches()
        assert tuple(size_large) == (20, 1)

        # Edge case: float size input and negative number of colors (converted to 0 colors)
        pal_float = color_palette("husl", 0) # should be an empty palette
        misc.palplot(pal_float, 1.5)
        size_float = plt.gcf().get_size_inches()
        assert tuple(size_float) == (0, 1.5)

        # Edge case: single color palette and negative size input, should default size to positive
        pal_single = color_palette("husl", 1)
        # This will raise a ValueError, so let's catch it and assert that it is raised
        import pytest
        with pytest.raises(ValueError):
            misc.palplot(pal_single, -2)

        # Regular case with 6 colors and size set to a float
        pal6 = color_palette("husl", 6)
        misc.palplot(pal6, 2.5)
        size6 = plt.gcf().get_size_inches()
        assert tuple(size6) == (15, 2.5)


class TestDogPlot:

    @_network(url="https://github.com/mwaskom/seaborn-data")
    def test_dogplot(self):
        misc.dogplot()
        ax = plt.gca()
        assert len(ax.images) == 1