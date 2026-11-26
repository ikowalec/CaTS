import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

from cats.multiwfn.plot_uvvis import _autoscale_intensity_axis


def test_autoscale_intensity_axis_keeps_peak_below_top():
    _, axis = plt.subplots()
    axis.plot([300, 400, 500], [0, 25, 2])
    axis.set_ylim(0, 0.8)

    _autoscale_intensity_axis(axis)

    assert axis.get_ylim()[0] == 0
    assert axis.get_ylim()[1] > 25
    plt.close(axis.figure)
