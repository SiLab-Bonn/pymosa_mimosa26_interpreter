import numpy as np
from matplotlib.figure import Figure
from matplotlib.artist import setp
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors, cm
from matplotlib.backends.backend_pdf import PdfPages


def plot_fancy_occupancy(hist, title, z_max=None, filename=None):
    if z_max == 'median':
        cmap = cm.get_cmap('coolwarm')
    else:
        cmap = cm.get_cmap('viridis')
    cmap.set_bad('w')
    if z_max == 'median':
        z_max = 2 * np.ma.median(hist)
    elif z_max == 'maximum' or z_max is None:
        z_max = np.ma.max(hist)
    if z_max < 1 or hist.all() is np.ma.masked or np.allclose(0, hist):
        z_max = 1.0

    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_title(title, size=6)
    extent = [0.5, 1152.5, 576.5, 0.5]
    bounds = np.linspace(start=0, stop=z_max, num=255, endpoint=True)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    hist = np.ma.masked_equal(hist, 0)
    im = ax.imshow(hist, interpolation='none', aspect='auto', cmap=cmap, norm=norm, extent=extent)
    ax.set_ylim((576.5, 0.5))
    ax.set_xlim((0.5, 1152.5))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(ax)
    axHistx = divider.append_axes("top", 1.2, pad=0.2, sharex=ax)
    axHisty = divider.append_axes("right", 1.2, pad=0.2, sharey=ax)

    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, ticks=np.linspace(start=0, stop=z_max, num=9, endpoint=True))
    cb.set_label("#")
    # make some labels invisible
    setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(), visible=False)
    hight = np.ma.sum(hist, axis=0)

    axHistx.bar(x=range(1, 1153), height=hight, align='center', linewidth=0)
    axHistx.set_xlim((0.5, 1152.5))
    if hist.all() is np.ma.masked or np.allclose(0, hist):
        axHistx.set_ylim((0, 1))
    else:
        x_c_max = np.ceil(np.percentile(hight, 99))
        axHistx.set_ylim(0, max(1, x_c_max))
    axHistx.locator_params(axis='y', nbins=3)
    axHistx.ticklabel_format(style='sci', scilimits=(0, 4), axis='y')
    axHistx.set_ylabel('#')
    width = np.ma.sum(hist, axis=1)

    axHisty.barh(y=range(1, 577), width=width, align='center', linewidth=0)
    axHisty.set_ylim((576.5, 0.5))
    if hist.all() is np.ma.masked or np.allclose(0, hist):
        axHisty.set_xlim((0, 1))
    else:
        y_c_max = np.ceil(np.percentile(width, 99))
        axHisty.set_xlim(0, max(1, y_c_max))
    axHisty.locator_params(axis='x', nbins=3)
    axHisty.ticklabel_format(style='sci', scilimits=(0, 4), axis='x')
    axHisty.set_xlabel('#')

    if not filename:
        fig.show()
    elif isinstance(filename, PdfPages):
        filename.savefig(fig)
    else:
        fig.savefig(filename)


def _plot_1d_hist(hist, yerr=None, title=None, x_axis_title=None, y_axis_title=None, x_ticks=None, color='r', plot_range=None, log_y=False, filename=None):
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)

    hist = np.array(hist)
    if plot_range is None:
        plot_range = range(0, len(hist))
    plot_range = np.array(plot_range)
    plot_range = plot_range[plot_range < len(hist)]
    if yerr is not None:
        ax.bar(x=plot_range, height=hist[plot_range],
               color=color, align='center', yerr=yerr)
    else:
        ax.bar(x=plot_range,
               height=hist[plot_range], color=color, align='center')
    ax.set_xlim((min(plot_range) - 0.5, max(plot_range) + 0.5))

    ax.set_title(title, color='red')
    if x_axis_title is not None:
        ax.set_xlabel(x_axis_title)
    if y_axis_title is not None:
        ax.set_ylabel(y_axis_title)
    if x_ticks is not None:
        ax.set_xticks(plot_range)
        ax.set_xticklabels(x_ticks)
        ax.tick_params(which='both', labelsize=8)
    if np.allclose(hist, 0.0):
        ax.set_ylim((0, 1))
    else:
        if log_y:
            ax.set_yscale('log')
            ax.set_ylim((1e-1, np.amax(hist) * 2))
    ax.grid(True)

    if not filename:
        fig.show()
    elif isinstance(filename, PdfPages):
        filename.savefig(fig)
    else:
        fig.savefig(filename)


def plot_event_status(hist, title=None, filename=None):
    _plot_1d_hist(
        hist=hist,
        title=title,
        y_axis_title='Number of events',
        log_y=True,
        x_ticks=('Trigger\nerror', 'No\ntrigger\nword', 'Trigger\ntimestamp\nOVF', 'Trigger\nnumber\nOVF', 'M26\ndata\nerror', 'M26\ntimestamp\nOVF', 'M26\nframe ID\nOVF', 'M26\nrow OVF\nflag'),
        color='g',
        plot_range=range(8),
        filename=filename)
