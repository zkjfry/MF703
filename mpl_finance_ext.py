import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import BoxStyle
from six.moves import xrange, zip

from angled_box_style import AngledBoxStyle
from candlestick_pattern_evaluation import draw_pattern_evaluation
from signal_evaluation import draw_signal_evaluation
from signal_evaluation import draw_verticals

# Colors:
label_colors = '#c1c1c1'
background_color = '#ffffff'

red = '#c2c2c2'     # '#fe0000'
green = '#13bebc'   # '#00fc01'

color_set = ['#13bebc', '#b0c113', '#c1139e', '#c17113', '#0d8382']
# Create angled box style
BoxStyle._style_list["angled"] = AngledBoxStyle


def _tail(fig, ax, kwa, data=None, plot_columns=None):

    # Vertical span and lines:
    vline = kwa.get('vline', None)
    if vline is not None:
        plot_vline(
            axis=ax, index=vline
        )

    vspan = kwa.get('vspan', None)
    if vspan is not None:
        plot_vspan(
            axis=ax, index=vspan
        )

    # Names, title, labels
    name = kwa.get('name', None)
    if name is not None:
        ax.text(
            0.5, 0.95, name, color=label_colors,
            horizontalalignment='center',
            fontsize=10, transform=ax.transAxes,
            zorder=120
        )

    xlabel = kwa.get('xlabel', None)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ylabel = kwa.get('ylabel', None)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    title = kwa.get('title', None)
    if title is not None:
        ax.set_title(title)

    # Plot columns
    enable_flags = kwa.get('enable_flags', True)
    if kwa.get('set_flags_at_the_end', True):
        last_index = data.index.values[-1]
    else:
        last_index = None

    if plot_columns is not None and data is not None:
        for i, col in enumerate(plot_columns):
            series = data[col]
            ax.plot(series, linewidth=0.7,
                    color=color_set[i])
            if enable_flags:
                add_price_flag(
                    fig=fig, axis=ax,
                    series=data[col],
                    color=color_set[i],
                    last_index=last_index
                )

    xhline = kwa.get('xhline1', None)
    if xhline is not None:
        ax.axhline(xhline, color=label_colors,
                   linewidth=0.5)

    xhline2 = kwa.get('xhline2', None)
    if xhline2 is not None:
        ax.axhline(xhline2, color=label_colors,
                   linewidth=0.5)

    xhline_red = kwa.get('xhline_red', None)
    if xhline_red is not None:
        ax.axhline(xhline_red, color=red,
                   linewidth=0.5)

    xhline_green = kwa.get('xhline_green', None)
    if xhline_green is not None:
        ax.axhline(xhline_green, color=green,
                   linewidth=0.5)

    xhline_dashed_1 = kwa.get('xhline_dashed1', None)
    if xhline_dashed_1 is not None:
        ax.axhline(xhline_dashed_1, color=label_colors,
                   linewidth=0.6, linestyle='--')

    xhline_dashed_2 = kwa.get('xhline_dashed2', None)
    if xhline_dashed_2 is not None:
        ax.axhline(xhline_dashed_2, color=label_colors,
                   linewidth=0.6, linestyle='--')

    xhline_dotted_1 = kwa.get('xhline_dotted1', None)
    if xhline_dotted_1 is not None:
        ax.axhline(xhline_dotted_1, color=label_colors,
                   linewidth=0.9, linestyle=':')

    xhline_dotted_2 = kwa.get('xhline_dotted2', None)
    if xhline_dotted_2 is not None:
        ax.axhline(xhline_dotted_2, color=label_colors,
                   linewidth=0.9, linestyle=':')

    main_spine = kwa.get('main_spine', 'left')
    fancy_design(ax, main_spine=main_spine)
    rotation = kwa.get('xtickrotation', 35)
    plt.setp(ax.get_xticklabels(), rotation=rotation)
    if kwa.get('disable_x_ticks', False):
        # Deactivates labels always for all shared axes
        labels = [
            item.get_text()
            for item in ax.get_xticklabels()
        ]
        ax.set_xticklabels([''] * len(labels))

    save = kwa.get('save', '')
    if save:
        plt.savefig(save, facecolor=fig.get_facecolor())

    if kwa.get('axis', None) is None and \
            kwa.get('show', True):
        plt.show()
    return fig, ax


def _head(kwargs, data=None):
    # Prepare data ------------------------------------------
    if data is not None:
        for col in list(data):
            data[col] = pd.to_numeric(
                data[col], errors='coerce')

    # Build ax ----------------------------------------------
    fig = kwargs.get('fig', None)
    if fig is None:
        fig, _ = plt.subplots(facecolor=background_color)

    ax = kwargs.get('axis', None)
    if ax is None:
        ax = plt.subplot2grid(
            (4, 4), (0, 0),
            rowspan=4, colspan=4,
            facecolor=background_color
        )
    return fig, ax


def fancy_design(axis, main_spine='left'):
    """
    This function changes the design for
        - the legend
        - spines
        - ticks
        - grid
    :param axis: Axis
    """
    legend = axis.legend(
        loc='best', fancybox=True, framealpha=0.3
    )

    legend.get_frame().set_facecolor(background_color)
    legend.get_frame().set_edgecolor(label_colors)

    for line, text in zip(legend.get_lines(),
                          legend.get_texts()):
        text.set_color(line.get_color())

    axis.grid(linestyle='dotted',
              color=label_colors, alpha=0.7)
    axis.yaxis.label.set_color(label_colors)
    axis.xaxis.label.set_color(label_colors)
    axis.yaxis.label.set_color(label_colors)
    for spine in axis.spines:
        if spine == main_spine:
            axis.spines[spine].set_color(label_colors)
        else:
            axis.spines[spine].set_color(background_color)
    axis.tick_params(
        axis='y', colors=label_colors,
        which='major', labelsize=10,
        direction='in', length=2,
        width=1
    )

    axis.tick_params(
        axis='x', colors=label_colors,
        which='major', labelsize=10,
        direction='in', length=2,
        width=1
    )


def add_price_flag(fig, axis, series, color, last_index=None):
    """
    Add a price flag at the end of the data
    series in the chart
    :param fig: Figure
    :param axis: Axis
    :param series: Pandas Series
    :param color: Color of the flag
    :param last_index: Last index
    """

    series = series.dropna()
    value = series.tail(1)

    index = value.index.tolist()[0]
    if last_index is not None:
        axis.plot(
            [index, last_index], [value.values[0], value.values[0]],
            color=color, linewidth=0.6, linestyle='--', alpha=0.6
        )
    else:
        last_index = index

    trans_offset = mtrans.offset_copy(
        axis.transData, fig=fig,
        x=0.05, y=0.0, units='inches'
    )

    # Add price text box for candlestick
    value_clean = format(value.values[0], '.6f')
    axis.text(
        last_index, value.values, value_clean,
        size=7, va="center", ha="left",
        transform=trans_offset,
        color='white',
        bbox=dict(
            boxstyle="angled,pad=0.2",
            alpha=0.6, color=color
        )
    )

def plot(data, plot_columns, **kwargs):
    """
    This function provides a simple way to plot time series
    for example data['close'].
    :param data: Pandas DataFrame object
    :param plot_columns: Name of the columns to plot
    :param kwargs:
        'fig': Figure.
        'axis': Axis. If axis is not given the chart will
            plt.plot automatically
        'name': Name of the chart
        'enable_flags': Enable flags
        'set_flags_at_the_end': Set flags at the end of the chart
        'xhline1': Normal horizontal line 1
        'xhline2': Normal horizontal line 1
        'xhline_red': Red horizontal line
        'xhline_green': Green horizontal line
        'xhline_dashed1': Dashed horizontal line 1
        'xhline_dashed2': Dashed horizontal line 2
        'xhline_dotted1': Dotted horizontal line 1
        'xhline_dotted2': Dotted horizontal line 2
        'vline': Index of vline
        'vspan': [start index, end index]
        'xlabel': x label
        'ylabel': x label
        'title': title
        'disable_x_ticks': Disables the x ticks
        'show': If true the chart will be plt.show'd
        'save': Save the image to a specified path like
            save='path_to_picture.png'
    :return: fig, ax
    """
    fig, ax = _head(kwargs=kwargs, data=data)

    return _tail(
        fig=fig,
        ax=ax,
        kwa=kwargs,
        data=data,
        plot_columns=plot_columns
    )


def plot_vline(axis, index, linestyle='--', color=color_set[0]):
    """
    Plots a vertical line
    :param axis: Axis
    :param index: Index
    :param linestyle: Can be '-', '--', '-.', ':'
    :param color: Color
    """
    axis.axvline(
        index, color=color,
        linewidth=0.8, alpha=0.8, linestyle=linestyle
    )


def plot_vspan(axis, index, color=color_set[0], alpha=0.05):
    """
    Plots a vertical span
    :param axis: Axis
    :param index: [start index, end index]
    :param color: Color
    :param alpha: Alpha
    :return:
    """
    axis.axvspan(
        index[0], index[1],
        facecolor=color,
        alpha=alpha
    )
