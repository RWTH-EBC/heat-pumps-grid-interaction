import pathlib
from typing import List

import matplotlib.pyplot as plt


def add_image_and_text_as_label(
        ax: plt.axes,
        which_axis: str,
        technology: str,
        quotas: list, width: float,
        distance_to_others: float = 0.001,
        with_text: bool = False):
    if which_axis == "x":
        ax.set_xticklabels(quotas)
        window = ax.set_xlabel(technology if with_text else "A")
        bbox_label = window.get_window_extent().transformed(ax.transAxes.inverted())
        x_pos = bbox_label.x1 + distance_to_others * width if with_text else 0.5 - width / 2
        bbox = (
            x_pos,
            bbox_label.y0 - (distance_to_others + width) / 2,
            width, width
        )
        if not with_text:
            ax.set_xlabel("")
    else:
        ax.set_yticklabels(quotas)
        window = ax.set_ylabel(technology if with_text else "A")
        bbox_label = window.get_window_extent().transformed(ax.transAxes.inverted())
        y_pos = bbox_label.y1 + distance_to_others * width if with_text else 0.5 - width / 2
        bbox = (
            bbox_label.x0 - (distance_to_others + width) / 2,
            y_pos,
            width, width
        )
        if not with_text:
            ax.set_ylabel("")
    _add_single_image(ax=ax, bbox=bbox, image_path=get_technology_image_path(technology))


def add_images_to_title(
        technologies: List[str],
        ax: plt.axes,
        width: float,
        distance_to_others: float = 0.001
):
    n_images = len(technologies)
    assert width * n_images < 1, (f"Images won't fit, width should be "
                                  f"smaller than 1 but is {width * n_images}")
    y_start = 1 + distance_to_others * width
    x_start = 0.5 - (width + distance_to_others) * n_images / 2
    for idx, tech in enumerate(technologies):
        x_curr = x_start + width * (1 + distance_to_others) * idx
        _add_single_image(ax=ax, bbox=(x_curr, y_start, width, width), image_path=get_technology_image_path(tech))
        ax.text(x_start, y_start + width / 2, "Fixed:", transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='center')


def _add_single_image(ax: plt.axes, bbox: tuple, image_path: pathlib.Path):
    from matplotlib.image import BboxImage, imread
    from matplotlib.transforms import Bbox, TransformedBbox

    bbox0 = Bbox.from_bounds(*bbox)
    # use the `ax.transData` transform to tell the bounding box we have given
    # it position + size in data.  If you want to specify in Axes fraction
    # use ax.transAxes
    bbox = TransformedBbox(bbox0, ax.transAxes)
    bbox_image = BboxImage(
        bbox,
        norm=None,
        origin=None,
        clip_on=False,
    )
    bbox_image.set_data(imread(image_path))
    ax.add_artist(bbox_image)


def get_technology_image_path(technology: str) -> pathlib.Path:
    file_map = {
        "average": "BLDG_4.png",
        "adv_retrofit": "BLDG_box.png",
        "no_retrofit": "BLDG_3.png",
        "retrofit": "BLDG_2.png",
        "p_ret": "BLDG_2.png",
        "p_adv_ret": "BLDG_box.png",
        "gas": "BOI_box.png",
        "heating_rod": "heating_rod.png",
        "hybrid": "hybrid.png",
        "heat_pump": "HP_AS_box.png",
        "household": "Electricity_box.png",
        "pv": "PV_box.png",
        "battery": "BAT_box.png",
        "e_mobility": "CAR_box.png",
    }
    from hps_grid_interaction import DATA_PATH
    return DATA_PATH.joinpath("icons", file_map[technology])


def scale_ticks_to_axes(ticks: list, lim: list):
    return [(tick - lim[0]) / (lim[1] - lim[0]) for tick in ticks]


def add_images_to_axis(
        technologies: List[List[str]],
        ax: plt.axes,
        width: float,
        which_axis: str,
        distance_to_others: float = 0.001
):
    if which_axis == "y":
        ticks = ax.get_yticks()
        lim = ax.get_ylim()
    elif which_axis == "x":
        ticks = ax.get_xticks()
        lim = ax.get_xlim()
    else:
        raise ValueError("Only x and y axis are supported ")
    i_tick = 0
    assert len(technologies) == len(ticks), "Length of images and ticks does not match"
    for tick in scale_ticks_to_axes(ticks, lim):
        for idx, tech in enumerate(technologies[i_tick]):
            if which_axis == "y":
                y_start = tick - width / 2
                x_start = 0 - (width + distance_to_others) * idx - distance_to_others * width
                bbox = (x_start - width, y_start, width, width)
            else:
                x_start = tick - width / 2
                y_start = 0 - (width + distance_to_others) * idx - distance_to_others * width
                bbox = (x_start, y_start - width, width, width)
            _add_single_image(ax=ax, bbox=bbox, image_path=get_technology_image_path(tech))
        i_tick += 1

    if which_axis == "y":
        ax.set_yticklabels([])
        ax.set_yticks([])
    elif which_axis == "x":
        ax.set_xticklabels([])
        ax.set_xticks([])
