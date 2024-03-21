
def get_figure_size(n_columns, height_factor: float = 1, one_column_size: float = 134.46 / 315.71):
    # Column widths ratios are taken from screenshot of two-column elsevier paper
    # on an A4 Page
    din_a4_width = 21 / 2.54  # in inches
    return [
        din_a4_width * one_column_size * n_columns,
        3.15 * height_factor
    ]

