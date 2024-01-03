import pandas as pd
import numpy as np
from hps_grid_interaction import KERBER_NETZ_XLSX, PROJECT_FOLDER


def get_modifier(
    night_start: float = 22,
    dT_set_back: float = 2
):
    def modelica_bool(dT):
        if dT > 0:
            return "true"
        return "false"
    return f"use_nigSetBac={modelica_bool(dT_set_back)}, " \
           f"houNigEnd={night_start + 8 - 24}, " \
           f"houNigStart={night_start}, " \
           f"dTNigSetBac={dT_set_back}"


def create_user_sleep_schedules():
    df = pd.read_excel(KERBER_NETZ_XLSX, index_col=0, sheet_name="Kerber Netz Neubau")
    lowest_start = 4
    highest_start = 8
    mean = (lowest_start + highest_start) / 2
    std = (mean - lowest_start) / 3
    size = len(df)
    random_samples = np.random.normal(loc=mean, scale=std, size=size)
    #random_samples = np.random.uniform(lowest_start, highest_start, size=size)
    if np.any(random_samples < lowest_start) or np.any(random_samples > highest_start):
        raise ValueError("Draw again")
    pd.DataFrame({"night_start": random_samples, "dT_set_back": np.ones(size) * 2}).to_excel(
        PROJECT_FOLDER.joinpath("Night_set_backs.xlsx"), sheet_name="users"
    )


def plot():
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")

    def plot_normal_distribution(mean, sigma, num_draws=10000):
        # Generate 1000 random draws from a normal distribution
        random_samples = np.random.normal(loc=mean, scale=sigma, size=num_draws)
        #random_samples = np.random.uniform(4, 8, size=num_draws)

        # Create a histogram of the random draws
        plt.hist(random_samples, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

        # Add labels and title
        plt.title('Histogram of Random Draws from Normal Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # Show the plot
        plt.show()

    # Set the mean and standard deviation
    mean_value = 6
    sigma_value = 2 / 3

    # Call the plot function
    plot_normal_distribution(mean_value, sigma_value)


if __name__ == '__main__':
    create_user_sleep_schedules()
