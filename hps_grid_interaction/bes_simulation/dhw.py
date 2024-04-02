import logging
from pathlib import Path

import pandas as pd

from ebcpy import TimeSeriesData
from ebcpy.utils.conversion import convert_tsd_to_modelica_txt

from pydantic import BaseModel, FilePath, Field


class DHWCalcConfig(BaseModel):
    daily_volume: float
    time_step: int = 60
    TDHW_hot: float = 40  # °C
    TDHW_cold: float = 10  # °C
    path: FilePath = None
    tCrit: float = Field(
        default=None,
        title="Critical time according to EN 15450 in h"
    )
    QCrit: float = Field(
        default=None,
        title="Energy demand during critical time according to EN 15450 in kWh"
    )

    def get_name(self):
        return "_".join([str(s) for s in [self.daily_volume, self.time_step, self.TDHW_hot, self.TDHW_hot]])

    def get_modelica_modifier(self):
        fine_tuning_modifiers = dict(
            tCrit=self.tCrit,
            QCrit=self.QCrit
        )
        fine_tuning_modifier = "\n".join(
            [f"    {key}={value},"
             for key, value in fine_tuning_modifiers.items()
             if value is not None]
        )
        m_flow_nominal = pd.read_csv(self.path, skiprows=2, sep="\t").loc[:, "2_m_flow_"].max()
        path = str(self.path).replace("\\", "//")
        return f"DHW(tableName=\"DHWCalc\",\n" \
               f"    {fine_tuning_modifier}" \
               f"    fileName=Modelica.Utilities.Files.loadResource(\"{path}\"),\n" \
               f"    VDHWDay={self.daily_volume / 1000},\n" \
               f"    mDHW_flow_nominal={m_flow_nominal})"


def generate_dhw_calc_tapping(save_path_file: Path, config: DHWCalcConfig, with_plot: bool = False):
    from OpenDHW import OpenDHW
    # generate time-series with OpenDHW
    timeseries_df = OpenDHW.generate_dhw_profile(
        s_step=config.time_step,
        categories=1,
        mean_drawoff_vol_per_day=config.daily_volume,
    )
    # Compute Heat from Water TimeSeries
    timeseries_df = OpenDHW.compute_heat(
        timeseries_df=timeseries_df,
        temp_dT=config.TDHW_hot - config.TDHW_cold
    )

    tsd = TimeSeriesData(timeseries_df.copy())
    tsd.to_float_index()

    tsd["2_m_flow"] = tsd["Water_LperH"] / 3600
    tsd["1_Q_flow"] = tsd["2_m_flow"] * 4184 * (config.TDHW_hot - config.TDHW_cold)
    tsd["3_T_set"] = config.TDHW_hot
    tsd["4_T_set"] = config.TDHW_hot
    tsd = tsd[["1_Q_flow", "2_m_flow", "3_T_set", "4_T_set"]]
    tsd = tsd.sort_index(axis=1)
    mass_flow_rate = tsd.to_df().copy()["2_m_flow"]
    tsd = tsd[mass_flow_rate.diff() != 0]  # Filter all but ramps

    convert_tsd_to_modelica_txt(
        tsd=tsd,
        table_name="DHWCalc",
        save_path_file=save_path_file
    )
    logging.info("Deviation between requested and actual daily DHW tapping: %s l",
                 round(config.daily_volume - sum(tsd["2_m_flow"] * config.time_step) / 365, 2))

    if with_plot:
        start_plot = '2019-03-31-00'
        end_plot = '2019-04-01-00'

        # Generate Histogram from the loaded timeseries
        OpenDHW.draw_histplot(timeseries_df=timeseries_df, extra_kde=False,
                              save_fig=True)

        # Generate Lineplot from the loaded timeseries
        OpenDHW.draw_lineplot(timeseries_df=timeseries_df, start_plot=start_plot,
                              end_plot=end_plot, save_fig=True)


def create_dhw_profiles():
    from hps_grid_interaction import PROJECT_FOLDER, KERBER_NETZ_XLSX
    import logging
    logging.basicConfig(level="INFO")
    path = PROJECT_FOLDER.joinpath("00_new_dhw_tappings")
    os.makedirs(path, exist_ok=True)
    df = pd.read_excel(
        KERBER_NETZ_XLSX,
        index_col=0,
        sheet_name="Kerber Netz Neubau"
    )
    dhw_litre_per_person_per_day = 25

    for idx, row in df.iterrows():
        generate_dhw_calc_tapping(
            save_path_file=path.joinpath(f"DHWCalc_{idx}.txt"),
            config=DHWCalcConfig(daily_volume=row["Anzahl Bewohner"] * dhw_litre_per_person_per_day, time_step=TIME_STEP)
        )

