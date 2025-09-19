import requests
from local_config import apptoken, apptoken_secret
import os
import polars as pl
import plotly.express as px
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from typing import Literal
import plotly.io as pio
from plotly import graph_objects

# apptoken for reading in open data
os.environ["APPTOKEN"] = apptoken
os.environ["apptoken_secret"] = apptoken_secret

# set plotly express settings to open in browser window
pio.renderers.default = "browser"


def px_configure(
    plot: "px.chart",
) -> "px.Chart":
    """Default configuration settings for plotly charts

    params
    ------
        plot (px.chart): Plotly chart object (line, bar, etc)


    returns
    -------
    'px.Chart' with x-axis, y-axis, layout, and traces added in

    """

    # don't want decimal places showing in percentage y-axis - even if it shows in the hover text

    plot = (
        plot.update_traces(
            hovertemplate=(
                "Stops: %{customdata[0]}<br>"
                "Trips: %{customdata[1]}<br>"
                "%{y}"
            ),
            hoverinfo="skip",
            marker=dict(
                line=dict(
                    color="black",
                    width=0.05
                )
            )
        )
        .update_xaxes(
            # controls display of vertical lines in background at each xtick mark
            showgrid=True,
            # controls display of line at bottom of graph
            showline=True,
            # color of line at bottom of graph, denoting xaxis
            linecolor="#888888",
            # color of tick marks denoting bottom of x axis
            tickcolor="#888888",  # FC0000
            # color of faded vertical lines in background at each xtick
            gridcolor="#ebebeb",
            tick0=0,  # set where the first tick appars on the x-axis
            # makes a parallel line appear at the top of the graph
            mirror=True,
            title_font=dict(
                size=12,  # size of y-axis label font
                family="Helvetica",  # font style of y-axis label font
            ),
        )
        .update_yaxes(
            linecolor="#888888",  # color of line alongside y-axis, denoting yaxis
            tickcolor="#888888",  # color of tick marks along y-axis
            gridcolor="#ebebeb",  # color of horizontal lines in background at each ytick
            tickformat=",.0%",
            range=[0, 1],
            mirror=True,
            showgrid=True,
            showline=True,
            title_font=dict(
                size=12,  # size of y-axis label font
                family="Helvetica",  # font style of y-axis label font
            ),
        )
        .for_each_xaxis(lambda axis: axis.update(title_text=""))
        .for_each_yaxis(lambda axis: axis.update(title_text=""))
        .update_layout(
            xaxis_title=dict(
                    text="Car order",
                    font=dict(size=12, family="Helvetica"),
            ),
            yaxis_title=dict(
                    text="Proportion full",
                    font=dict(size=12, family="Helvetica"),
            ),
            # https://plotly.com/python/reference/layout/
            plot_bgcolor='white',
            paper_bgcolor="white",
            # plot_bgcolor="rgba(0, 0, 0, 0)",  # controls display of background within chart-box
            # paper_bgcolor="rgba(0, 0, 0, 0)",  # controls display of background including chart-box
            coloraxis_colorbar=dict(
                title=dict(
                    text="Percent full",
                    font=dict(size=12, family="Helvetica"),
                ),
                thickness=10,
                len=0.5,
                y=0.5,
                yanchor="bottom",
            ),
            font=dict(family="Helvetica", size=12),
            autosize=True,  # i think, adjust size of chart automatically
            margin=dict(
                l=20, r=0, b=0, t=20, pad=0
            ),  # controls position of chart relative to box set aside for it
        )
    )

    return plot


def establish_nys_session(
    retries: Retry | None | str = "default", proxy_obj: dict = None
):
    """Initializes a session using the environmental variable APPTOKEN; for communication with NYS Open Data Portal"""

    session = requests.Session()

    session.headers.update({"access_token": os.environ["APPTOKEN"]})

    if retries is not None:

        if retries == "default":

            retries = Retry(total=7, backoff_factor=0.1)

        session.mount("https://", HTTPAdapter(max_retries=retries))

    if proxy_obj is not None:

        session.proxies.update(proxy_obj)

    return session

def get_sessiondata(
    data_url: str,
    _session: requests.Session | None | str,
    nys_params: dict = None,
    timeout: int = 15,
    **kwargs,
) -> "session.content":
    """Establishes session with NYS open data portal, and requests content of a dataset

    Params:

        data_url (str): An NYS open dataset API or 'Export' url to read in

        _session (requests.Session, None, str): A session object. If None, runs `establish_nys_session`. If 'session_state', returns session object stored in `st.session_state`

        nys_params (None , dict): Optional set of parameters to feed to `params` argument of `session.get`, for a SODA API data_url. Default None

        timeout (int): How long to wait before connection times out. Default 10

        last_updated (dt.datetime, None): Last updated date of dataset, for cacheing purposes. In `load_data`, provided by `get_dateupdated`. Default None

        **kwargs: Additional keyword specific arguments to `establish_nys_session` (if session is None) and to `session.get` method

    Returns:

        Content of a dataset from a requests package sesssion.get command
    """

    if _session == "session_state":
        _session = st.session_state["nys_session"]

    if _session is None:
        _session = establish_nys_session(**kwargs)

    url_data = _session.get(
        data_url, params=nys_params, timeout=timeout, **kwargs
    ).content

    return url_data


# API for trip occupancy dataset
lirr_trip = "https://data.ny.gov/resource/73th-g5ad.json"

# aggregate trip occupancy for 2024
trip_occpancy_overall = """select 
    count(trip_id),
    sum(case(max_occupancy >= 0.0 and max_occupancy < 0.1, 1)) as trips_0_10,
    sum(case(max_occupancy >= 0.10 and max_occupancy < 0.2, 1)) as trips_10_20,
    sum(case(max_occupancy >= 0.20 and max_occupancy < 0.3, 1)) as trips_20_30,
    sum(case(max_occupancy >= 0.30 and max_occupancy < 0.4, 1)) as trips_30_40,
    sum(case(max_occupancy >= 0.40 and max_occupancy < 0.5, 1)) as trips_40_50,
    sum(case(max_occupancy >= 0.50 and max_occupancy < 0.6, 1)) as trips_50_60,
    sum(case(max_occupancy >= 0.60 and max_occupancy < 0.7, 1)) as trips_60_70,
    sum(case(max_occupancy >= 0.70 and max_occupancy < 0.8, 1)) as trips_70_80,
    sum(case(max_occupancy >= 0.80 and max_occupancy < 0.9, 1)) as trips_80_90,
    sum(case(max_occupancy >= 0.90 and max_occupancy < 1.0, 1)) as trips_90_100,
    sum(case(max_occupancy >= 1.0, 1)) as trips_100plus,
    sum(case(max_occupancy >= 0.5, 1)) as trips_50,
    sum(case(max_occupancy >= 0.6, 1)) as trips_60,
    sum(case(max_occupancy >= 0.7, 1)) as trips_70,
    sum(case(max_occupancy >= 0.8, 1)) as trips_80,
    sum(case(max_occupancy >= 0.9, 1)) as trips_90,
    sum(case(max_occupancy >= 0.5, 1)) * 100 / count(trip_id) as pct_trips_50,
    sum(case(max_occupancy >= 0.6, 1)) * 100 / count(trip_id) as pct_trips_60,
    sum(case(max_occupancy >= 0.7, 1)) * 100 / count(trip_id) as pct_trips_70,
    sum(case(max_occupancy >= 0.8, 1)) * 100 / count(trip_id) as pct_trips_80,
    sum(case(max_occupancy >= 0.9, 1)) * 100 / count(trip_id) as pct_trips_90
    where max_passengers is not null and service_date between '2024-01-01' and '2024-12-31' LIMIT 10000000"""

session = establish_nys_session()

# read in trip data
trip_overall = pl.read_json(
    get_sessiondata(
        lirr_trip,
        session,
        nys_params={"$query": trip_occpancy_overall},
    )
)

# all columns should be float type
trip_overall = trip_overall.select(
    *[pl.col(col).cast(pl.Float64) for col in trip_overall.columns]
)

# create data for showing trips by bucketed max occupancy
trip_bucket = trip_overall.select(
    trip_overall.columns[1:12]
).unpivot(
    value_name="Trips",
    variable_name="Max occupancy %"
).with_columns(
    pl.col("Trips").cast(pl.Int32),
    pl.col("Max occupancy %").str.replace(
        "trips_",
        ""
    ).str.replace(
        "_",
        "-"
    ).str.replace(
        "100plus",
        "100+"
    )
).with_columns(
    pl.col("Trips").sum().alias("total")
).with_columns(
    (pl.col("Trips") * 100 / pl.col("total")).alias("pct")
)

# Create chart - better to have parameterized px_configure above, but for convenience just using custom chart settings
(
    px.bar(
        trip_bucket,
        x="Max occupancy %",
        y="Trips",
        color_discrete_sequence=["#0D2A63"]
    ).update_xaxes(
        # controls display of vertical lines in background at each xtick mark
        showgrid=True,
        # controls display of line at bottom of graph
        showline=True,
        # color of line at bottom of graph, denoting xaxis
        linecolor="#888888",
        # color of tick marks denoting bottom of x axis
        tickcolor="#888888",  # FC0000
        # color of faded vertical lines in background at each xtick
        gridcolor="#ebebeb",
        tick0=0,  # set where the first tick appars on the x-axis
        # makes a parallel line appear at the top of the graph
        mirror=True,
        title_font=dict(
            size=12,  # size of y-axis label font
            family="Helvetica",  # font style of y-axis label font
        ),
    )
    .update_yaxes(
        linecolor="#888888",  # color of line alongside y-axis, denoting yaxis
        tickcolor="#888888",  # color of tick marks along y-axis
        gridcolor="#ebebeb",  # color of horizontal lines in background at each ytick
        mirror=True,
        showgrid=True,
        showline=True,
        title_font=dict(
            size=12,  # size of y-axis label font
            family="Helvetica",  # font style of y-axis label font
        ),
    )
    .update_layout(
        # https://plotly.com/python/reference/layout/
        plot_bgcolor='white',
        paper_bgcolor="white",
        # plot_bgcolor="rgba(0, 0, 0, 0)",  # controls display of background within chart-box
        # paper_bgcolor="rgba(0, 0, 0, 0)",  # controls display of background including chart-box
        font=dict(family="Helvetica", size=12),
        autosize=True,  # i think, adjust size of chart automatically
        margin=dict(
            l=20, r=0, b=0, t=20, pad=0
        ),  # controls position of chart relative to box set aside for it
    )
).show()

# LIRR stop level API
lirr_stop = "https://data.ny.gov/resource/hb2b-cimm.json"

# pull stop-occupancy data for 2024
query = "select * where total_passengers >= 200 and passengers_by_car is not null and service_date between '2024-01-01' and '2024-12-31' and car_count != 6 LIMIT 10000000"

# read in api data
lirr_data = pl.read_json(
    get_sessiondata(lirr_stop, session, nys_params={"$query": query})
)

lirr_data_process = (
    lirr_data.with_columns(
        # convert columns to right type
        pl.col("total_passengers").cast(pl.Float64),
        pl.col("train_occupancy").cast(pl.Float64),
        pl.col("car_count").cast(pl.Float32).cast(pl.Int16),
        pl.col("direction").cast(pl.Float32).cast(pl.Int16),
    )
    .with_columns(
        # convert array columns into arrays
        pl.col("passengers_by_car")
        .str.strip_chars("[]") 
        .str.replace_all("''", "'NaN'")
        .str.replace_all("'", "")
        .str.split(" ")
        .alias("passengers_by_car_array"),
        pl.col("car_capacities")
        .str.strip_chars("[]")
        .str.replace_all("''", "'NaN'")
        .str.replace_all("'", "")
        .str.split(" ")
        .alias("car_capacities_array"),
    )
    # explode arrays so 1 row per car per stop
    .explode(["car_capacities_array", "passengers_by_car_array"])
    # order columns - exploding should have put them in right order
    .with_columns(
        (pl.row_index().over("unique_trip", "stop_code") + 1).alias(
            "car_order"
        )
    )
    .with_columns(
        # strict = False because we want to get rid of stops with any Null cars
        pl.col("car_capacities_array").cast(pl.Int32, strict=False),
        pl.col("passengers_by_car_array").cast(
            pl.Int32, strict=False
        ),
    )
    .with_columns(
        # flag for any cars being null
        pl.col("passengers_by_car_array")
        .is_null()
        .over("unique_trip", "stop_code")
        .alias("null_capacity"),
    )
    .with_columns(
        pl.max("null_capacity")
        .over("unique_trip", "stop_code")
        .alias("any_null")
    )
    # remove stops with any cars null
    .filter(pl.col("any_null") == False)
    .with_columns(
        # first column here unused, but calculate occupancy percentage and % of passengers in total
        (
            pl.col("passengers_by_car_array")
            / pl.col("total_passengers")
        ).alias("prop_passengers"),
        (
            pl.col("passengers_by_car_array")
            / pl.col("car_capacities_array")
        ).alias("prop_capacity"),
    )
)

# count total trips included
lirr_data_process.select("unique_trip", "stop_code").unique().count()

def agg_plot(
    df: pl.DataFrame,
    passenger_count: int,
    vizcol: Literal[
        "prop_total",
        "prop_passengers",
        "prop_capacity_mean",
        "prop_capacity_median",
    ],
    trainocc: float = 0,
    directions: list = [0, 1],
) -> graph_objects.Figure:
    """Process, aggregate, write out, and visualize LIRR stop-level occupancy dataset based on given set of parameters
    
    params
    ------
        df (pl.DataFrame): LIRR car/stop occupancy dataset

        passenger_count (int): Minimum stop-level passengers to include stop

        vizol (Literal["prop_total", "prop_passengers", "prop_capacity_mean", "prop_capacity_median"]): Column to aggregate and visualize in plotly chart

        trainocc (float, optional): Minimum train-level occupancy of stops to include. Default 0

        directions (list, optional): Directions to include. Default [0, 1]

    returns
    -------
    'graph_objects.Figure' bar chart showing `vizcol` by `car_order`, broken out by `car_count` and `direction`
    """

    int_df = df.filter(
        (pl.col("total_passengers") >= passenger_count)
        & (pl.col("train_occupancy") >= trainocc)
        & (pl.col("direction").is_in(directions))
    )

    pass_df = (
        int_df.select(
            "unique_trip",
            "stop_code",
            "direction",
            "car_count",
            "total_passengers",
        )
        .unique()
        .group_by("car_count", "direction")
        .agg(
            pl.col("total_passengers").sum(),
            pl.len().alias("stops"),
            pl.col("unique_trip").n_unique().alias("trips")
        )
    )

    outdf = (
        int_df.group_by("car_order", "car_count", "direction")
        .agg(
            pl.col("passengers_by_car_array").sum(),
            pl.col("prop_passengers").mean(),
            pl.col("prop_capacity")
            .mean()
            .alias("prop_capacity_mean"),
            pl.col("prop_capacity")
            .median()
            .alias("prop_capacity_median"),
        )
        .join(pass_df, how="left", on=["car_count", "direction"])
        .with_columns(
            (
                pl.col("passengers_by_car_array")
                / pl.col("total_passengers")
            ).alias("prop_total")
        )
        .sort(pl.col("car_count", "car_order"))
    )

    outdf.write_csv(
        f"blog_aggdf_pass{passenger_count}_occ{trainocc}_{vizcol}.csv"
    )

    return (
        px_configure(
            px.bar(
                outdf,
                x="car_order",
                y=vizcol,
                facet_col="car_count",
                facet_row="direction",
                color=vizcol,
                labels={
                    "car_order": "",
                    "direction": "Direction",
                    "car_count": "Cars",
                    vizcol: "",
                },
                color_continuous_scale=px.colors.colorbrewer.RdYlGn_r,
                custom_data=[outdf["stops"], outdf["trips"]],
            )
        )
    )

# vizualize mean car occupancy % overall
chart_0_mean = agg_plot(lirr_data_process, 200, "prop_capacity_mean")

chart_0_mean.show()

# vizualize  mean occupancy overall, at least 50% occupied
chart_50_stop_overall = agg_plot(lirr_data_process, 200, "prop_capacity_mean", 0.5)

chart_50_stop_overall.show()

# vizualize  mean occupancy % for at least 80% occupied stops
chart_80_mean = agg_plot(lirr_data_process, 200, "prop_capacity_mean", 0.8)

chart_80_mean.show()

# vizualize  mean occupancy for Penn Station, at least 50% occupied
nyk = lirr_data_process.filter(pl.col("stop_code") == "NYK")

chart_50_stop_nyk = agg_plot(nyk, 200, "prop_capacity_mean", 0.5)

chart_50_stop_nyk.show()

# vizualize mean occupancy for Jamaica, at least 50% occupied
jam = lirr_data_process.filter(pl.col("stop_code") == "JAM")

chart_50_stop_jam = agg_plot(jam, 200, "prop_capacity_mean", 0.5)

chart_50_stop_jam.show()