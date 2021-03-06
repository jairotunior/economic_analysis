import os
import re
import logging
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta, datetime
from full_fred.fred import Fred
import quandl
import itertools

from pathlib import Path

import holoviews as hv
import bokeh
from bokeh.models.callbacks import CustomJS
from bokeh.events import Tap, MouseEnter, PointEvent
from bokeh.palettes import Dark2_5 as palette

from bokeh.plotting import figure, show, curdoc, Figure
from bokeh.models import ColumnDataSource, NumeralTickFormatter, HoverTool, Span, Div, Toggle, BoxAnnotation, Slider, \
    CrosshairTool, Button, DataRange1d, Row, LinearColorMapper, Tabs, GroupFilter, CDSView, IndexFilter
from bokeh.layouts import gridplot, column, row, widgetbox
from bokeh.sampledata.unemployment1948 import data

import time
import random

import param
import panel as pn

from src.sources import FREDSource, QuandlSource, FileSource
from src.transformers import FractionalDifferentiationEW, FractionalDifferentiationFFD, Differentiation, PercentageChange


# Config Logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
# logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename='main.log', level=logging.DEBUG)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fred_credentials = os.path.join(BASE_DIR, "src", "api_fred.txt")
fred = Fred(fred_credentials)

pn.extension(template='bootstrap')

pn.extension(loading_spinner='dots', loading_color='#00aa41', sizing_mode="stretch_width")
pn.param.ParamMethod.loading_indicator = True

bootstrap = pn.template.BootstrapTemplate(title='FIDI Capital')


gauge = pn.indicators.Gauge(
    name='Fear and Greed', value=20, bounds=(0, 100), format='{value}',
    colors=[(0.33, 'red'), (0.66, 'gold'), (1, 'green')]
)


def bitcoin_fear_and_greed():
    fng_bitcoin_base_url = "https://api.alternative.me/fng/?limit={limit}&format=json&date_format={date_format}"
    limit = 0
    date_format = "world"

    try:
        response = requests.get(fng_bitcoin_base_url.format(limit=limit, date_format=date_format))

        if response.status_code == 200:
            data = response.json()['data']
            df = pd.DataFrame(data=data)
            df = df.rename(columns={'value': 'BTCFNG', 'value_classification': 'class', 'timestamp': 'date',
                                    'time_until_update': 'update'})
            df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
            df = df.drop('update', axis=1)
            df = df.set_index('date')

            return df
    except:
        return None

    return None


def get_quandl_dataset(serie_id):
    df = quandl.get(serie_id)
    df.index.name = 'date'

    return df


def get_fred_dataset(serie_id, columns=None, rename_column=None):
    data = fred.get_series_df(serie_id)
    data['date'] = pd.to_datetime(data['date'], format="%Y/%m/%d")
    data['value'].replace(".", np.nan, inplace=True)
    data = data.set_index('date')
    data['value'] = data['value'].astype('float')

    if columns:
        for c in columns:
            column_name = c['name']
            column_type = c['type']
            periods = c['periods']

            if column_type == 'pct':
                data[column_name] = data['value'].pct_change(periods=periods) * 100

        data = data[[*[c['name'] for c in columns]]]
    else:
        data = data[['value']]

    if rename_column:
        data = data.rename(columns={'value': rename_column})

    return data


category_list = ['US IR and Yields', 'Indicator', 'Money Markets', 'US Credit', 'Inventories', 'US Federal Debt',
                 'Index', 'US Economy']

list_ts = [
    # Fear and Greed
    {'tick': 'BTCFNG', 'name': 'BITCOIN Fear and Greed', 'source': bitcoin_fear_and_greed, 'freq': 'D', 'category': 'Indicator'},

    # Major Index
    {'tick': 'SP500', 'name': 'S&P 500', 'source': 'fred', 'freq': 'D', 'category': 'Index'},
    {'tick': 'NASDAQ100', 'name': 'NASDAQ 100 Index', 'source': 'fred', 'freq': 'D', 'category': 'Index'},
    {'tick': 'DJIA', 'name': 'Dow Jones Industrial Average', 'source': 'fred', 'freq': 'D', 'category': 'Index'},
    {'tick': 'VIXCLS', 'name': 'CBOE Volatility Index: VIX', 'source': 'fred', 'freq': 'D', 'category': 'Index'},

    # Interes Rates and Yields
    {'tick': 'FEDFUNDS', 'name': 'Federal Funds Effective Rate', 'source': 'fred', 'freq': 'M',
     'category': 'US IR and Yields'},
    {'tick': 'DFEDTAR', 'name': 'Federal Funds Target Rate (DISCONTINUED)', 'source': 'fred', 'freq': 'D',
     'category': 'US IR and Yields'},
    {'tick': 'DFEDTARL', 'name': 'Federal Funds Target Range - Lower Limit', 'source': 'fred', 'freq': 'D',
     'category': 'US IR and Yields'},
    {'tick': 'DFEDTARU', 'name': 'Federal Funds Target Range - Upper Limit', 'source': 'fred', 'freq': 'D',
     'category': 'US IR and Yields'},
    {'tick': 'DFF', 'name': 'Federal Funds Effective Rate', 'source': 'fred', 'freq': 'D',
     'category': 'US IR and Yields'},
    {'tick': 'WM2NS', 'name': 'M2', 'source': 'fred', 'freq': 'W', 'category': 'US IR and Yields'},
    {'tick': 'M2V', 'name': 'Velocity of M2 Money Stock', 'source': 'fred', 'freq': 'Q',
     'category': 'US IR and Yields'},
    {'tick': 'USTREASURY/YIELD', 'name': 'Yield Curve', 'source': 'quandl', 'freq': 'D',
     'category': 'US IR and Yields'},

    # US Economy
    {'tick': 'CPILFESL',
     'name': 'Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average',
     'source': 'fred', 'freq': 'M', 'category': 'US Economy'},
    {'tick': 'CPIAUCSL', 'name': 'Consumer Price Index for All Urban Consumers: All Items in U.S. City Average',
     'source': 'fred', 'freq': 'M', 'category': 'US Economy'},
    {'tick': 'PCE', 'name': 'Personal Consumption Expenditures', 'source': 'fred', 'freq': 'M',
     'category': 'US Economy'},
    {'tick': 'PCEPILFE', 'name': 'Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index)',
     'source': 'fred', 'freq': 'M', 'category': 'US Economy'},
    {'tick': 'GDP', 'name': 'Gross Domestic Product', 'units': 'Billions of Dollars', 'source': 'fred', 'freq': 'Q',
     'category': 'US Economy'},
    {'tick': 'NETEXP', 'name': 'Net Exports of Goods and Services', 'source': 'fred', 'freq': 'Q',
     'category': 'US Economy'},
    {'tick': 'UNRATE', 'name': 'Unemployment Rate', 'source': 'fred', 'freq': 'M', 'category': 'US Economy'},
    {'tick': 'PAYEMS', 'name': 'All Employees, Total Nonfarm', 'source': 'fred', 'freq': 'M', 'category': 'US Economy'},
    {'tick': 'HSN1F', 'name': 'New One Family Houses Sold: United States', 'source': 'fred', 'freq': 'M',
     'category': 'US Economy'},

    # US Federal Debt
    {'tick': 'FYFSD', 'name': 'Federal Surplus or Deficit', 'source': 'fred', 'freq': 'Y',
     'category': 'US Federal Debt'},
    {"tick": "Q1527BUSQ027NNBR", "name": "Government Purchases of Goods and Services for United States",
     'source': 'fred', 'freq': 'Q', 'category': 'US Federal Debt'},
    {"tick": "Q15036USQ027SNBR",
     "name": "Federal Government Purchases of Goods and Services, National Defense for United States", 'source': 'fred',
     'freq': 'Q', 'category': 'US Federal Debt'},
    {'tick': 'W068RCQ027SBEA', 'name': 'Government total expenditures', 'source': 'fred', 'freq': 'Q',
     'category': 'US Federal Debt'},
    {'tick': 'GFDEBTN', 'name': 'Federal Debt: Total Public Debt', 'source': 'fred', 'freq': 'Q',
     'category': 'US Federal Debt'},
    {'tick': 'GFDEGDQ188S', 'name': 'Federal Debt: Total Public Debt as Percent of Gross Domestic Product',
     'source': 'fred', 'freq': 'Q', 'category': 'US Federal Debt'},

    # US Credit
    {'tick': 'BUSLOANS', 'name': 'Commercial and Industrial Loans, All Commercial Banks', 'source': 'fred', 'freq': 'M',
     'category': 'US Credit'},
    {'tick': 'DPSACBW027SBOG', 'name': 'Deposits, All Commercial Banks', 'source': 'fred', 'freq': 'W',
     'category': 'US Credit'},
    {'tick': 'CCLACBW027SBOG', 'name': 'Consumer Loans: Credit Cards and Other Revolving Plans, All Commercial Banks',
     'source': 'fred', 'freq': 'W', 'category': 'US Credit'},
    {'tick': 'DRSDCIS',
     'name': 'Net Percentage of Domestic Banks Reporting Stronger Demand for Commercial and Industrial Loans from Small Firms',
     'source': 'fred', 'freq': 'Q', 'category': 'US Credit'},
    {'tick': 'DRTSCLCC', 'name': 'Net Percentage of Domestic Banks Tightening Standards for Credit Card Loans',
     'source': 'fred', 'freq': 'Q', 'category': 'US Credit'},

    # Inventories
    {'tick': 'ISRATIO', 'name': 'Total Business: Inventories to Sales Ratio', 'source': 'fred', 'freq': 'M',
     'category': 'Inventories'},
    {'tick': 'RETAILIRSA', 'name': 'Retailers: Inventories to Sales Ratio', 'source': 'fred', 'freq': 'M',
     'category': 'Inventories'},
    {'tick': 'MNFCTRIRSA', 'name': 'Manufacturers: Inventories to Sales Ratio', 'source': 'fred', 'freq': 'M',
     'category': 'Inventories'},
    {'tick': 'AISRSA', 'name': ' Auto Inventory/Sales Ratio', 'source': 'fred', 'freq': 'M', 'category': 'Inventories'},
    {'tick': 'BUSINV', 'name': 'Total Business Inventories', 'source': 'fred', 'freq': 'M', 'category': 'Inventories'},
    {'tick': 'RETAILIMSA', 'name': 'Retailers Inventories', 'source': 'fred', 'freq': 'M', 'category': 'Inventories'},
    {'tick': 'MNFCTRIMSA', 'name': 'Manufacturers Inventories', 'source': 'fred', 'freq': 'M',
     'category': 'Inventories'},
    {'tick': 'AUINSA', 'name': 'Domestic Auto Inventories', 'source': 'fred', 'freq': 'M', 'category': 'Inventories'},
    {'tick': 'RRSFS', 'name': 'Advance Real Retail and Food Services Sales', 'source': 'fred', 'freq': 'M',
     'category': 'Inventories'},

    # Money Markets
    {'tick': 'SOFR', 'name': 'Secured Overnight Financing Rate', 'source': 'fred', 'freq': 'D',
     'category': 'Money Markets'},
    {'tick': 'WLRRAL', 'name': 'Liabilities and Capital: Liabilities: Reverse Repurchase Agreements: Wednesday Level',
     'source': 'fred', 'freq': 'W', 'category': 'Money Markets'},  # Overnight Repo levels of trading
    {'tick': 'RPONTTLD',
     'name': 'Overnight Repurchase Agreements: Total Securities Purchased by the Federal Reserve in the Temporary Open Market Operations',
     'source': 'fred', 'freq': 'D', 'category': 'Money Markets'},
    {'tick': 'RPONTSYD',
     'name': 'Overnight Repurchase Agreements: Treasury Securities Purchased by the Federal Reserve in the Temporary Open Market Operations',
     'source': 'fred', 'freq': 'D', 'category': 'Money Markets'},
    {'tick': 'RRPONTSYD',
     'name': 'Overnight Reverse Repurchase Agreements: Treasury Securities Sold by the Federal Reserve in the Temporary Open Market Operations',
     'source': 'fred', 'freq': 'D', 'category': 'Money Markets'},
    {'tick': 'WORAL', 'name': 'Assets: Other: Repurchase Agreements: Wednesday Level', 'source': 'fred', 'freq': 'W',
     'category': 'Money Markets'},
]


def serie_pct_change(df, serie_id, periods=12):
    return df[serie_id].pct_change(periods=periods) * 100

def serie_differentiation(df, serie_id, lags=1):
    mask = df['{}_base'.format(serie_id)] == 1
    return df[mask][serie_id].diff(periods=lags)


def get_name_list_search():
    new_list = []

    for l in list_ts:
        tick = l['tick']
        name = l['name']
        new_list.append(name + " - " + tick)

    return new_list

"""
def create_fig(title, legend_label, x_label, y_label, source, x, y, x_data_range, tooltip_format, vtooltip=False,
               htooltip=False, tools=None, view=None, secondary_plot=False):
    # create a new plot and share only one range
    fig = figure(plot_height=400, plot_width=600, tools=tools, active_scroll='xwheel_zoom',
                 title=title, x_axis_type='datetime', x_axis_label=x_label, x_range=x_data_range,
                 y_axis_label=y_label)

    line = None

    if view is None:
        line = fig.line(x=x, y=y, source=source, legend_label=legend_label, line_width=2)
    else:
        # line = fig.line(x=x, y=y, source=source, view=view, legend_label=legend_label, line_width=2)
        line = fig.circle(x=x, y=y, source=source, size=5,
                          fill_color="blue", hover_fill_color="firebrick",
                          fill_alpha=0.9, hover_alpha=0.3,
                          line_color=None, hover_line_color="white")

    cr = None

    if secondary_plot:
        # Add the HoverTool to the figure
        cr = fig.circle(x=x, y=y, source=source, size=10,
                        fill_color="grey", hover_fill_color="firebrick",
                        fill_alpha=0.05, hover_alpha=0.3,
                        line_color=None, hover_line_color="white")

    horizontal_hovertool_fig = None
    vertical_hovertool_fig = None

    if htooltip:
        horizontal_hovertool_fig = HoverTool(tooltips=None, renderers=[cr if secondary_plot else line], mode='hline')
        fig.add_tools(horizontal_hovertool_fig)
    if vtooltip:
        vertical_hovertool_fig = HoverTool(tooltips=tooltip_format, renderers=[line], formatters={'@date': 'datetime'}, mode='vline')
        fig.add_tools(vertical_hovertool_fig)

    return fig, horizontal_hovertool_fig, vertical_hovertool_fig
"""

def create_fig(title, legend_label, x_label, y_label, source, x, y, x_data_range, tooltip_format, vtooltip=False,
               htooltip=False, tools=None, view=None, secondary_plot=False):
    # create a new plot and share only one range
    fig = figure(plot_height=400, plot_width=600, tools=tools, active_scroll='xwheel_zoom',
                 title=title, x_axis_type='datetime', x_axis_label=x_label, x_range=x_data_range,
                 y_axis_label=y_label)

    line = fig.line(x=x, y=y, source=source, name="line", legend_label=legend_label, line_width=2)
    cr = fig.circle(x=x, y=y, source=source, name="cr", size=10,
                      fill_color="grey", hover_fill_color="firebrick",
                      fill_alpha=0.05, hover_alpha=0.3,
                      line_color=None, hover_line_color="white")

    # print("*** Create Fig Method ***: ", legend_label)
    #cr.visible = False

    horizontal_hovertool_fig = None
    vertical_hovertool_fig = None

    if htooltip:
        horizontal_hovertool_fig = HoverTool(tooltips=None, renderers=[cr], names=['cr'], mode='hline')
        fig.add_tools(horizontal_hovertool_fig)
    if vtooltip:
        #vertical_hovertool_fig = HoverTool(tooltips=tooltip_format, renderers=[line], names=['line'], formatters={'@date': 'datetime'}, mode='vline')
        vertical_hovertool_fig = HoverTool(tooltips=None, renderers=[cr], mode='vline')
        fig.add_tools(vertical_hovertool_fig)

    return fig, horizontal_hovertool_fig, vertical_hovertool_fig


def combine_with_releases(df_base, df_releases, prefix_column='custom'):
    df_rel = df_releases.rename(
        columns={"ReleaseDates": "{}_rd".format(prefix_column), "MeasureMonth": "{}_mm".format(prefix_column),
                 "CrossDate": "date"})
    df_rel = df_rel.set_index("date")

    df_concat = pd.concat([df_base, df_rel], axis=1)

    # df_concat = df_concat.reset_index()
    # df_rel = df_rel.set_index("{}_mm".format(prefix_column))

    # df_results = pd.concat([df_base, df_rel], axis=1)

    return df_concat

df_list = []
df_result = None
yield_columns = ['3 MO', '2 YR', '5 YR', '7 YR', '10 YR', '20 YR', '30 YR']

download_data = False

"""
if not os.path.exists(os.path.join(BASE_DIR, "src", "dataset", "dataset_complete.csv")) or download_data:
    for k in list_ts:
        tick = k['tick']
        print(tick)
        freq = k['freq']
        source = k['source']

        df = None

        # Es una funcion custom
        if callable(source):
            df = source()
        elif source == 'fred':
            df = get_fred_dataset(tick, rename_column=tick)
        elif source == 'quandl':
            df = get_quandl_dataset(tick)

        df.fillna(method='ffill', inplace=True)
        df_list.append(df)

    df_result = pd.concat(df_list, axis=1)

    df_result = df_result.reset_index()

    # df_result.to_csv("dataset/dataset_complete.csv")
    df_result.to_pickle(os.path.join(BASE_DIR, "src", "dataset", "dataset_complete.pkl"))
else:
    # df_result = pd.read_csv("dataset/dataset_complete.csv")
    df_result = pd.read_pickle(os.path.join(BASE_DIR, "src", "dataset", "dataset_complete.pkl"))

start_date = df_result.date.min()
end_date = df_result.date.max()

x_data_range = DataRange1d(start=start_date, end=end_date)

y_yields_data_range = DataRange1d(start=df_result[yield_columns].iloc[:, -1].min(),
                                  end=df_result[yield_columns].iloc[:, -1].max())

df_yields = pd.DataFrame(
    {'yields': df_result[yield_columns].iloc[-5].T.values, 'yield_category': df_result[yield_columns].T.index})

yield_source = ColumnDataSource(data=df_yields)
main_source = ColumnDataSource(df_result)
"""


# ****************************************** Global Variables ***************************
# analist_dict format
# {"Default": [{'tick': "", "category": "", "name": "", "freq": ""},]}
#
#

FIDI_PATH = os.path.join("C:", "fidi")
ANALYSIS_PATH = os.path.join(FIDI_PATH, 'analysis')

if not os.path.exists(FIDI_PATH):
    logging.info("[+] Se ha creado directorio de archivos.")
    os.mkdir(FIDI_PATH)
    os.mkdir(ANALYSIS_PATH)


analisis_list = []
analisis_dict = {}
fig_dict = {}

start_date = None
end_date = None
x_data_range = None


fig_list = []
horizontal_hover_tool_list = []
vertical_hover_tool_list = []

tools = ['pan', 'reset', 'save', 'xwheel_zoom', 'ywheel_zoom', 'box_select', 'lasso_select']
crosshair = CrosshairTool(dimensions="both")


class ManagerTransformer:
    def __init__(self, **kwargs):
        self.transformers = {}

    def register(self, transformer):
        assert transformer.name not in self.transformers, "Ya existe un transformer registrado con ese nombre"

        self.transformers[transformer.name] = transformer

    def get_transformer_all(self):
        return [self.transformers[k] for k in self.transformers.keys()]

    def get_transformer_names(self):
        return self.transformers.keys()

class ManagerSources:
    def __init__(self, **kwargs):
        self.sources = {}

    def register(self, source):
        assert source.name not in self.sources, "Ya existe un transformer registrado con ese nombre"

        self.sources[source.name] = source

    def get_source_by_name(self, name):
        return self.sources[name]

    def get_sources_names(self):
        return self.sources.keys()

class Serie:
    def __init__(self, **kwargs):
        self.serie_id = kwargs.pop('serie_id', None)
        self.serie_name = kwargs.pop('serie_name', None)
        self.column = kwargs.pop('column', None)
        self.freq = kwargs.pop('freq', None)
        self.units = kwargs.pop('units', None)
        self.source = kwargs.pop('source', None)
        self.units_show = kwargs.pop('units_show', self.units)

        self.analysis = kwargs.pop('analysis', None)
        self.manager = self.analysis.manager

    def update(self, transform_name, **kwargs):
        is_updated = False

        for t in self.manager.transformers.get_transformer_all():
            if t.name == transform_name:
                self.column = "{}{}".format(self.serie_id, t.suffix)
                print(t.name, self.column)
                self.units_show = t.units_show
                is_updated = True

        if not is_updated:
            self.column = self.serie_id
            self.units_show = self.units

    def get_data_representation(self):
        return {
            "parent": self.analysis.name,
            "serie_id": self.serie_id,
            "serie_name": self.serie_name,
            "column": self.column,
            "units": self.units,
            "units_show": self.units_show,
            "freq": self.freq,
            "source": self.source,
        }

class Analysis:
    def __init__(self, **kwargs):
        self.name = kwargs.pop('analysis_name', "Default{}".format(str(random.randint(0, 1000))))
        self.series = kwargs.pop('series', []) # A list with Serie Object
        self.series_data = kwargs.pop('series_data', []) # A list dictionary with the serie info
        self.df = kwargs.pop('df', None)
        self.manager = kwargs.pop('manager', None)

        self.start_date = self.end_date = self.x_data_range = self.data_source = None

        if self.df is not None:
            self.start_date = self.df.date.min()
            self.end_date = self.df.date.max()
            self.x_data_range = DataRange1d(start=self.start_date, end=self.end_date)
            self.data_source = ColumnDataSource(self.df)

        if len(self.series_data) > 0:
            self._load_initial_data()
            self.update_data_source()

    def save(self):
        logging.info("[+] Guardando analisis: {}".format(self.name))

        df = None

        for i, s in enumerate(self.series):
            data = s.get_data_representation()

            if i == 0:
                df = pd.DataFrame(data={k: [] for k in data.keys()})

            df = df.append(data, ignore_index=True)

        df.to_pickle(os.path.join(self.manager.path, "{}.pkl".format(self.name)))
        logging.debug("[+] Se ha guardado exitosamente")

    def add_serie(self, **kwargs):
        #self._get_data(**kwargs)

        new_serie = Serie(analysis=self, **kwargs)
        self.series.append(new_serie)
        return new_serie

    def _load_initial_data(self):
        list_df_analisis = []

        for s in self.series_data:
            serie_id = s['serie_id']
            source = s['source']

            current_source = self.manager.sources.get_source_by_name(name=source)
            df = current_source.get_data_serie(serie_id, rename_column=serie_id)
            list_df_analisis.append(df)

        df_aux = pd.concat(list_df_analisis, axis=1)
        # df_aux.fillna(method='ffill', inplace=True)

        columns = [c for c in df_aux.columns if not re.search("_base$", c)]
        df_aux.loc[:, columns] = df_aux[columns].fillna(method='ffill')
        # df_aux[columns].fillna(method='ffill', inplace=True)

        df_aux = df_aux.reset_index()

        self.df = df_aux
        self.start_date = self.df.date.min()
        self.end_date = self.df.date.max()
        self.x_data_range = DataRange1d(start=self.start_date, end=self.end_date)
        self.data_source = ColumnDataSource(self.df)

        for s in self.series_data:
            new_serie = Serie(analysis=self, **s)
            self.series.append(new_serie)


    def _get_data(self, **kwargs):
        serie_id = kwargs.get('serie_id')
        source = kwargs.get('source')

        if self.df is None:
            current_source = self.manager.sources.get_source_by_name(name=source)
            df_serie = current_source.get_data_serie(serie_id, rename_column=serie_id)

            df = df_serie.reset_index()
            self.df = df
            self.start_date = self.df.date.min()
            self.end_date = self.df.date.max()
            self.x_data_range = DataRange1d(start=self.start_date, end=self.end_date)
            self.data_source = ColumnDataSource(self.df)
        else:
            df = self.df
            if not serie_id in df.columns:
                current_source = self.manager.sources.get_source_by_name(name=source)
                df_serie = current_source.get_data_serie(serie_id, rename_column=serie_id)

                if 'date' in df.columns:
                    df = df.set_index('date')
                    df = pd.concat([df, df_serie], axis=1)

                    columns = [c for c in df.columns if not re.search("_base$", c)]
                    df.loc[:, columns] = df[columns].fillna(method='ffill')
                    df = df.reset_index()

                    self.df = df
                    self.start_date = self.df.date.min()
                    self.end_date = self.df.date.max()
                    self.x_data_range = DataRange1d(start=self.start_date, end=self.end_date)
                    self.data_source.data = self.df
                else:
                    raise NotImplementedError("No existe la columns fecha.")

    def update_data_source(self):
        # print("**** Analysis Update Data Source *****")
        modification = False
        df = self.df

        for c in self.series:
            column_name = c.column
            serie_id = c.serie_id

            # Descarga la informacion que se necesaria
            self._get_data(serie_id=c.serie_id, source=c.source)

            for ot in self.manager.transformers.get_transformer_all():
                if "{}{}".format(serie_id, ot.suffix) == column_name:
                    if not column_name in df.columns:
                        mask = df['{}_base'.format(serie_id)] == 1

                        df.loc[mask, column_name] = ot.transform(series=df.loc[mask][serie_id])

                        columns = [c for c in df.columns if not re.search("_base$", c)]
                        df.loc[:, columns] = df[columns].fillna(method='ffill')

                        modification = True

        if modification:
            print("*** Analysis Updated DataSource ***")
            self.df = df
            if self.data_source is None:
                self.data_source = ColumnDataSource(self.df)
            else:
                self.data_source.data = self.df


class ManagerAnalysis:
    def __init__(self, path, **kwargs):
        self.path = path
        self.analysis_dict = {}

    def register(self, analysis):
        assert analysis.name not in self.analysis_dict.keys(), "Ya existe un transformer registrado con ese nombre"

        self.analysis_dict[analysis.name] = analysis

    def get_analysis_by_name(self, name):
        return self.analysis_dict[name]

    def get_analysis(self):
        return self.analysis_dict

class ManagerData:
    def __init__(self, path, **kwargs):
        self.path = path
        self.transformers = ManagerTransformer()
        self.sources = ManagerSources()
        self.analysis = ManagerAnalysis(path=path)

    def load(self):
        self._load_analysis()

    def add_analysis(self, **kwargs):
        new_analysis = Analysis(manager=self, **kwargs)
        self.sources.register(new_analysis)
        return new_analysis

    def _load_analysis(self):
        logging.info("[+] Cargando datos...")
        analysis = {}

        # Iterate Analysis Files
        for file in os.listdir(self.path):
            if file.endswith(".pkl"):
                df_analysis = pd.read_pickle(os.path.join(self.path, file))

                name_analysis = file.split('.')[0]

                # 1. Get data of series in Analysis
                list_df_analisis = []
                list_series = []

                # Iterate each one time serie
                for i, s in df_analysis.iterrows():
                    serie_id = s['serie_id']
                    source = s['source']
                    units = s['units']
                    units_show = s['units_show']
                    freq = s['freq']
                    serie_name = s['serie_name']
                    column = s['column']

                    serie_info = {"serie_id": serie_id, "source": source, 'units': units, 'units_show': units_show, 'freq': freq,
                         'serie_name': serie_name, 'column': column}

                    # Add Info Series in List
                    list_series.append(serie_info)

                # 2. Create Analysis Object
                new_analysis = Analysis(analysis_name=name_analysis, manager=self, series_data=list_series)

                # 3. Register Analysis
                self.analysis.register(new_analysis)

# Define Manager
manager = ManagerData(path=ANALYSIS_PATH)

# **************** Register Transformers ***************************
differentiation_transform = Differentiation(units_show='Returns')
fractional_diff_transform = FractionalDifferentiationEW(units_show='Fractional Return')
fractional_diff_ffd_transform = FractionalDifferentiationFFD(units_show='FFD Return')
percentage_change_transform = PercentageChange(units_show='Percentage Change')
percentage_change_from_year_transform = PercentageChange(name="Percentage Change from Year Ago", units_show='Percentage Change from Year Ago', periods=12)

manager.transformers.register(fractional_diff_transform)
manager.transformers.register(fractional_diff_ffd_transform)
manager.transformers.register(differentiation_transform)
manager.transformers.register(percentage_change_transform)
manager.transformers.register(percentage_change_from_year_transform)

# ******************* Register Sources ******************************
fred_source = FREDSource(fred_credentials=fred_credentials)
quandl_source = QuandlSource(api_key="4dvrfm6eBSwRSxwBP1Jx")
file_source = FileSource(dir=os.path.join(BASE_DIR, "src", "dataset", "yields"))

manager.sources.register(fred_source)
manager.sources.register(quandl_source)
manager.sources.register(file_source)

manager.load()

# ************************************** Load Analysis *********************************************


"""
for file in os.listdir(ANALYSIS_PATH):
    if file.endswith(".pkl"):
        df_analysis = pd.read_pickle(os.path.join(ANALYSIS_PATH, file))
        
        name_analysis = file.split('.')[0]

        analisis_dict[name_analysis] = {}
        analisis_dict[name_analysis]['series'] = []

        list_df_analisis = []

        for i, s in df_analysis.iterrows():
            serie_id = s['serie_id']
            source = s['source']
            units = s['units']
            units_show = s['units_show']
            freq = s['freq']
            serie_name = s['serie_name']
            column = s['column']

            # Add Info Series in List
            analisis_dict[name_analysis]['series'].append({"serie_id": serie_id, "source": source, 'units': units, 'units_show': units_show, 'freq': freq, 'serie_name': serie_name, 'column': column})

            # Temporal Dataframe
            df = None

            # Es una funcion custom
            if source == 'file':
                df = file_source.get_data_serie(serie_id, rename_column=serie_id)
            elif source == 'fred':
                df = fred_source.get_data_serie(serie_id, rename_column=serie_id)

                serie_data = fred.search_for_series([serie_id], limit=20)

                if re.search('Daily*', serie_data['seriess'][0]['frequency']):
                    # min_date = df.index.min()
                    # max_date = df.index.max()
                    # print(pd.date_range(start=min_date, end=max_date, freq=pd.offsets.MonthBegin(1)))
                    df = df.resample(pd.offsets.MonthBegin(1)).agg({serie_id: 'last'})
                elif re.search('Week*', serie_data['seriess'][0]['frequency']):
                    df = df.resample(pd.offsets.MonthBegin(1)).agg({serie_id: 'last'})

                df.loc[:, "{}_{}".format(serie_id, 'base')] = 1
            elif source == 'quandl':
                df = quandl_source.get_data_serie(serie_id, rename_column=serie_id)

                if re.search('Daily*', freq):
                    # min_date = df.index.min()
                    # max_date = df.index.max()
                    # print(pd.date_range(start=min_date, end=max_date, freq=pd.offsets.MonthBegin(1)))
                    df = df.resample(pd.offsets.MonthBegin(1)).agg({serie_id: 'last'})
                elif re.search('Week*', freq):
                    df = df.resample(pd.offsets.MonthBegin(1)).agg({serie_id: 'last'})

            if re.search("_pct12$", column):
                df.loc[:, "{}_pct12".format(serie_id)] = serie_pct_change(df, serie_id, periods=12)
                # df.drop(serie_id, axis=1, inplace=True)
            elif re.search("_pct1", column):
                df.loc[:, "{}_pct1".format(serie_id)] = serie_pct_change(df, serie_id, periods=1)
                # df.drop(serie_id, axis=1, inplace=True)

            #df.fillna(method='ffill', inplace=True)
            list_df_analisis.append(df)

        df_aux = pd.concat(list_df_analisis, axis=1)
        #df_aux.fillna(method='ffill', inplace=True)

        columns = [c for c in df.columns if not re.search("_base$", c)]

        df_aux.loc[:, columns] = df_aux[columns].fillna(method='ffill')
        # df_aux[columns].fillna(method='ffill', inplace=True)

        df_aux = df_aux.reset_index()

        analisis_dict[name_analysis]['df'] = df_aux

        analisis_dict[name_analysis]['start_date'] = df_aux.date.min()
        analisis_dict[name_analysis]['end_date'] = df_aux.date.max()

        analisis_dict[name_analysis]['x_data_range'] = DataRange1d(start=df_aux.date.min(), end=df_aux.date.max())

"""

# ****************************************** Create Figures *******************************
"""
for k, ts in analisis_dict.items():
    # Crea nueva key en el diccionario
    fig_dict[k] = []
    analisis_list.append(k)

    for r in ts:
        tick = r['tick']
        title = r['name']
        category = r['category']
        freq = r['freq']

        fig = h_hovertool = v_hovertool = None

        # Format the tooltip
        tooltip_format = [
            ('Date', '@date{%F}'),
            ('Rate', '@{}'.format(tick)),
        ]

        if tick in ['USTREASURY/YIELD', ]:
            # create a new plot and share only one range
            fig = figure(plot_height=400, plot_width=600, tools=tools, active_scroll='xwheel_zoom',
                         title="U.S. Yield Curve", x_axis_label='Months', y_axis_label='Yield %', x_range=yield_columns,
                         y_range=y_yields_data_range)

            fig.line(x='yield_category', y='yields', source=yield_source, legend_label="Yield Curve", line_width=2)

        else:
            df_filter = df_result[pd.notnull(df_result[tick])]

            x_range = DataRange1d(start=df_filter.date.min(), end=df_filter.date.max())

            if freq != 'D':
                view = CDSView(source=main_source, filters=[IndexFilter(df_filter.index)])

                fig, h_hovertool, v_hovertool = create_fig(title=title, legend_label=tick, x_label='date', y_label='units',
                                                           source=main_source, view=view,
                                                           x='date', y=tick, x_data_range=x_data_range,
                                                           tooltip_format=tooltip_format, htooltip=True, vtooltip=False,
                                                           tools=tools)
            else:
                fig, h_hovertool, v_hovertool = create_fig(title=title, legend_label=tick, x_label='date', y_label='units',
                                                           source=main_source,
                                                           x='date', y=tick, x_data_range=x_data_range,
                                                           tooltip_format=tooltip_format, htooltip=True, vtooltip=False,
                                                           tools=tools)

            fig.x_range = x_range

        fig_list.append({'figure': fig, 'category': category, 'title': title, 'tick': tick})
        fig_dict[k].append({'figure': fig, 'category': category, 'title': title, 'tick': tick})

        if h_hovertool:
            horizontal_hover_tool_list.append(h_hovertool)
        if v_hovertool:
            vertical_hover_tool_list.append(v_hovertool)
"""


# Use js_link to connect button active property to glyph visible property
toggle_depression_recession = Toggle(label="Show Depression / Recession", button_type="success", active=False)
toggle_chairman = Toggle(label="Show Chairman", button_type="success", active=False)
toggle_qe = Toggle(label="Show QE", button_type="success", active=False)
toggle_qt = Toggle(label="Show QT", button_type="success", active=False)
toggle_taper = Toggle(label="Show Taper", button_type="success", active=False)


# for d in fig_list:
#    f = d['figure']
#    f.add_tools(crosshair)


current_time = Span(location=datetime.now().date(), dimension='height', line_color='red', line_width=2,
                    line_dash='dashed', line_alpha=0.3)

callback_figs = CustomJS(args=dict(span=current_time), code="""
span.location = cb_obj.x;
console.log(new Date(cb_obj.x));
""")



def point_event_callback(event):
    # print(event.x, event.y, event.sx, event.sy)
    # Linea que muestra la linea de tiempo
    current_time.location = event.x

    try:
        # slider.value = event.x / 1000
        selected_date = date.fromtimestamp(event.x / 1000)

        number_elements = len(df_result.loc[df_result.date.dt.date == selected_date, yield_columns])

        if number_elements > 0:
            df_current = pd.DataFrame(
                {'yields': df_result.loc[df_result.date.dt.date == selected_date, yield_columns].T.values.ravel(),
                 'yield_category': df_result[yield_columns].T.index})
            yield_source.data = df_current
    except:
        pass

"""
for k, fig_lst in fig_dict.items():
    for d in fig_lst:
        f = d['figure']
        tick = d['tick']

        if not tick in ['USTREASURY/YIELD', ]:
            f.renderers.extend([current_time])
            f.js_on_event('tap', callback_figs)
            # f.on_event(Tap, point_event_callback)
            f.on_event('tap', point_event_callback)
            # f.on_event("mousemove", point_event_callback)
"""


def enable_vertical_hovertool_callback(event):
    for d in fig_list:
        f = d['figure']
        f.toolbar.active_inspect = None

# ************************************* Methods ***********************************
def create_ts(source, x_data_range, **kwargs):
    tick = kwargs.get('column', None)
    serie_name = kwargs.get('serie_name', None)
    freq = kwargs.get('freq', None)
    units = kwargs.get('units', None)

    fig = h_hovertool = v_hovertool = None

    # Format the tooltip
    tooltip_format = [
        ('Date', '@date{%F}'),
        ('Rate', '@{}'.format(tick)),
    ]

    fig, h_hovertool, v_hovertool = create_fig(title=serie_name, legend_label=tick, x_label='date',
                                               y_label=units,
                                               source=source,
                                               x='date', y=tick, x_data_range=x_data_range,
                                               tooltip_format=tooltip_format, htooltip=True, vtooltip=True,
                                               tools=tools)

    return fig, h_hovertool, v_hovertool


def add_current_time_span(fig):
    fig.renderers.extend([current_time])
    fig.js_on_event('tap', callback_figs)
    # f.on_event(Tap, point_event_callback)
    # Actualizacion de la Yield Curve
    #fig.on_event('tap', point_event_callback)
    # f.on_event("mousemove", point_event_callback)


def add_auto_adjustment_y(fig, source, column):
    #source = ColumnDataSource(dict(index=df.date, x=df[column]))

    callback_rescale = CustomJS(args=dict(y_range=fig.y_range, source=source, column=column), code='''
        console.log(source.data[column]);
        clearTimeout(window._autoscale_timeout);
 
        var index = source.data.index,
            x = source.data[column],
            start = cb_obj.start,
            end = cb_obj.end,
            min = 1e10,
            max = -1;

        for (var i=0; i < index.length; ++i) {
            if (start <= index[i] && index[i] <= end) {
                max = Math.max(x[i], max);
                min = Math.min(x[i], min);
            }
        }
        var pad = (max - min) * .05;

        window._autoscale_timeout = setTimeout(function() {
            y_range.start = min - pad;
            y_range.end = max + pad;
        }, 50);
    ''')

    fig.x_range.js_on_change('start', callback_rescale)


def add_sync_crosshair(fig):
    fig.add_tools(crosshair)

# Show Crisis about BEA, QE Periods, QT Periods, Fed Chairman, Show Shock Events (Ex. Mexican Crisis, LTCM, Asian Crisis)
def add_recession_info(fig):
    df_business_cycle = pd.read_csv(os.path.join(BASE_DIR, "src", "dataset", "business_cycle.csv"))
    df_business_cycle['start'] = pd.to_datetime(df_business_cycle['start'], format="%d/%m/%Y")
    df_business_cycle['end'] = pd.to_datetime(df_business_cycle['end'], format="%d/%m/%Y")

    recession_list = []

    for i, row in df_business_cycle.iterrows():
        r = BoxAnnotation(left=row['start'], right=row['end'], fill_color='#009E73', fill_alpha=0.1)
        r.visible = False
        recession_list.append(r)

    for r in recession_list:
        fig.add_layout(r)
        toggle_depression_recession.js_link('active', r, 'visible')

def add_chairman_fed(fig):
    df_chairman = pd.read_csv(os.path.join(BASE_DIR, "src", "dataset", "chairmanfed.csv"))
    df_chairman['start'] = pd.to_datetime(df_chairman['start'], format="%d/%m/%Y")
    df_chairman['end'] = pd.to_datetime(df_chairman['end'], format="%d/%m/%Y")

    list_chairman_fed = []

    for i, row in df_chairman.iterrows():
        start = row['start']
        end = row['end']

        if pd.isnull(row['end']):
            end = date.today()

        r = BoxAnnotation(left=start, right=end, fill_color=palette[i], fill_alpha=0.1)
        r.visible = False
        list_chairman_fed.append(r)

    for c in list_chairman_fed:
        fig.add_layout(c)
        toggle_chairman.js_link('active', c, 'visible')

def add_qe(fig):
    df_qe = pd.DataFrame(data={"name": ['QE1', 'QE2', 'QE3 - Operation Twist', "QE4", 'QE5', 'QECovid'],
                                "start": ['25/11/2008', '3/11/2010', '21/9/2012', '2/1/2013', '11/10/2019', '15/3/2020'],
                                "end": ['31/3/2010', '29/6/2012', '31/12/2012', '29/10/2014', '15/3/2020', None]
                               })

    df_qe['start'] = pd.to_datetime(df_qe['start'], format="%d/%m/%Y")
    df_qe['end'] = pd.to_datetime(df_qe['end'], format="%d/%m/%Y")

    list_qe = []

    for i, row in df_qe.iterrows():
        start = row['start']
        end = row['end']

        if pd.isnull(row['end']):
            end = date.today()

        r = BoxAnnotation(left=start, right=end, fill_color='#C44125', fill_alpha=0.1)
        r.visible = False
        list_qe.append(r)

    for c in list_qe:
        fig.add_layout(c)
        toggle_qe.js_link('active', c, 'visible')

def add_qt(fig):
    df_qt = pd.DataFrame(data={"name": ['QT', ],
                                "start": ['14/6/2017', ],
                                "end": ['11/10/2019', ]
                               })

    df_qt['start'] = pd.to_datetime(df_qt['start'], format="%d/%m/%Y")
    df_qt['end'] = pd.to_datetime(df_qt['end'], format="%d/%m/%Y")

    list_qt = []

    for i, row in df_qt.iterrows():
        start = row['start']
        end = row['end']

        if pd.isnull(row['end']):
            end = date.today()

        r = BoxAnnotation(left=start, right=end, fill_color='#FFC300', fill_alpha=0.1)
        r.visible = False
        list_qt.append(r)

    for c in list_qt:
        fig.add_layout(c)
        toggle_qt.js_link('active', c, 'visible')

def add_taper(fig):
    df_taper = pd.DataFrame(data={"name": ['Taper1', 'TaperCovid'],
                                "start": ['18/12/2013', '3/11/2021'],
                                "end": ['29/10/2014', None]
                               })

    df_taper['start'] = pd.to_datetime(df_taper['start'], format="%d/%m/%Y")
    df_taper['end'] = pd.to_datetime(df_taper['end'], format="%d/%m/%Y")

    list_taper = []

    for i, row in df_taper.iterrows():
        start = row['start']
        end = row['end']

        if pd.isnull(row['end']):
            end = date.today()

        r = BoxAnnotation(left=start, right=end, fill_color='#581845', fill_alpha=0.1)
        r.visible = False
        list_taper.append(r)

    for c in list_taper:
        fig.add_layout(c)
        toggle_taper.js_link('active', c, 'visible')

def layout_row(list_figs, figs_per_row):
    list_result = []
    current_list = []

    for i, f in enumerate(list_figs, start=1):
        if i % figs_per_row == 0:
            current_list.append(f)
            list_result.append(current_list)
            current_list = []
        else:
            current_list.append(f)

    if len(current_list) > 0:
        list_result.append(current_list)

    return list_result


class ChartForm(param.Parameterized):

    def __init__(self, **kwargs):
        assert kwargs.get('serie', None), "Debe suministrar un objeto serie"
        assert kwargs.get('parent', None), "Debe suministrar un objeto AnalysisForm"

        self.parent = kwargs.pop('parent', None)
        self.serie = kwargs.pop('serie', None)

        self.analysis = self.serie.analysis
        self.manager = self.analysis.manager

        super().__init__(**kwargs)

        option_transforms = self.manager.transformers.get_transformer_names()

        select_value = 'Normal'

        for ot in self.manager.transformers.get_transformer_all():
            if re.search("{}$".format(ot.suffix), self.serie.column):
                select_value = ot.name
                break

        #self.select_processing_2 = pn.widgets.Select(name='Transform', value=select_value, options=['Normal', 'Percentage Change', 'Percentage Change from Year Ago'])
        self.select_processing_2 = pn.widgets.Select(name='Transform', value=select_value, options=['Normal', *option_transforms])
        self.select_processing_2.param.watch(self.set_column, 'value', onlychanged=True, precedence=0)

        fg, h_hovertool, v_hovertool = create_ts(source=self.analysis.data_source, x_data_range=self.analysis.x_data_range,
                                                 column=self.serie.column, serie_name=self.serie.serie_name, freq=self.serie.freq,
                                                 units=self.serie.units_show)
        add_recession_info(fg)
        add_current_time_span(fg)
        # add_auto_adjustment_y(fg, self.parent.data_source, self.column)
        add_chairman_fed(fg)
        add_qe(fg)
        add_qt(fg)
        add_taper(fg)

        fg.add_tools(self.parent.crosshair)

        self.fig = fg
        self.h_hovertool = h_hovertool
        self.v_hovertool = v_hovertool

    def set_column(self, event):
        transformation_name = event.new
        """
        # print("** Chart Form: ", transformation)
        serie_id = self.serie_id

        is_update = False

        for ot in self.manager.transformers.get_transformer_all():
            if transformation == ot.name:
                column = "{}{}".format(serie_id, ot.suffix)
                self.serie.column = column
                self.serie.units_show = self.serie.units if ot.units_show is None else ot.units_show
                is_update = True
                break

        if is_update:
            self.serie.column = serie_id
            self.serie.units_show = self.serie.units

        print("ChartForm Set Column: ", self.serie.column)
        """

        self.serie.update(transformation_name)

        self.update_plot()

    def update_plot(self):
        line = self.fig.select_one({'name': 'line'})
        cr = self.fig.select_one({'name': 'cr'})

        line.glyph.y = self.serie.column
        cr.glyph.y = self.serie.column

        self.fig.yaxis[0].axis_label = self.serie.units_show

        # Set Name of legent
        self.fig.legend[0].name = self.serie.column

    #@param.depends('select_processing')
    def view(self):
        #data_source = self.output()
        # print("******** Chart Form View**************")
        # print("************ Chart Form - Set column **************")
        return self.fig

    def panel(self):
        return pn.Card(pn.Column(self.view, self.select_processing_2), title=self.serie.serie_name)

    def __repr__(self, *_):
        return "Value"


class AnalysisForm(param.Parameterized):

    action_update_analysis = param.Action(lambda x: x.param.trigger('action_update_analysis'), label='Update')
    action_save_analysis = param.Action(lambda x: x.param.trigger('action_save_analysis'), label='Save')

    chart_forms = param.List([], item_type=ChartForm)

    def __init__(self, parent, **kwargs):
        assert kwargs.get('analysis', None), "Debe suministrar un objeto analysis"

        self.analysis = kwargs.pop('analysis', None)
        self.manager = kwargs.pop('manager', [])

        #self.analysis_name = kwargs.pop('analysis_name', "Default{}".format(str(random.randint(0, 1000))))

        self.ncols = kwargs.pop('ncols', 3)

        super().__init__(**kwargs)

        self.parent = parent

        self.crosshair = CrosshairTool(dimensions="both")

        self.param.watch(self.save_analysis, 'action_save_analysis')

        self._load()

    def _load(self):
        list_chart_form = []
        for s in self.analysis.series:
            chart_form = ChartForm(parent=self, serie=s)
            chart_form.select_processing_2.param.watch(self.update_view, 'value', precedence=1)
            list_chart_form.append(chart_form)

        self.chart_forms = list_chart_form

    def save_analysis(self, event):
        self.analysis.save()

    def add_chart(self, **kwargs):
        #serie_id = kwargs.get('serie_id')

        # Update DataSource
        new_serie = self.analysis.add_serie(**kwargs)

        self.analysis.update_data_source()

        chart_form = ChartForm(parent=self, serie=new_serie)
        chart_form.select_processing_2.param.watch(self.update_view, 'value', precedence=1)

        self.chart_forms = [*self.chart_forms, chart_form]

        return chart_form

    def get_data_source(self):
        # print("**** Analysis Update Data Source *****")
        self.analysis.update_data_source()

        """
        modification = False
        df = self.df

        for c in self.chart_forms:
            processing = c.select_processing_2.value
            # print("** Analysis Processing Variable: ", c.select_processing_2)
            serie_id = c.serie_id
            # print("** Analysis Serie: ", serie_id)

            if processing == 'Percentage Change':
                column = "{}_{}".format(serie_id, "pct1")
                if not column in df.columns:
                    mask = df['{}_base'.format(serie_id)] == 1
                    df.loc[mask, column] = serie_pct_change(df.loc[mask], serie_id, periods=1)

                    columns = [c for c in df.columns if not re.search("_base$", c)]
                    df.loc[:, columns] = df[columns].fillna(method='ffill')

                    c.column = column
                    modification = True
            elif processing == 'Percentage Change from Year Ago':
                column = "{}_{}".format(serie_id, "pct12")
                if not column in df.columns:
                    mask = df['{}_base'.format(serie_id)] == 1
                    df.loc[mask, column] = serie_pct_change(df.loc[mask], serie_id, periods=12)

                    columns = [c for c in df.columns if not re.search("_base$", c)]
                    df.loc[:, columns] = df[columns].fillna(method='ffill')

                    c.column = column
                    modification = True
            elif processing == 'Differentiation':
                column = "{}_{}".format(serie_id, "diff")
                if not column in df.columns:
                    df[column] = serie_differentiation(df, serie_id)
                    c.column = column
                    modification = True
            elif processing == 'Normal':
                c.column = serie_id
                continue

        if modification:
            print("*** Analysis Updated DataSource ***")
            self.df = df
            self.data_source.data = self.df

        #print(self.df.columns)

        #return self.data_source
        """

    def update_view(self, event):
        self.param.trigger('action_update_analysis')

    @param.depends('action_update_analysis', 'chart_forms')
    def view(self):
        # print("***** Analysis View *********")
        # Arreglar esto aqui ya que al traer el dato modifica el datasource
        #self.data_source = self.get_data_source()
        #self.get_data_source()
        self.analysis.update_data_source()
        return pn.GridBox(*[c.panel() for c in self.chart_forms], ncols=self.parent.plot_by_row)

    def panel(self):
        # print("********** Analysis Panel ****************")
        return pn.Column(self.param, self.view)

    def __repr__(self, *_):
        return "Value"


class SeriesForm(param.Parameterized):
    plot_by_row = param.Integer(3, bounds=(1, 4))

    autocomplete_search_serie = pn.widgets.TextInput(name='Search Serie', placeholder='Ticker or Serie Name')
    search_source = pn.widgets.Select(name='Source', value='fred', options=['fred', 'quandl', 'file'])

    button_open_modal = pn.widgets.Button(name='Add Serie', width_policy='fit', height_policy='fit', button_type='primary')
    button_add_serie = pn.widgets.Button(name='Add Serie', width_policy='fit', height_policy='fit', button_type='primary')
    #button_create_analisis = pn.widgets.Button(name='New Analysis', button_type='success')

    selected_analysis_name = param.String(default="")
    action_new_analysis = param.Action(lambda x: x.param.trigger('action_new_analysis'), label='New Analysis')

    action_update_tabs = param.Action(lambda x: x.param.trigger('action_update_tabs'), label='Update Tabs')
    action_update_alerts = param.Action(lambda x: x.param.trigger('action_update_alerts'), label='Update Alerts')
    action_update_search_results = param.Action(lambda x: x.param.trigger('action_update_search_results'), label='Update Search Result')

    analysis_list = param.List([], item_type=AnalysisForm)

    def __init__(self, **kwargs):
        assert kwargs.get('manager', None), "Definir el manager para la vista"
        manager = kwargs.pop('manager')

        super().__init__(**kwargs)

        self.button_add_serie.param.watch(self.add_serie_buttom, 'value')
        #self.button_create_analisis.param.watch(self.create_analysis, 'value')

        self.param.watch(self.create_analysis, 'action_new_analysis')

        self.button_open_modal.param.watch(self.open_modal, 'value')
        self.autocomplete_search_serie.param.watch(self.do_search, 'value')

        self.alerts = []
        self.search_result = []

        self.manager = manager

        self._load()

    def _load(self):
        logging.info("[+] Loading Analysis")
        list_form_analysis = []
        for k in self.manager.analysis.get_analysis().keys():
            analysis = self.manager.analysis.get_analysis_by_name(k)
            form_analysis = AnalysisForm(parent=self, analysis=analysis)
            list_form_analysis.append(form_analysis)
        self.analysis_list = list_form_analysis

    def _get_selected_data_source(self):
        return self.manager.sources.get_source_by_name(self.search_source.value)

    def add_analysis(self, **kwargs):
        new_analysis = self.manager.add_analysis(**kwargs)
        form_analysis = AnalysisForm(parent=self, analysis=new_analysis)
        self.analysis_list = [*self.analysis_list, form_analysis]
        return form_analysis

    def do_search(self, event):
        logging.info("[+] Obteniendo resultados de busqueda ...")
        if event.new != "":
            self._get_selected_data_source().do_search(event.new)
            self.param.trigger('action_update_search_results')

    def open_modal(self, event):
        bootstrap.open_modal()

    def add_serie(self, event):
        serie_name = self.autocomplete_search_serie.value.split("-")

        if isinstance(serie_name, list):
            if len(serie_name) == 2:
                pos = [i for i, t in enumerate(list_ts) if t['tick'] == serie_name[1].replace(" ", "")]

                if serie_name != "" and pos[0]:
                    tab_id = analisis_list[0]
                    fg = create_ts(list_ts[pos[0]], tab_id)
                    add_recession_info(fg)
                    add_current_time_span(fg)
                    add_sync_crosshair(fg)
            else:
                self.alerts.append("Seleccione una serie de tiempo valida.")
                self.param.trigger('action_update_alerts')
        else:
            self.alerts.append("Seleccione una serie de tiempo valida.")
            self.param.trigger('action_update_alerts')

        self.param.trigger('action_update_tabs')

    def _get_current_analysis_form(self):
        return self.analysis_list[self.tabs.active]

    def add_serie_buttom(self, event):
        logging.info("[+] Agregando nueva serie de tiempo ...")
        serie_id = event.obj.name

        # Serie Information
        serie_data = None

        for s in self._get_selected_data_source().get_search_results():
            if s['id'] == serie_id:
                serie_data = {"serie_id": s['id'], "source": self.search_source.value, 'units': s['units'], 'freq': s['frequency'], 'serie_name': s['name'], 'column': s['id']}
                break

        analysis_form = self._get_current_analysis_form()

        analysis_form.add_chart(**serie_data)

        bootstrap.close_modal()

    @param.depends('action_update_alerts', watch=False)
    def get_alerts(self):
        list_alerts = []

        for a in self.alerts:
            list_alerts.append(pn.pane.Alert('## Alert\n{}'.format(a)))

        self.alerts = []

        return list_alerts[0] if len(list_alerts) > 0 else None

    def tabinfo(self, event):
        print("TAB: ", self.tabs.active)

    @param.depends('action_update_tabs', 'plot_by_row', 'analysis_list', watch=False)
    def get_tabs(self):
        tabs = None
        tuplas = []

        for a in self.analysis_list:
            panel = a.panel()
            tuplas.append((a.analysis.name, panel))

        self.tabs = pn.Tabs(*tuplas, closable=True)

        self.tabs.param.watch(self.tabinfo, 'active')

        return self.tabs

    @param.depends('action_update_search_results', watch=False)
    def get_search_results(self):
        logging.info("[+] Renderizando resultados de busqueda ...")

        rows = []
        description = """
        **{}**\n
        {} to {}\n
        {}\n
        {}
        """

        selected_source = self._get_selected_data_source()

        for r in selected_source.get_search_results():
            button_select = pn.widgets.Button(name=r['id'])
            button_select.param.watch(self.add_serie_buttom, 'value')

            rows.append(pn.Card(
                    pn.Column(description.format(r['name'], r['observation_start'], r['observation_end'], r['frequency'], r['notes']), button_select),
                    header=pn.panel(selected_source.logo, height=40), header_color=selected_source.header_color, header_background=selected_source.header_background)
            )

        return pn.Column("**Search Results:**", pn.GridBox(*rows, ncols=4)) if len(rows) > 0 else None

    def create_analysis(self, event):
        logging.debug("[+] Creando nuevo analisis: {}".format(self.selected_analysis_name))

        if self.selected_analysis_name == "":
            self.alerts.append("Digite un nombre para el nuevo analisis.")
            self.param.trigger('action_update_alerts')
        else:
            self.add_analysis(analysis_name=self.selected_analysis_name)
            self.param.trigger('action_update_tabs')

    def __repr__(self, *_):
        return "Value"


# ******************************************** Create Figures *************************************
logging.info("[+] Renderizando informacion...")

series_form = SeriesForm(manager=manager)

"""
for k in analisis_dict.keys():
    series = analisis_dict[k]['series']

    current_analysis = series_form.add_analysis(analysis_name=k, **analisis_dict[k])

    # print("[+] Analisis: {}".format(k))
    # print(series)

    tab_id = k

    for s in series:
        # print(s)
        current_analysis.add_chart(s)
"""

alerts = pn.Row(pn.panel(series_form.get_alerts, width=300))
container = pn.Row(pn.panel(series_form.get_tabs, width=300))


bootstrap.sidebar.append(series_form.button_open_modal)
# bootstrap.sidebar.append(series_form.button_create_analisis)


bootstrap.sidebar.append(toggle_depression_recession)
bootstrap.sidebar.append(toggle_qe)
bootstrap.sidebar.append(toggle_qt)
bootstrap.sidebar.append(toggle_taper)
bootstrap.sidebar.append(toggle_chairman)

bootstrap.sidebar.append(series_form.param)
"""
bootstrap.sidebar.append(checkbox_crisis)
bootstrap.sidebar.append(checkbox_qe)
bootstrap.sidebar.append(checkbox_qt)
bootstrap.sidebar.append(checkbox_shock_events)
bootstrap.sidebar.append(checkbox_enable_highlight)
bootstrap.sidebar.append(checkbox_enable_syncronize_chart)
bootstrap.sidebar.append(checkbox_enable_vertical_tooltip)
bootstrap.sidebar.append(checkbox_enable_horizontal_tooltip)

"""

"""
bootstrap.sidebar.append(pn.Card(gauge, title="Bitcoin Fear and Greed"))

# Bitcoin Fear and Greed
gauge.value = int(df_result[pd.notnull(df_result['BTCFNG'])]['BTCFNG'].iat[-1])
gauge.name = df_result[pd.notnull(df_result['BTCFNG'])]['class'].iat[-1]

trend_list = []

for t in list_ts:
    tick = t['tick']
    category = t['category']
    name = t['name']

    if category == 'Index':
        df_filter = df_result[pd.notnull(df_result[tick])].tail(30)

        data = {'x': df_filter.index.values.tolist(), 'y': df_filter[tick].values.tolist()}

        trend = pn.indicators.Trend(
            title=name, data=data, width=200, height=200, plot_type='area', value=df_filter[tick].iat[-1]
        )

        trend_list.append(trend)

number = pn.indicators.Number(
    name='CPI', value=72, format='{value}%',
    colors=[(33, 'green'), (66, 'gold'), (100, 'red')]
)

analitics = pn.Row(*trend_list)

bootstrap.main.append(analitics)
"""

bootstrap.modal.append(pn.Column(pn.Row(series_form.autocomplete_search_serie, series_form.search_source), series_form.get_search_results))

bootstrap.main.append(alerts)
bootstrap.main.append(container)

bootstrap.show()

