import os
import re
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta, datetime
from full_fred.fred import Fred
import quandl

from pathlib import Path

import holoviews as hv
import bokeh
from bokeh.models.callbacks import CustomJS
from bokeh.events import Tap, MouseEnter, PointEvent

from bokeh.plotting import figure, show, curdoc, Figure
from bokeh.models import ColumnDataSource, NumeralTickFormatter, HoverTool, Span, Div, Toggle, BoxAnnotation, Slider, \
    CrosshairTool, Button, DataRange1d, Row, LinearColorMapper, Tabs, GroupFilter, CDSView, IndexFilter
from bokeh.layouts import gridplot, column, row, widgetbox
from bokeh.sampledata.unemployment1948 import data

import time
import random

import param
import panel as pn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fred_credentials = os.path.join(BASE_DIR, "src", "api_fred.txt")
fred = Fred(fred_credentials)

pn.extension(template='bootstrap')

pn.extension(loading_spinner='dots', loading_color='#00aa41', sizing_mode="stretch_width")
pn.param.ParamMethod.loading_indicator = True

bootstrap = pn.template.BootstrapTemplate(title='FIDI Capital')

# Show Crisis about BEA, QE Periods, QT Periods, Fed Chairman, Show Shock Events (Ex. Mexican Crisis, LTCM, Asian Crisis)
checkbox_crisis = pn.widgets.Checkbox(name='Show Crisis')
checkbox_qe = pn.widgets.Checkbox(name='Show QE Periods')
checkbox_qt = pn.widgets.Checkbox(name='Show QT Periods')
checkbox_shock_events = pn.widgets.Checkbox(name='Show Shock Events')
checkbox_enable_highlight = pn.widgets.Checkbox(name='Show Highlight TS')
checkbox_enable_syncronize_chart = pn.widgets.Checkbox(name='Syncronize Chart')
checkbox_enable_vertical_tooltip = pn.widgets.Checkbox(name='Enable Vertical Tooltip', value=True)
checkbox_enable_horizontal_tooltip = pn.widgets.Checkbox(name='Enable Horizontal Tooltip', value=True)

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
    fig = figure(plot_height=400, plot_width=600, tools=tools, active_scroll='wheel_zoom',
                 title=title, x_axis_type='datetime', x_axis_label=x_label, x_range=x_data_range,
                 y_axis_label=y_label)

    line = fig.line(x=x, y=y, source=source, legend_label=legend_label, line_width=2)
    cr = fig.circle(x=x, y=y, source=source, size=10,
                      fill_color="grey", hover_fill_color="firebrick",
                      fill_alpha=0.05, hover_alpha=0.3,
                      line_color=None, hover_line_color="white")

    print("*** Create Fig Method ***: ", legend_label)

    horizontal_hovertool_fig = None
    vertical_hovertool_fig = None

    if htooltip:
        horizontal_hovertool_fig = HoverTool(tooltips=None, renderers=[cr], mode='hline')
        fig.add_tools(horizontal_hovertool_fig)
    if vtooltip:
        vertical_hovertool_fig = HoverTool(tooltips=tooltip_format, renderers=[line], formatters={'@date': 'datetime'}, mode='vline')
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
analisis_list = []
analisis_dict = {}
fig_dict = {}

start_date = None
end_date = None
x_data_range = None


fig_list = []
horizontal_hover_tool_list = []
vertical_hover_tool_list = []

tools = ['pan', 'reset', 'save', 'wheel_zoom', 'box_select', 'lasso_select']
crosshair = CrosshairTool(dimensions="both")



class ManagerData:
    def __init__(self):
        pass

# ************************************** Load Analysis *********************************************
df_analysis = pd.DataFrame({"name": ["Analisis 1", "Analisis 2", "Analisis 1"],
                            "serie_id": ['GDP', 'PCE', 'CPIAUCSL'],
                            'serie_name': ['Lorem', 'Ipsum', 'GGDDD LP'],
                            'source': ['fred', 'fred', 'fred'],
                            'units': ['Billions', 'Billions', 'Billions'],
                            'freq': ['M', 'M', 'M'],
                            'column_name': ['GDP', 'PCE', 'CPIAUCSL']})

analisis_list = df_analysis['name'].unique()

for a in analisis_list:
    analisis_dict[a] = {}
    analisis_dict[a]['series'] = []
    analisis_dict[a]['figures'] = []
    analisis_dict[a]['datasource'] = None
    analisis_dict[a]['hover_tool_h'] = []
    analisis_dict[a]['hover_tool_v'] = []

    df_series = df_analysis[df_analysis['name'] == a]

    list_df_analisis = []

    for i, s in df_series.iterrows():
        serie_id = s['serie_id']
        source = s['source']
        units = s['units']
        freq = s['freq']
        serie_name = s['serie_name']
        column = s['column_name']

        # Add Info Series in List
        analisis_dict[a]['series'].append({"serie_id": serie_id, "source": source, 'units': units, 'freq': freq, 'serie_name': serie_name, 'column': column})

        # Temporal Dataframe
        df = None

        # Es una funcion custom
        if callable(source):
            df = source()
        elif source == 'fred':
            df = get_fred_dataset(serie_id, rename_column=serie_id)
            #df.loc[:, "{}_{}".format(serie_id, 'base')] = 1
        elif source == 'quandl':
            df = get_quandl_dataset(serie_id)

        #df.fillna(method='ffill', inplace=True)
        list_df_analisis.append(df)

    df_aux = pd.concat(list_df_analisis, axis=1)
    df_aux.fillna(method='ffill', inplace=True)

    #columns = [c for c in df_aux.columns if not re.search("_base$", c)]
    #df_aux.loc[:, columns] = df_aux[columns].fillna(method='ffill', inplace=True)

    df_aux = df_aux.reset_index()

    analisis_dict[a]['df'] = df_aux

    analisis_dict[a]['start_date'] = df_aux.date.min()
    analisis_dict[a]['end_date'] = df_aux.date.max()

    analisis_dict[a]['x_data_range'] = DataRange1d(start=df_aux.date.min(), end=df_aux.date.max())

    analisis_dict[a]['datasource'] = ColumnDataSource(analisis_dict[a]['df'])


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
toggle1 = Toggle(label="Show Depression / Recession", button_type="success", active=True)


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


def serie_pct_change(df, serie_id, periods=12):
    return df[serie_id].pct_change(periods=periods) * 100

def serie_differentiation(df, serie_id, lags=1):
    return df[serie_id].diff(periods=lags)


# ************************************* Methods ***********************************
def create_ts(ts, source, x_data_range):
    tick = ts['column']
    title = ts['serie_name']
    freq = ts['freq']
    units = ts['units']

    fig = h_hovertool = v_hovertool = None

    # Format the tooltip
    tooltip_format = [
        ('Date', '@date{%F}'),
        ('Rate', '@{}'.format(tick)),
    ]

    fig, h_hovertool, v_hovertool = create_fig(title=title, legend_label=tick, x_label='date',
                                               y_label=units,
                                               source=source,
                                               x='date', y=tick, x_data_range=x_data_range,
                                               tooltip_format=tooltip_format, htooltip=True, vtooltip=False,
                                               tools=tools)

    return fig, h_hovertool, v_hovertool


def add_current_time_span(fig):
    fig.renderers.extend([current_time])
    fig.js_on_event('tap', callback_figs)
    # f.on_event(Tap, point_event_callback)
    # Actualizacion de la Yield Curve
    #fig.on_event('tap', point_event_callback)
    # f.on_event("mousemove", point_event_callback)


def add_auto_adjustment_y(fig, df, serie_id):
    source = ColumnDataSource(dict(index=df.date, x=df[serie_id]))

    callback_rescale = CustomJS(args=dict(y_range=fig.y_range, source=source), code='''
        clearTimeout(window._autoscale_timeout);
 
        var index = source.data.index,
            x = source.data.x,
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


def add_recession_info(fig):
    df_business_cycle = pd.read_csv(os.path.join(BASE_DIR, "src", "dataset", "business_cycle.csv"))
    df_business_cycle['start'] = pd.to_datetime(df_business_cycle['start'], format="%d/%m/%Y")
    df_business_cycle['end'] = pd.to_datetime(df_business_cycle['end'], format="%d/%m/%Y")

    recession_list = []

    for i, row in df_business_cycle.iterrows():
        r = BoxAnnotation(left=row['start'], right=row['end'], fill_color='#009E73', fill_alpha=0.1)
        recession_list.append(r)

    for r in recession_list:
        fig.add_layout(r)
        toggle1.js_link('active', r, 'visible')



def add_chairman_fed(fig):
    df_chairman = pd.read_csv(os.path.join(BASE_DIR, "src", "dataset", "chairmanfed.csv"))
    df_chairman['start'] = pd.to_datetime(df_chairman['start'], format="%d/%m/%Y")
    df_chairman['end'] = pd.to_datetime(df_chairman['end'], format="%d/%m/%Y")

    print(df_chairman.head())
    print(df_chairman.dtypes)

    list_chairman_fed = []

    for i, row in df_chairman.iterrows():
        start = row['start']
        end = row['end']

        if pd.isnull(row['end']):
            end = date.today()

        r = BoxAnnotation(left=start, right=end, fill_color='#009E73', fill_alpha=0.1)
        list_chairman_fed.append(r)

    for c in list_chairman_fed:
        fig.add_layout(c)
        #toggle1.js_link('active', r, 'visible')


def add_qe(fig):
    """
    QE1,25/11/2008,25/6/2010
    QE2,11/2010,6/2011
    QE3 - Operation Twist,9/2012,12/2012
    QE4,1/2013,10/2014
    QE5,15/3/2020,
    QT,
    """

    df_chairman = pd.read_csv(os.path.join(BASE_DIR, "src", "dataset", "chairmanfed.csv"))
    df_chairman['start'] = pd.to_datetime(df_chairman['start'], format="%d/%m/%Y")
    df_chairman['end'] = pd.to_datetime(df_chairman['end'], format="%d/%m/%Y")

    print(df_chairman.head())
    print(df_chairman.dtypes)

    list_chairman_fed = []

    for i, row in df_chairman.iterrows():
        start = row['start']
        end = row['end']

        if pd.isnull(row['end']):
            end = date.today()

        r = BoxAnnotation(left=start, right=end, fill_color='#009E73', fill_alpha=0.1)
        list_chairman_fed.append(r)

    for c in list_chairman_fed:
        fig.add_layout(c)
        #toggle1.js_link('active', r, 'visible')

    """
    arrow_style = dict(facecolor='black', edgecolor='white', shrink=0.05)
    ax.annotate('QE1', xy=('2008-11-25', 0), xytext=('2008-11-25', -4), size=12, ha='right', arrowprops=arrow_style)
    ax.annotate('QE1+', xy=('2009-03-18', 0), xytext=('2009-03-18', -6), size=12, ha='center', arrowprops=arrow_style)
    ax.annotate('QE2', xy=('2010-11-03', 0), xytext=('2010-11-03', -4), size=12, ha='center', arrowprops=arrow_style)
    ax.annotate('QE2+', xy=('2011-09-21', 0), xytext=('2011-09-21', -4.5), size=12, ha='center', arrowprops=arrow_style)
    ax.annotate('QE2+', xy=('2012-06-20', 0), xytext=('2012-06-20', -6.5), size=12, ha='right', arrowprops=arrow_style)
    ax.annotate('QE3', xy=('2012-09-13', 0), xytext=('2012-09-13', -8), size=12, ha='center', arrowprops=arrow_style)
    ax.annotate('Tapering', xy=('2013-12-18', 0), xytext=('2013-12-18', -8), size=12, ha='center', arrowprops=arrow_style)
    """



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

    #select_processing = param.Selector(['Normal', 'Percentage Change', 'Percentage Change from Year Ago'])

    def __init__(self, parent, ts, **params):
        super().__init__(**params)

        self.ts = ts
        self.parent = parent

        self.select_processing_2 = pn.widgets.Select(name='Processing', options=['Normal', 'Percentage Change', 'Percentage Change from Year Ago'])
        self.select_processing_2.param.watch(self.set_column, 'value')

    #@param.output(('datasource', ColumnDataSource))
    def output(self):
        processing = self.select_processing
        serie_id = self.ts['serie_id']
        df = self.parent.analysis['df']

        modification = False

        if processing == 'Percentage Change':
            column = "{}_{}".format(serie_id, "pct1")
            if not column in df.columns:
                df[column] = serie_pct_change(df, serie_id, periods=1)
                self.ts['column'] = column
                modification = True
        elif processing == 'Percentage Change from Year Ago':
            column = "{}_{}".format(serie_id, "pct12")
            if not column in df.columns:
                df[column] = serie_pct_change(df, serie_id, periods=12)
                self.ts['column'] = column
                modification = True
        elif processing == 'Differentiation':
            column = "{}_{}".format(serie_id, "diff")
            if not column in df.columns:
                df[column] = serie_differentiation(df, serie_id)
                self.ts['column'] = column
                modification = True
        elif processing == 'Normal':
            self.ts['column'] = serie_id
            return self.parent.analysis['datasource']

        if modification:
            self.parent.analysis['df'] = df
            self.parent.analysis['datasource'] = ColumnDataSource(df)

        return self.parent.analysis['datasource']

    def set_column(self, event):
        processing = event.new
        print("** Chart Form: ", processing)
        serie_id = self.ts['serie_id']
        df = self.parent.analysis['df']

        if processing == 'Percentage Change':
            column = "{}_{}".format(serie_id, "pct1")
            if not column in df.columns:
                self.ts['column'] = column
        elif processing == 'Percentage Change from Year Ago':
            column = "{}_{}".format(serie_id, "pct12")
            if not column in df.columns:
                self.ts['column'] = column
        elif processing == 'Differentiation':
            column = "{}_{}".format(serie_id, "diff")
            if not column in df.columns:
                self.ts['column'] = column
        elif processing == 'Normal':
            self.ts['column'] = serie_id

    #@param.depends('select_processing')
    def view(self):
        #data_source = self.output()
        print("******** Chart Form View**************")
        print("************ Chart Form - Set column **************")
        #self.set_column()
        fg, h_hovertool, v_hovertool = create_ts(ts=self.ts, source=self.parent.analysis['datasource'], x_data_range=self.parent.analysis['x_data_range'])
        add_recession_info(fg)
        add_current_time_span(fg)
        add_auto_adjustment_y(fg, self.parent.analysis['df'], self.ts['serie_id'])
        add_sync_crosshair(fg)

        return fg

    def panel(self):
        return pn.Card(pn.Column(self.view, self.select_processing_2), title=self.ts['serie_name'])

    def __repr__(self, *_):
        return "Value"


class AnalysisForm(param.Parameterized):

    action_update_analysis = param.Action(lambda x: x.param.trigger('action_update_analysis'), label='Update Analysis')

    def __init__(self, parent, analysis_name, analysis, **params):
        super().__init__(**params)

        self.analysis_name = analysis_name
        self.analysis = analysis
        self.chart_forms = []
        self.parent = parent
        self.ncols = 3

    def add_chart(self, s):
        chart_form = ChartForm(self, s)
        chart_form.select_processing_2.param.watch(self.update_view, 'value')
        self.chart_forms.append(chart_form)
        return chart_form

    @param.output(ColumnDataSource)
    def get_datasource(self):
        print("**** Analysis Update Data Source *****")
        modification = False
        df = self.analysis['df']

        for c in self.chart_forms:
            processing = c.select_processing_2.value
            print("** Analysis: ", processing)
            serie_id = c.ts['serie_id']
            print("** Analysis Serie: ", serie_id)

            if processing == 'Percentage Change':
                column = "{}_{}".format(serie_id, "pct1")
                if not column in df.columns:
                    df[column] = serie_pct_change(df, serie_id, periods=1)
                    c.ts['column'] = column
                    modification = True
            elif processing == 'Percentage Change from Year Ago':
                column = "{}_{}".format(serie_id, "pct12")
                print("** Analysis Column: ", column)
                if not column in df.columns:
                    print("** Analysis Entro ***: ")
                    df[column] = serie_pct_change(df, serie_id, periods=12)
                    c.ts['column'] = column
                    modification = True
            elif processing == 'Differentiation':
                column = "{}_{}".format(serie_id, "diff")
                if not column in df.columns:
                    df[column] = serie_differentiation(df, serie_id)
                    c.ts['column'] = column
                    modification = True
            elif processing == 'Normal':
                c.ts['column'] = serie_id
                continue

        if modification:
            print("*** Analysis Updated DataSource ***")
            self.analysis['df'] = df
            self.analysis['datasource'] = ColumnDataSource(df)

        print(self.analysis['df'].columns)

        return self.analysis['datasource']

        #self.param.trigger('action_update_analysis')

        #self.param.trigger('update_analysis')

    def update_view(self, event):
        self.param.trigger('action_update_analysis')

    @param.depends('action_update_analysis')
    def view(self):
        print("***** Analysis View *********")
        self.analysis['datasource'] = self.get_datasource()
        #return pn.GridBox(*self.chart_forms, ncols=self.ncols)
        return pn.GridBox(*[c.panel() for c in self.chart_forms], ncols=3)


    def panel(self, plot_by_row):
        print("********** Analysis Panel ****************")
        #self.update_datasource()
        #return pn.GridBox(*[c.panel() for c in self.chart_forms], ncols=plot_by_row)
        return pn.Column(self.param, self.view)

    def __repr__(self, *_):
        return "Value"

def get_tabs(analisis_dict, ncols=3):
    tabs = None
    tuplas = []

    for k in analisis_dict.keys():
        #figs = [pn.Card(f, title=s['serie_name']) for f, s in zip(analisis_dict[k]['figures'], analisis_dict[k]['series'])]
        #figs = [f.panel() for f in analisis_dict[k]['forms']]
        figs = [f.panel() for f in analisis_dict[k]['forms']]
        #tuplas.append((k, pn.Column(*[pn.Row(*c) for c in layout_row(figs, 3)])))
        tuplas.append((k, pn.GridBox(*figs, ncols=ncols)))

    #cross_selector = pn.widgets.CrossSelector(name='Fruits', value=[], options=df_result.columns.to_list())

    #tuplas.append(('Analysis', cross_selector))

    tabs = pn.Tabs(*tuplas, closable=True)

    return tabs


class SeriesForm(param.Parameterized):
    """
    autocomplete_search_serie = pn.widgets.AutocompleteInput(
        name='Search Serie', options=get_name_list_search(),
        placeholder='Ticker or Serie Name', case_sensitive=False)
    """

    plot_by_row = param.Integer(3, bounds=(1, 4))

    autocomplete_search_serie = pn.widgets.TextInput(name='Search Serie', placeholder='Ticker or Serie Name')

    """
    start_datetime_picker = pn.widgets.DatetimePicker(name='Start Date',
                                                      value=datetime.now() - timedelta(days=365 * 10))
    end_datetime_picker = pn.widgets.DatetimePicker(name='End Date', value=datetime.now())
    """

    #checkbox_with_end_date = pn.widgets.Checkbox(name='Disable End Date')

    #action_add_serie = param.Action(lambda x: x.param.trigger('action_add_serie'), label='Add Serie')
    #action_add_analysis = param.Action(lambda x: x.param.trigger('action_add_analysis'), label='Add Analysis')

    button_open_modal = pn.widgets.Button(name='Add Serie', width_policy='fit', height_policy='fit', button_type='primary')
    button_add_serie = pn.widgets.Button(name='Add Serie', width_policy='fit', height_policy='fit', button_type='primary')
    button_create_analisis = pn.widgets.Button(name='New Analysis', button_type='success')

    action_update_tabs = param.Action(lambda x: x.param.trigger('action_update_tabs'), label='Update Tabs')
    action_update_alerts = param.Action(lambda x: x.param.trigger('action_update_alerts'), label='Update Alerts')
    action_update_search_results = param.Action(lambda x: x.param.trigger('action_update_search_results'), label='Update Search Result')

    def __init__(self, **params):
        super().__init__(**params)

        #self.checkbox_with_end_date.param.watch(self.disable_end_date, ['value'], onlychanged=False)

        self.button_add_serie.param.watch(self.add_serie_buttom, 'value')
        self.button_create_analisis.param.watch(self.create_analysis, 'value')
        self.button_open_modal.param.watch(self.open_modal, 'value')
        self.autocomplete_search_serie.param.watch(self._update_countries, 'value')

        self.alerts = []
        self.search_result = []

        self.analysis = []
        """
        self.view = pn.Param(self,
            widgets = {
                "action_add_analysis": {"button_type": "primary"},
                "action_add_serie": {"button_type": "success"},
            }
        )
        """

    def add_analysis(self, analysis_name, analysis):
        analysis = AnalysisForm(parent=self, analysis_name=analysis_name, analysis=analysis)
        self.analysis.append(analysis)
        return analysis

    #def disable_end_date(self, event):
    #    self.end_datetime_picker.disabled = event.new

    def _update_countries(self, event):
        time.sleep(1)
        if event.new != "":
            series = fred.search_for_series([str(event.new)], limit=20)
            self.search_result = []

            for s in series['seriess']:
                t = {"name": s['title'], "id": s['id'], 'observation_start': s['observation_start'],
                                       'observation_end': s['observation_end'], 'frequency': s['frequency'], 'units': s['units'],
                                       'seasonal_adjustment': s['seasonal_adjustment'], 'notes': ''}
                self.search_result.append(t)

            self.param.trigger('action_update_search_results')

    def open_modal(self, event):
        bootstrap.open_modal()

    def add_serie(self, event):
        serie_name = self.autocomplete_search_serie.value.split("-")

        if isinstance(serie_name, list):
            if len(serie_name) == 2:
                pos = [i for i, t in enumerate(list_ts) if t['tick'] == serie_name[1].replace(" ", "")]

                if serie_name != "" and pos[0]:
                    time.sleep(2)
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

    def _get_current_analysis(self):
        return self.analysis[self.tabs.active]

    def add_serie_buttom(self, event):
        serie_id = event.obj.name
        print(serie_id)
        current_analysis = self._get_current_analysis()

        # Get Data and Join data
        df = analisis_dict[current_analysis]['df']
        df = df.set_index('date')

        df_serie = get_fred_dataset(serie_id, rename_column=serie_id)
        #df_serie.fillna(method='ffill', inplace=True)

        df = pd.concat([df, df_serie], axis=1)
        df.fillna(method='ffill', inplace=True)

        #columns = [c for c in df.columns if not re.search("_base$", c)]
        #df[columns] = df[columns].fillna(method='ffill', inplace=True)

        df = df.reset_index()

        # Serie Information
        serie_data = None

        for s in self.search_result:
            if s['id'] == serie_id:
                serie_data = {"serie_id": s['id'], "source": "fred", 'units': s['units'], 'freq': s['frequency'], 'serie_name': s['name']}
                break

        analisis_dict[current_analysis]['series'].append(serie_data)

        # ************************** Set Dataset
        analisis_dict[current_analysis]['df'] = df

        # Update x_data_range
        analisis_dict[current_analysis]['start_date'] = df.date.min()
        analisis_dict[current_analysis]['end_date'] = df.date.max()

        analisis_dict[current_analysis]['x_data_range'] = DataRange1d(start=df.date.min(), end=df.date.max())

        # Update DataSource
        data_source = ColumnDataSource(df)
        analisis_dict[current_analysis]['datasource'] = data_source

        analisis_dict[current_analysis]['series'].append(serie_data)

        # ********************* Set Dataset
        current_analysis.analysis['df'] = df

        # Update x_data_range
        current_analysis.analysis['start_date'] = df.date.min()
        current_analysis.analysis['end_date'] = df.date.max()

        current_analysis.analysis['x_data_range'] = DataRange1d(start=df.date.min(), end=df.date.max())

        # Update DataSource
        current_analysis.analysis['datasource'] = data_source

        self.param.trigger('action_update_tabs')

        bootstrap.close_modal()


    def create_analysis(self, event):
        time.sleep(2)
        name = "Default" + str(random.randint(0, 1000))
        analisis_list.append(name)
        fig_dict[name] = []

        #self.param.trigger('action_update_tabs')

    @param.depends('action_update_alerts', watch=False)
    def get_alerts(self):
        list_alerts = []

        for a in self.alerts:
            list_alerts.append(pn.pane.Alert('## Alert\n{}'.format(a)))

        self.alerts = []

        return list_alerts[0] if len(list_alerts) > 0 else None

    def tabinfo(self, event):
        print("TAB: ", self.tabs.active)

    @param.depends('action_update_tabs', 'plot_by_row', watch=False)
    def get_tabs(self):
        tabs = None
        tuplas = []

        for a in self.analysis:
            # figs = [pn.Card(f, title=s['serie_name']) for f, s in zip(analisis_dict[k]['figures'], analisis_dict[k]['series'])]
            # figs = [f.panel() for f in analisis_dict[k]['forms']]
            #figs = a.panel()
            # tuplas.append((k, pn.Column(*[pn.Row(*c) for c in layout_row(figs, 3)])))
            #tuplas.append((k, pn.GridBox(*figs, ncols=self.plot_by_row)))

            panel = a.panel(self.plot_by_row)

            tuplas.append((a.analysis_name, panel))

        self.tabs = pn.Tabs(*tuplas, closable=True)

        self.tabs.param.watch(self.tabinfo, 'active')

        return self.tabs


    @param.depends('action_update_search_results', watch=False)
    def get_search_results(self):
        rows = []
        description = """
        **{}**\n
        {} to {}\n
        {}\n
        {}
        """

        logo = 'https://fred.stlouisfed.org/images/masthead-88h-2x.png'
        header_color = 'black'
        header_background = '#2f2f2f'
        fed_base_link = "https://fred.stlouisfed.org/series/{}"

        for r in self.search_result:
            button_select = pn.widgets.Button(name=r['id'])
            button_select.param.watch(self.add_serie_buttom, 'value')

            button_open_browser = pn.widgets.Button(name='View', button_type='primary')
            button_open_browser.js_on_click(args={'target': fed_base_link.format(r['id'])}, code='window.open(target.value)')

            rows.append(pn.Card(
                    pn.Column(description.format(r['name'], r['observation_start'], r['observation_end'], r['frequency'], r['notes']), button_open_browser, button_select),
                    header=pn.panel(logo, height=40), header_color=header_color, header_background=header_background)
            )
        return pn.Column("**Search Results:**", pn.GridBox(*rows, ncols=4)) if len(rows) > 0 else None


    def __repr__(self, *_):
        return "Value"


# ******************************************** Create Figures *************************************
series_form = SeriesForm()


for k in analisis_dict.keys():
    series = analisis_dict[k]['series']
    data_source = analisis_dict[k]['datasource']
    x_data_range = analisis_dict[k]['x_data_range']

    current_analysis = series_form.add_analysis(k, analisis_dict[k])

    print("Analisis: {}".format(k))
    print(series)

    tab_id = k

    for s in series:
        print(s)
        current_analysis.add_chart(s)

        """
        fg, h_hovertool, v_hovertool = create_ts(ts=s, source=data_source, x_data_range=x_data_range)
        add_recession_info(fg)
        add_current_time_span(fg)
        add_auto_adjustment_y(fg, analisis_dict[k]['df'], s['serie_id'])
        add_sync_crosshair(fg)

        # Add Figures to Dict
        analisis_dict[tab_id]['figures'].append(fg)

        if h_hovertool:
            analisis_dict[tab_id]['hover_tool_h'].append(h_hovertool)
        if v_hovertool:
            analisis_dict[tab_id]['hover_tool_v'].append(v_hovertool)
        """



alerts = pn.Row(pn.panel(series_form.get_alerts, width=300))
container = pn.Row(pn.panel(series_form.get_tabs, width=300))


bootstrap.sidebar.append(series_form.button_open_modal)
bootstrap.sidebar.append(series_form.button_create_analisis)


bootstrap.sidebar.append(toggle1)
bootstrap.sidebar.append(series_form.param)
bootstrap.sidebar.append(checkbox_crisis)
bootstrap.sidebar.append(checkbox_qe)
bootstrap.sidebar.append(checkbox_qt)
bootstrap.sidebar.append(checkbox_shock_events)
bootstrap.sidebar.append(checkbox_enable_highlight)
bootstrap.sidebar.append(checkbox_enable_syncronize_chart)
bootstrap.sidebar.append(checkbox_enable_vertical_tooltip)
bootstrap.sidebar.append(checkbox_enable_horizontal_tooltip)

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

bootstrap.modal.append(pn.Column(pn.Row(series_form.autocomplete_search_serie), series_form.get_search_results))

bootstrap.main.append(alerts)
bootstrap.main.append(container)

bootstrap.show()

