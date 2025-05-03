import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import interpolate
import time

# Page config with expanded sidebar by default
st.set_page_config(page_title="RRG Indicator", layout="wide", initial_sidebar_state="expanded")

# Initialize session state for animation
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0
if 'enabled_tickers' not in st.session_state:
    st.session_state.enabled_tickers = None
if 'show_date_overlay' not in st.session_state:
    st.session_state.show_date_overlay = True

# Title and description
st.title('RRG Indicator')
st.write('An interactive visualization tool for Relative Rotation Graphs (RRG)')

# Initialize data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_data(period='1y', interval='1wk'):
    tickers = ['^CNXAUTO', '^CNXFMCG', '^CNXIT', '^CNXREALTY', '^CNXCONSUM', 
               '^CNXMETAL', '^CNXENERGY', '^CNXMEDIA', '^CNXINFRA', '^CNXPSUBANK', '^NSEBANK']
    benchmark = '^NSEI'

    # Adjust window size based on interval
    window_sizes = {
        '1h': 24,   # 24 hours
        '2h': 12,   # 24 hours
        '4h': 6,    # 24 hours
        '1d': 14,   # 14 days
        '1wk': 14   # 14 weeks
    }
    window = window_sizes.get(interval, 14)

    tickers_data = yf.download(tickers, period=period, interval=interval, auto_adjust=False, multi_level_index=False)['Adj Close']
    benchmark_data = yf.download(benchmark, period=period, interval=interval, auto_adjust=False, multi_level_index=False)['Adj Close']

    return tickers, tickers_data, benchmark_data, window

# Period selection with validation
st.sidebar.title('📊 Chart Controls')
st.sidebar.markdown("""
Use these controls to adjust the visualization:
""")

st.sidebar.subheader('📅 Time Period')
period_options = {
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y',
    '2 Years': '2y',
    '5 Years': '5y'
}
selected_period = st.sidebar.selectbox(
    'Select analysis period:',
    options=list(period_options.keys()),
    index=2,  # Default to 1 Year
    help='Choose the time period for analysis.'
)

# Add interval selection for shorter periods
interval = '1wk'  # default interval
if period_options[selected_period] in ['3mo', '6mo']:
    interval_options = {
        'Hourly (1h)': '1h',
        '2 Hours (2h)': '2h',
        '4 Hours (4h)': '4h',
        'Daily (1d)': '1d',
        'Weekly (1wk)': '1wk'
    }
    selected_interval = st.sidebar.selectbox(
        'Select data interval:',
        options=list(interval_options.keys()),
        index=3,  # Default to Daily
        help='Choose the interval between data points'
    )
    interval = interval_options[selected_interval]

    # Add informational message about data points
    st.sidebar.info(f"Using {selected_interval.lower()} data for analysis. Adjust interval if needed for more/fewer data points.")
else:
    st.sidebar.info("Using weekly data for longer time periods.")

# Fetch data based on selected period and interval
tickers, tickers_data, benchmark_data, window = fetch_data(period_options[selected_period], interval)


# Function to calculate RS (Relative Strength)
def calculate_rsr_and_rsm():
    rs_tickers = []
    rsr_tickers = []
    rsr_roc_tickers = []
    rsm_tickers = []

    for i in range(len(tickers)):
        # Calculate relative strength
        rs_tickers.append(100 * (tickers_data[tickers[i]] / benchmark_data))

        # Calculate RS-Ratio with error handling
        if len(rs_tickers[i]) < window:
            st.error(f"Not enough data points for {tickers[i]}. Need at least {window} weeks of data, but only found {len(rs_tickers[i])} weeks.")
            continue

        rsr = (100 + (rs_tickers[i] - rs_tickers[i].rolling(window=window).mean()) / 
               rs_tickers[i].rolling(window=window).std(ddof=0)).dropna()
        rsr_tickers.append(rsr)

        # Calculate ROC using first available point
        if len(rsr) > 0:
            rsr_roc = 100 * ((rsr / rsr.iloc[0]) - 1)
            rsr_roc_tickers.append(rsr_roc)

            # Calculate RS-Momentum
            if len(rsr_roc) >= window:
                rsm = (101 + ((rsr_roc - rsr_roc.rolling(window=window).mean()) / 
                            rsr_roc.rolling(window=window).std(ddof=0))).dropna()
                rsm_tickers.append(rsm)

                # Align the data points
                rsr_tickers[i] = rsr[rsr.index.isin(rsm.index)]
                rsm_tickers[i] = rsm[rsm.index.isin(rsr.index)]
            else:
                st.warning(f"Limited data points for {tickers[i]}. Results may be less reliable.")
                return [], [], []
        else:
            st.error(f"No valid data available for {tickers[i]}. Please select a longer time period.")
            return [], [], []

    return rs_tickers, rsr_tickers, rsm_tickers

rs_tickers, rsr_tickers, rsm_tickers = calculate_rsr_and_rsm()

# Status and color functions
def get_status(x, y):
    if x < 100 and y < 100:
        return 'lagging'
    elif x > 100 and y > 100:
        return 'leading'
    elif x < 100 and y > 100:
        return 'improving'
    elif x > 100 and y < 100:
        return 'weakening'

def get_color(x, y):
    status = get_status(x, y)
    if status == 'lagging':
        return 'red'
    elif status == 'leading':
        return 'green'
    elif status == 'improving':
        return 'blue'
    elif status == 'weakening':
        return 'yellow'

# Initialize or get enabled tickers from session state
if st.session_state.enabled_tickers is None:
    st.session_state.enabled_tickers = {ticker: True for ticker in tickers}

# Add ticker visibility controls
st.sidebar.subheader('📊 Ticker Visibility')
st.sidebar.markdown('Toggle which tickers to display')

# Update enabled tickers based on checkbox state
for ticker in tickers:
    checked = st.sidebar.checkbox(ticker, value=st.session_state.enabled_tickers[ticker], key=f'ticker_{ticker}')
    st.session_state.enabled_tickers[ticker] = checked

# Add animation controls
st.sidebar.subheader('🎬 Animation Controls')
col1, col2 = st.sidebar.columns(2)
with col1:
    play_button = st.button('⏸️ Pause' if st.session_state.playing else '▶️ Play')
with col2:
    animation_speed = st.select_slider('Speed', options=['Slow', 'Normal', 'Fast'], value='Normal')

if play_button:
    st.session_state.playing = not st.session_state.playing

# Add more descriptive controls with help text
st.sidebar.subheader('🔄 Trail Length')
st.sidebar.markdown('Adjust how many historical points to show in the trail')
tail = st.sidebar.slider(
    'Select number of points to display in the trail:',
    min_value=1,
    max_value=10,
    value=5,
    help='Longer tail shows more historical movement'
)

st.sidebar.subheader('📅 End Date')
st.sidebar.markdown('Choose the specific end date for the analysis')

# Add display options
st.sidebar.subheader('🎨 Display Options')
show_date = st.sidebar.checkbox('Show Date Overlay', value=st.session_state.show_date_overlay)
st.session_state.show_date_overlay = show_date


# Get dates list for the slider
dates = rsr_tickers[0].index.strftime('%Y-%m-%d').tolist() if len(rsr_tickers) > 0 and len(rsr_tickers[0]) > 0 else []

# Validate and adjust tail based on available data
if len(dates) < tail:
    tail = max(1, len(dates) - 1)
    st.warning(f"Adjusting trail length to {tail} due to limited data points in selected period")

# Always show the end date slider with proper validation
if len(dates) > tail:
    end_date_idx = st.sidebar.slider(
        'Select the end date:',
        min_value=tail,
        max_value=len(dates) - 1,
        value=len(dates) - 1,
        format=None  # Don't use a format string
    )
else:
    st.error("Not enough data points available for the selected period. Please choose a longer time period.")
    end_date_idx = 0

if st.session_state.playing:
    # Update current frame when playing
    max_frame = max(0, len(dates) - tail - 1)
    if max_frame > 0:
        st.session_state.current_frame = (st.session_state.current_frame + 1) % max_frame
        end_date_idx = st.session_state.current_frame + tail
    else:
        st.session_state.playing = False
else:
    st.session_state.current_frame = max(0, end_date_idx - tail)

# Update RRG plot
def create_rrg_plot(tail, end_date_idx):
    start_date = rsr_tickers[0].index[end_date_idx - tail] if end_date_idx - tail >= 0 and len(rsr_tickers) > 0 and len(rsr_tickers[0]) > 0 else rsr_tickers[0].index[0] if len(rsr_tickers) > 0 and len(rsr_tickers[0]) > 0 else None
    end_date = rsr_tickers[0].index[end_date_idx] if len(rsr_tickers) > 0 and len(rsr_tickers[0]) > 0 else None

    scatter_traces = []

    for i, ticker in enumerate(tickers):
        # Skip disabled tickers
        if not st.session_state.enabled_tickers[ticker]:
            continue

        filtered_rsr_tickers = rsr_tickers[i].loc[(rsr_tickers[i].index > start_date) & 
                                                (rsr_tickers[i].index <= end_date)] if start_date is not None and end_date is not None else pd.Series()
        filtered_rsm_tickers = rsm_tickers[i].loc[(rsm_tickers[i].index > start_date) & 
                                                (rsm_tickers[i].index <= end_date)] if start_date is not None and end_date is not None else pd.Series()

        if len(filtered_rsr_tickers) == 0 or len(filtered_rsm_tickers) == 0:
            continue

        color = get_color(filtered_rsr_tickers.values[-1], filtered_rsm_tickers.values[-1])

        # Create single trace with different markers for trail and current point
        marker_symbols = ['circle'] * (len(filtered_rsr_tickers) - 1) + ['triangle-up']
        marker_sizes = [10] * (len(filtered_rsr_tickers) - 1) + [15]

        scatter_traces.append(go.Scatter(
            x=filtered_rsr_tickers.values,
            y=filtered_rsm_tickers.values,
            mode='markers+lines',
            marker=dict(
                size=marker_sizes,
                color=color,
                symbol=marker_symbols
            ),
            name=ticker,
            hovertext=[f"{ticker} (historical)" if i < len(filtered_rsr_tickers)-1 else f"{ticker} (current)" 
                      for i in range(len(filtered_rsr_tickers))],
            hoverinfo="text",
            showlegend=True
        ))

    # Create layout with optional date overlay
    layout = go.Layout(
        title=dict(
            text='RRG Indicator' + (f' - {end_date.strftime("%Y-%m-%d")}' if st.session_state.show_date_overlay and end_date is not None else ''),
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(title='RS Ratio', range=[94, 106]),
        yaxis=dict(title='RS Momentum', range=[94, 106]),
        shapes=[
            {'type': 'line', 'x0': 100, 'x1': 100, 'y0': 94, 'y1': 106, 
             'line': {'color': 'black', 'dash': 'dash'}},
            {'type': 'line', 'x0': 94, 'x1': 106, 'y0': 100, 'y1': 100, 
             'line': {'color': 'black', 'dash': 'dash'}},
            {'type': 'rect', 'x0': 94, 'x1': 100, 'y0': 94, 'y1': 100, 
             'fillcolor': 'red', 'opacity': 0.2, 'layer': "below"},
            {'type': 'rect', 'x0': 100, 'x1': 106, 'y0': 94, 'y1': 100, 
             'fillcolor': 'yellow', 'opacity': 0.2, 'layer': "below"},
            {'type': 'rect', 'x0': 100, 'x1': 106, 'y0': 100, 'y1': 106, 
             'fillcolor': 'green', 'opacity': 0.2, 'layer': "below"},
            {'type': 'rect', 'x0': 94, 'x1': 100, 'y0': 100, 'y1': 106, 
             'fillcolor': 'blue', 'opacity': 0.2, 'layer': "below"}
        ],
        annotations=[
            {'x': 95, 'y': 105, 'text': 'Improving', 'showarrow': False, 'font': {'size': 14}},
            {'x': 104, 'y': 105, 'text': 'Leading', 'showarrow': False, 'font': {'size': 14}},
            {'x': 104, 'y': 95, 'text': 'Weakening', 'showarrow': False, 'font': {'size': 14}},
            {'x': 95, 'y': 95, 'text': 'Lagging', 'showarrow': False, 'font': {'size': 14}}
        ]
    )

    figure = go.Figure(data=scatter_traces, layout=layout)
    return figure, start_date, end_date

# Create and display plot
figure, start_date, end_date = create_rrg_plot(tail, end_date_idx)
chart = st.plotly_chart(figure, use_container_width=True)

# Create and display table
st.subheader('Financial Metrics')

# Create DataFrame for table
table_data = []
for i, ticker in enumerate(tickers):
    if st.session_state.enabled_tickers[ticker]:  # Only show enabled tickers in the table
        price = np.round(tickers_data[ticker][end_date], 2) if end_date is not None else None
        change = np.round((tickers_data[ticker][end_date] - tickers_data[ticker][start_date]) / 
                      tickers_data[ticker][start_date] * 100, 1) if start_date is not None and end_date is not None else None

        table_data.append({
            'Symbol': ticker,
            'Name': ticker,  # Using symbol as name since actual names aren't available
            'Price': price,
            'Change (%)': change
        })

df = pd.DataFrame(table_data)

# Style the dataframe
def color_negative_red(val):
    if isinstance(val, (int, float)):
        color = 'red' if val < 0 else 'green'
        return f'color: {color}'
    return ''

styled_df = df.style.map(color_negative_red, subset=['Change (%)'])
st.dataframe(styled_df, use_container_width=True)

# Auto-rerun when playing
if st.session_state.playing:
    speed_delays = {'Slow': 1.0, 'Normal': 0.7, 'Fast': 0.3}
    time.sleep(speed_delays[animation_speed])
    st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
