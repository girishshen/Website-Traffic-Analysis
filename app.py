import os
import logging

from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Flask App Initialization
# ------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------
try:
    df = pd.read_csv("data/raw/website_wata.csv")
    df.columns = df.columns.str.strip()
    logger.info(f"Loaded data with {len(df)} records from {df}")
except Exception as e:
    logger.exception(f"Failed to load CSV at {df}: {e}")
    raise

# ------------------------------------------------------------
# Numeric Features for Analysis
# ------------------------------------------------------------
NUMERIC_FEATURES = [
    'Page Views',
    'Session Duration',
    'Bounce Rate',
    'Time on Page',
    'Previous Visits',
    'Conversion Rate',
]
CATEGORY_FEATURE = 'Traffic Source'

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route('/')
def dashboard():
    # Helper to convert figures to HTML snippets
    def to_html(fig):
        return fig.to_html(full_html=False)

    # 1) Univariate Analysis: histograms for each numeric feature
    histograms = {}
    for feature in NUMERIC_FEATURES:
        fig = px.histogram(
            df, x=feature, nbins=50,
            title=f'{feature} Distribution'
        )

        histograms[feature] = to_html(fig)

    # 2) Multivariate Analysis: correlation heatmap of numeric features
    corr = df[NUMERIC_FEATURES].corr()
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='Viridis'
        )
    )

    fig_corr.update_layout(title='Correlation Matrix')
    graph_corr = to_html(fig_corr)

    # 3) Scatter Plot: Session Duration vs Conversion Rate
    fig_scatter = px.scatter(
        df, x='Session Duration', y='Conversion Rate', trendline='ols',
        title='Session Duration vs Conversion Rate'
    )

    graph_scatter = to_html(fig_scatter)

    # 4) Traffic Source Distribution (Pie Chart)
    source_counts = df[CATEGORY_FEATURE].value_counts()
    fig_pie = px.pie(
        names=source_counts.index, values=source_counts.values,
        title='Traffic Source Distribution'
    )

    graph_pie = to_html(fig_pie)

    # 5) Session Duration by Traffic Source (Box Plot)
    fig_box = px.box(
        df, x=CATEGORY_FEATURE, y='Session Duration',
        title='Session Duration by Traffic Source'
    )

    graph_box = to_html(fig_box)

    # 6) Page Views vs Time on Page by Traffic Source (Scatter)
    fig_scatter2 = px.scatter(
        df, x='Page Views', y='Time on Page',
        color=CATEGORY_FEATURE,
        title='Page Views vs Time on Page by Traffic Source'
    )

    graph_scatter2 = to_html(fig_scatter2)

    # Render the dashboard template with all graphs
    return render_template(
        'dashboard.html',
        histograms=histograms,
        graph_corr=graph_corr,
        graph_scatter=graph_scatter,
        graph_pie=graph_pie,
        graph_box=graph_box,
        graph_scatter2=graph_scatter2
    )

# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)