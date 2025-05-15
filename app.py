from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import logging
import os

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
log_format = "%(asctime)s - %(levelname)s - %(message)s"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/app.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- Flask App Setup ---
app = Flask(__name__)

# --- Global Plotly Template (center all chart titles) ---
base = pio.templates["plotly_white"].to_plotly_json()

base.setdefault("layout", {})
base["layout"].setdefault("title", {})

base["layout"]["title"].update({
    "x": 0.5,
    "xanchor": "center"
})

pio.templates["centered"] = base
pio.templates.default = "centered"


# --- Load Random Forest Model ---
with open("models/RandomForest_pipeline_model.pkl", "rb") as f:
    rf_model = pickle.load(f)


# --- Load Dataset ---
df = pd.read_csv("data/cleaned/Cleaned_Data.csv")


FEATURES = [
    "Page Views",
    "Session Duration",
    "Bounce Rate",
    "Traffic Source",
    "Time on Page",
    "Previous Visits"
]


# --- Baseline Visualizations ---
def create_graphs():
    graphs = []

    # 1) Feature Importances
    if hasattr(rf_model, "named_steps"):
        # Adjusted to access the correct step name
        actual = rf_model.named_steps.get("regressor", None)

        if actual and hasattr(actual, "feature_importances_"):
            imps = actual.feature_importances_
            names = getattr(actual, "feature_names_in_", FEATURES)

            # Ensure names and imps are of the same length
            min_len = min(len(names), len(imps))
            names = names[:min_len]
            imps = imps[:min_len]

            imp_df = pd.DataFrame({
                "Feature": names,
                "Importance": imps
            })

            fig = px.bar(
                imp_df, x="Importance", y="Feature",
                orientation="h", title="Feature Importances by Random Forest"
            )

            graphs.append(pio.to_html(fig, full_html=False))
        else:
            logger.warning("No feature importances found; skipping that graph.")

    # 2) Actual vs Predicted (RF)
    y_true = df["Conversion Rate"]
    y_pred_rf = rf_model.predict(df[FEATURES])

    fig = px.scatter(
        x=y_true, y=y_pred_rf,
        labels={"x": "Actual Conversion Rate", "y": "Predicted Conversion Rate"},
        title="Actual vs Predicted Conversion Rate (Random Forest)"
    )

    fig.add_shape(
        type="line",
        x0=y_true.min(), y0=y_true.min(),
        x1=y_true.max(), y1=y_true.max(),
        line=dict(dash="dash")
    )

    graphs.append(pio.to_html(fig, full_html=False))

    # 3) Conversion Rate Distribution
    fig = px.histogram(df, x="Conversion Rate", nbins=30, title="Conversion Rate Distribution")

    graphs.append(pio.to_html(fig, full_html=False))

    # 4) Conversion by Traffic Source
    fig = px.box(df, x="Traffic Source", y="Conversion Rate", title="Conversion Rate by Traffic Source")

    graphs.append(pio.to_html(fig, full_html=False))

    # 5) Bounce vs Conversion
    fig = px.scatter(
        df, x="Bounce Rate", y="Conversion Rate",
        trendline="ols", title="Bounce Rate vs Conversion Rate"
    )

    graphs.append(pio.to_html(fig, full_html=False))

    # 6) Session Duration vs Conversion
    fig = px.scatter(
        df, x="Session Duration", y="Conversion Rate",
        trendline="ols", title="Session Duration vs Conversion Rate"
    )

    graphs.append(pio.to_html(fig, full_html=False))

    # 7) Correlation Heatmap
    corr = df[FEATURES + ["Conversion Rate"]].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap")

    graphs.append(pio.to_html(fig, full_html=False))

    # 8) Avg Conversion by Visits
    avg_df = df.groupby("Previous Visits")["Conversion Rate"].mean().reset_index()
    fig = px.bar(
        avg_df, x="Previous Visits", y="Conversion Rate",
        title="Average Conversion Rate by Previous Visits"
    )

    graphs.append(pio.to_html(fig, full_html=False))

    return graphs

# --- Post-Prediction Visualizations ---
def generate_visualizations(pred_rf):
    visuals = []

    # Residuals
    y_true = df["Conversion Rate"]
    y_pred = rf_model.predict(df[FEATURES])

    resid = y_true - y_pred
    res_fig = go.Figure(go.Scatter(
        x=y_pred, y=resid, mode="markers", name="Residuals"
    ))

    res_fig.update_layout(
        title="Random Forest Residuals",
        xaxis_title="Predicted Conversion Rate",
        yaxis_title="Residual (Actual − Predicted)"
    )

    visuals.append(pio.to_html(res_fig, full_html=False))

    return visuals

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def dashboard():
    return render_template(
        "dashboard.html",
        baseline_graphs=create_graphs(),
        post_visuals=[],
        prediction_rf=None,
        error=None,
        FEATURES=FEATURES
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    baseline = create_graphs()

    try:
        # 1) Parse inputs
        inp = {
            f: float(request.form[f])
            for f in FEATURES
        }

        logger.info(f"Received input values: {inp}")

        # 2) DataFrame for pipeline
        df_in = pd.DataFrame([inp], columns=FEATURES)

        # 3) Raw model output
        raw_rf = rf_model.predict(df_in)[0]

        logger.info(f"Raw prediction (0-1 scale) — RF: {raw_rf:.4f}")

        # 4) Inverse‐transform target if wrapped
        if hasattr(rf_model, "named_steps") and "target_transformer" in rf_model.named_steps:
            raw_rf = rf_model.named_steps["target_transformer"].inverse_transform([raw_rf])[0]

        # 5) Convert to percentage and round to 2 decimal places
        pred_rf = round(raw_rf * 100, 2)

        logger.info(f"Final prediction: RF: {pred_rf: .2f}%")
        
        # 6) Generate visuals
        post = generate_visualizations(raw_rf)

        return render_template(
            "dashboard.html",
            FEATURES=FEATURES,
            baseline_graphs=baseline,
            post_visuals=post,
            prediction_rf=pred_rf,
            error=None
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")

        return render_template(
            "dashboard.html",
            FEATURES=FEATURES,
            baseline_graphs=baseline,
            post_visuals=[],
            prediction_rf=None,
            error=f"Input error: {e}",
        )

if __name__ == "__main__":
    app.run(debug=True)