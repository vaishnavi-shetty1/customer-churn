from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predict import (
    DEFAULT_CUSTOMER,
    predict_churn,
    load_metadata,
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "customer_churn.csv"
MODEL_DIR = ROOT_DIR / "models"
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
INTERNET_ORDER = ["DSL", "Fiber optic", "No"]
PAYMENT_ORDER = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
TENURE_ORDER = [
    "0–6 months",
    "7–12 months",
    "13–24 months",
    "25–48 months",
    "49–72 months",
]
FILTER_KEYS = [
    "contract_filter",
    "internet_filter",
    "payment_filter",
    "senior_filter",
    "tenure_filter",
    "charges_filter",
]

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 3px 14px rgba(15, 23, 42, 0.06);
    }
    [data-testid="stMetricLabel"] {color: #475569;}
    div[data-testid="stTabs"] button {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading customer data…")
def load_dashboard_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["ChurnFlag"] = data["Churn"].eq("Yes").astype(int)
    data["SeniorCitizenLabel"] = data["SeniorCitizen"].map({0: "Non-senior", 1: "Senior"})
    data["TenureBand"] = pd.cut(
        data["tenure"],
        bins=[-1, 6, 12, 24, 48, 73],
        labels=TENURE_ORDER,
        ordered=True,
    )
    data["RevenueAtRisk"] = data["MonthlyCharges"] * data["ChurnFlag"]
    return data


def reset_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


def currency(value: float) -> str:
    return f"${value:,.0f}"


def percent(value: float) -> str:
    return f"{value:.1%}"


def segment_summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        data.groupby(column, observed=True, dropna=False)
        .agg(
            Customers=("ChurnFlag", "size"),
            Churned=("ChurnFlag", "sum"),
            MonthlyRevenue=("MonthlyCharges", "sum"),
            RevenueAtRisk=("RevenueAtRisk", "sum"),
        )
        .reset_index()
    )
    summary["ChurnRate"] = (summary["Churned"] / summary["Customers"].replace(0, pd.NA)).fillna(0)
    return summary


def make_donut(data: pd.DataFrame) -> go.Figure:
    counts = data["Churn"].value_counts()
    figure = go.Figure(
        go.Pie(
            labels=["Retained", "Churned"],
            values=[int(counts.get("No", 0)), int(counts.get("Yes", 0))],
            hole=0.68,
            marker=dict(
                colors=["#0f766e", "#e11d48"],
                line=dict(color="#ffffff", width=3),
            ),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>",
        )
    )
    figure.add_annotation(
        text=f"<b>{data['ChurnFlag'].mean():.1%}</b><br>churn rate",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="#0f172a"),
    )
    figure.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=15, b=55),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.02),
    )
    return figure


def make_churn_bar(data: pd.DataFrame, column: str) -> go.Figure:
    summary = segment_summary(data, column).sort_values("ChurnRate")
    figure = px.bar(
        summary,
        x="ChurnRate",
        y=column,
        orientation="h",
        color="ChurnRate",
        color_continuous_scale=["#14b8a6", "#f59e0b", "#ef4444"],
        custom_data=["Customers", "Churned"],
        labels={column: "", "ChurnRate": "Churn rate"},
    )
    figure.update_traces(
        texttemplate="%{x:.1%}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>Churn rate: %{x:.1%}<br>"
            "Customers: %{customdata[0]:,.0f}<br>"
            "Churned: %{customdata[1]:,.0f}<extra></extra>"
        ),
    )
    figure.update_layout(
        height=max(330, 68 * len(summary) + 110),
        margin=dict(l=10, r=45, t=15, b=15),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(
        title="Churn rate",
        tickformat=".0%",
        range=[0, max(0.1, float(summary["ChurnRate"].max()) * 1.2)],
        gridcolor="#e2e8f0",
    )
    figure.update_yaxes(showgrid=False, categoryorder="array", categoryarray=summary[column].tolist())
    return figure


def make_tenure_chart(data: pd.DataFrame) -> go.Figure:
    summary = segment_summary(data, "TenureBand")
    summary = summary.set_index("TenureBand").reindex(TENURE_ORDER, fill_value=0).reset_index()
    figure = go.Figure()
    figure.add_bar(
        x=summary["TenureBand"],
        y=summary["Customers"],
        name="Customers",
        marker_color="#0f766e",
        hovertemplate="<b>%{x}</b><br>Customers: %{y:,.0f}<extra></extra>",
    )
    figure.add_scatter(
        x=summary["TenureBand"],
        y=summary["ChurnRate"],
        name="Churn rate",
        mode="lines+markers",
        marker=dict(color="#e11d48", size=8),
        line=dict(color="#e11d48", width=3),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Churn rate: %{y:.1%}<extra></extra>",
    )
    figure.update_layout(
        height=390,
        barmode="group",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis2=dict(
            title="Churn rate",
            overlaying="y",
            side="right",
            range=[0, 1],
            tickformat=".0%",
            showgrid=False,
        ),
    )
    figure.update_yaxes(title="Customers", rangemode="tozero", gridcolor="#e2e8f0")
    figure.update_xaxes(showgrid=False)
    return figure


def make_revenue_chart(data: pd.DataFrame) -> go.Figure:
    summary = segment_summary(data, "Contract")
    summary = summary.set_index("Contract").reindex(CONTRACT_ORDER).dropna().reset_index()
    figure = px.bar(
        summary,
        x="Contract",
        y=["MonthlyRevenue", "RevenueAtRisk"],
        barmode="group",
        color_discrete_map={
            "MonthlyRevenue": "#0f766e",
            "RevenueAtRisk": "#fb7185",
        },
        labels={
            "Contract": "",
            "MonthlyRevenue": "Monthly revenue",
            "RevenueAtRisk": "Revenue at risk",
        },
    )
    figure.update_traces(
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: $%{y:,.0f}<extra></extra>"
    )
    figure.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_yaxes(title="Monthly revenue", tickprefix="$", gridcolor="#e2e8f0")
    figure.update_xaxes(showgrid=False)
    return figure


def show_image(filename: str, caption: str) -> None:
    path = MODEL_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Run `python train_model.py` to generate {filename}.")


def customer_signals(customer: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    if customer["Contract"] == "Month-to-month":
        signals.append("Month-to-month contract")
    if customer["InternetService"] == "Fiber optic":
        signals.append("Fiber optic internet service")
    if customer["PaymentMethod"] == "Electronic check":
        signals.append("Electronic check payment")
    if customer["tenure"] <= 12:
        signals.append("Tenure of 12 months or less")
    if customer["SeniorCitizen"] == 1:
        signals.append("Senior customer")
    if not signals:
        signals.append("No high-signal profile flags from the selected inputs")
    return signals


def render_prediction_result(result: dict[str, Any], customer: dict[str, Any]) -> None:
    probability = result["churn_probability"]
    risk_color = {"Low risk": "#0f766e", "Medium risk": "#f59e0b", "High risk": "#e11d48"}[
        result["risk_level"]
    ]
    st.subheader("Prediction result")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Churn probability", percent(probability))
    metric_columns[1].metric("Retention probability", percent(result["retention_probability"]))
    metric_columns[2].metric("Risk level", result["risk_level"])
    metric_columns[3].metric("Model decision", result["outcome"])
    if result["prediction"] == 1:
        st.error("This customer is classified as likely to churn.")
    else:
        st.success("This customer is classified as likely to stay.")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 30], "color": "#ccfbf1"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#ffe4e6"},
                ],
                "threshold": {"line": {"color": "#0f172a", "width": 2}, "value": probability * 100},
            },
        )
    )
    gauge.update_layout(
        height=270,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})
    st.progress(probability, text=f"Estimated churn risk: {percent(probability)}")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Customer summary")
        summary = pd.DataFrame(
            [
                {"Field": "Tenure", "Value": f"{customer['tenure']} months"},
                {"Field": "Monthly charges", "Value": currency(customer["MonthlyCharges"])},
                {"Field": "Contract", "Value": customer["Contract"]},
                {"Field": "Internet service", "Value": customer["InternetService"]},
                {"Field": "Payment method", "Value": customer["PaymentMethod"]},
            ]
        )
        st.dataframe(summary, hide_index=True, width="stretch")
        st.download_button(
            "Download customer summary",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="customer_summary.csv",
            mime="text/csv",
            width="stretch",
        )
    with right:
        st.subheader("Profile signals")
        for signal in customer_signals(customer):
            st.markdown(f"- {signal}")
        st.caption("These are descriptive input signals, not causal explanations from the model.")
    st.info(
        "Use the prediction to prioritize a retention conversation, not as a guaranteed outcome. "
        "The model is trained on historical customer records."
    )


def render_model_performance(metadata: dict[str, Any] | None) -> None:
    if metadata is None:
        st.error("Model artifacts are not available. Run `python train_model.py` first.")
        return
    st.subheader("Model comparison")
    metric_rows = []
    for model_name, result in metadata["models"].items():
        metrics = result["metrics"]
        metric_rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"] * 100,
                "Precision": metrics["precision"] * 100,
                "Recall": metrics["recall"] * 100,
                "F1 score": metrics["f1_score"] * 100,
                "ROC-AUC": metrics["roc_auc"] * 100,
            }
        )
    metric_table = pd.DataFrame(metric_rows)
    st.dataframe(
        metric_table,
        hide_index=True,
        width="stretch",
        column_config={
            column: st.column_config.NumberColumn(format="%.2f%%")
            for column in metric_table.columns
            if column != "Model"
        },
    )
    st.success(
        f"Selected model: {metadata['selected_model']} based on {metadata['selection_metric'].replace('_', ' ')}."
    )
    st.caption(
        f"Logistic regression best parameters: {metadata['best_logistic_regression_params']} · "
        f"Test rows: {metadata['split']['testing_rows']:,}"
    )
    chart_columns = st.columns(2)
    with chart_columns[0]:
        show_image("model_comparison.png", "Logistic regression versus random forest")
    with chart_columns[1]:
        show_image("confusion_matrix.png", "Selected model confusion matrix")
    chart_columns = st.columns(2)
    with chart_columns[0]:
        show_image("roc_curve.png", "ROC curve comparison")
    with chart_columns[1]:
        show_image("precision_recall_curve.png", "Precision-recall curve comparison")
    matrix = metadata["confusion_matrix"]
    true_negative, false_positive, false_negative, true_positive = (
        matrix[0][0],
        matrix[0][1],
        matrix[1][0],
        matrix[1][1],
    )
    st.subheader("Selected model outcomes")
    outcome_columns = st.columns(4)
    outcome_columns[0].metric("True negatives", f"{true_negative:,}")
    outcome_columns[1].metric("False positives", f"{false_positive:,}")
    outcome_columns[2].metric("False negatives", f"{false_negative:,}")
    outcome_columns[3].metric("True positives", f"{true_positive:,}")


def render_eda(metadata: dict[str, Any] | None) -> None:
    st.subheader("Exploratory data analysis")
    if metadata is not None:
        dataset = metadata["dataset"]
        st.caption(
            f"{dataset['cleaned_rows']:,} cleaned records · "
            f"{dataset['retained_customers']:,} retained · "
            f"{dataset['churned_customers']:,} churned · "
            f"{dataset['removed_missing_total_charges']:,} records removed for missing TotalCharges"
        )
    eda_columns = st.columns(2)
    with eda_columns[0]:
        show_image("churn_distribution.png", "Churn distribution")
    with eda_columns[1]:
        show_image("churn_by_contract.png", "Churn by contract")
    eda_columns = st.columns(2)
    with eda_columns[0]:
        show_image("churn_by_internet_service.png", "Churn by internet service")
    with eda_columns[1]:
        show_image("feature_importance.png", "Random forest feature importance")
    eda_columns = st.columns(2)
    with eda_columns[0]:
        show_image("tenure_vs_churn.png", "Tenure by churn status")
    with eda_columns[1]:
        show_image("monthly_charges_vs_churn.png", "Monthly charges by churn status")


try:
    dashboard_data = load_dashboard_data()
    metadata = load_metadata()
except FileNotFoundError as error:
    dashboard_data = None
    metadata = None
    st.warning(str(error))

st.sidebar.title("Customer Churn")
st.sidebar.caption("Machine learning for retention decisions")
if metadata is None:
    st.sidebar.warning("Model not trained")
else:
    st.sidebar.success(f"{metadata['selected_model']} ready")
if dashboard_data is not None:
    st.sidebar.info(f"{len(dashboard_data):,} source records loaded")

st.sidebar.divider()
st.sidebar.subheader("Dashboard filters")
st.sidebar.button("Reset filters", on_click=reset_filters, width="stretch")
if dashboard_data is not None:
    selected_contracts = st.sidebar.multiselect(
        "Contract",
        CONTRACT_ORDER,
        default=CONTRACT_ORDER,
        key="contract_filter",
    )
    selected_internet = st.sidebar.multiselect(
        "Internet service",
        INTERNET_ORDER,
        default=INTERNET_ORDER,
        key="internet_filter",
    )
    selected_payments = st.sidebar.multiselect(
        "Payment method",
        PAYMENT_ORDER,
        default=PAYMENT_ORDER,
        key="payment_filter",
    )
    selected_senior = st.sidebar.radio(
        "Senior citizen",
        ["All", "Non-senior", "Senior"],
        key="senior_filter",
    )
    selected_tenure = st.sidebar.slider(
        "Tenure (months)",
        min_value=int(dashboard_data["tenure"].min()),
        max_value=int(dashboard_data["tenure"].max()),
        value=(int(dashboard_data["tenure"].min()), int(dashboard_data["tenure"].max())),
        key="tenure_filter",
    )
    selected_charges = st.sidebar.slider(
        "Monthly charge",
        min_value=float(dashboard_data["MonthlyCharges"].min()),
        max_value=float(dashboard_data["MonthlyCharges"].max()),
        value=(float(dashboard_data["MonthlyCharges"].min()), float(dashboard_data["MonthlyCharges"].max())),
        step=1.0,
        key="charges_filter",
    )
else:
    selected_contracts = []
    selected_internet = []
    selected_payments = []
    selected_senior = "All"
    selected_tenure = (0, 100)
    selected_charges = (0.0, 1000.0)

st.title("Customer Churn Prediction")
st.caption(
    "Predict customer risk, explore retention patterns, and review model performance "
    "from customer service and billing data."
)

prediction_tab, dashboard_tab, performance_tab, eda_tab, about_tab = st.tabs(
    ["Predict churn", "Dashboard", "Model performance", "EDA", "About"]
)

with prediction_tab:
    st.header("Customer information")
    st.info(
        "Enter the customer profile below. The model returns a churn probability, "
        "a risk assessment, and a summary of the submitted inputs."
    )
    with st.form("customer_prediction_form"):
        profile_left, profile_right = st.columns(2)
        with profile_left:
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            senior_citizen = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda value: "Yes" if value == 1 else "No",
            )
            partner = st.selectbox("Partner", ["No", "Yes"], index=0)
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=100,
                value=12,
                step=1,
            )
            phone_service = st.selectbox("Phone service", ["Yes", "No"], index=0)
            multiple_lines = st.selectbox(
                "Multiple lines",
                ["No", "Yes", "No phone service"],
                index=0,
            )
            internet_service = st.selectbox(
                "Internet service",
                INTERNET_ORDER,
                index=1,
            )
        with profile_right:
            online_security = st.selectbox(
                "Online security",
                ["No", "Yes", "No internet service"],
                index=0,
            )
            online_backup = st.selectbox(
                "Online backup",
                ["No", "Yes", "No internet service"],
                index=0,
            )
            device_protection = st.selectbox(
                "Device protection",
                ["No", "Yes", "No internet service"],
                index=0,
            )
            tech_support = st.selectbox(
                "Tech support",
                ["No", "Yes", "No internet service"],
                index=0,
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
                index=0,
            )
            streaming_movies = st.selectbox(
                "Streaming movies",
                ["No", "Yes", "No internet service"],
                index=0,
            )
            contract = st.selectbox("Contract", CONTRACT_ORDER, index=0)
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"], index=0)
            payment_method = st.selectbox("Payment method", PAYMENT_ORDER, index=0)
            monthly_charges = st.number_input(
                "Monthly charges",
                min_value=0.0,
                max_value=1000.0,
                value=70.0,
                step=1.0,
            )
            total_charges = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=100000.0,
                value=840.0,
                step=10.0,
            )
        submitted = st.form_submit_button("Predict churn", type="primary", width="stretch")
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    is_valid = True
    if phone_service == "No" and multiple_lines != "No phone service":
        st.error("Multiple lines must be set to No phone service when phone service is No.")
        is_valid = False
    if internet_service == "No":
        service_values = [
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
        ]
        if any(value != "No internet service" for value in service_values):
            st.error("Internet add-ons must be set to No internet service when internet service is No.")
            is_valid = False
    if submitted and is_valid:
        st.session_state["prediction_result"] = predict_churn(customer)
        st.session_state["prediction_customer"] = customer
    result = st.session_state.get("prediction_result")
    submitted_customer = st.session_state.get("prediction_customer")
    if result is not None and submitted_customer is not None:
        render_prediction_result(result, submitted_customer)
    elif metadata is not None:
        st.caption("Submit the form to generate a prediction.")

with dashboard_tab:
    st.subheader("Retention dashboard")
    if dashboard_data is None:
        st.error("Customer data is unavailable.")
    else:
        filter_mask = (
            dashboard_data["Contract"].isin(selected_contracts)
            & dashboard_data["InternetService"].isin(selected_internet)
            & dashboard_data["PaymentMethod"].isin(selected_payments)
            & dashboard_data["tenure"].between(*selected_tenure)
            & dashboard_data["MonthlyCharges"].between(*selected_charges)
        )
        if selected_senior != "All":
            filter_mask &= dashboard_data["SeniorCitizenLabel"].eq(selected_senior)
        filtered = dashboard_data.loc[filter_mask].copy()
        st.caption(
            f"Showing {len(filtered):,} of {len(dashboard_data):,} customers · "
            "Use the sidebar filters to focus the analysis."
        )
        if filtered.empty:
            st.info("No customers match the selected filters.")
        else:
            filtered_rate = filtered["ChurnFlag"].mean()
            overall_rate = dashboard_data["ChurnFlag"].mean()
            kpi_columns = st.columns(5)
            kpi_columns[0].metric("Customers", f"{len(filtered):,}")
            kpi_columns[1].metric(
                "Churn rate",
                percent(filtered_rate),
                f"{(filtered_rate - overall_rate) * 100:+.1f} pp vs. all",
                delta_color="inverse",
            )
            kpi_columns[2].metric("Churned", f"{int(filtered['ChurnFlag'].sum()):,}")
            kpi_columns[3].metric("Monthly revenue", currency(filtered["MonthlyCharges"].sum()))
            kpi_columns[4].metric("Revenue at risk", currency(filtered["RevenueAtRisk"].sum()))
            overview_left, overview_right = st.columns([2.2, 1], gap="large")
            with overview_left:
                st.subheader("Churn by tenure")
                st.plotly_chart(
                    make_tenure_chart(filtered),
                    width="stretch",
                    config={"displayModeBar": False},
                )
            with overview_right:
                st.subheader("Customer status")
                st.plotly_chart(
                    make_donut(filtered),
                    width="stretch",
                    config={"displayModeBar": False},
                )
            contract_left, contract_right = st.columns(2, gap="large")
            with contract_left:
                st.subheader("Churn by contract")
                st.plotly_chart(
                    make_churn_bar(filtered, "Contract"),
                    width="stretch",
                    config={"displayModeBar": False},
                )
            with contract_right:
                st.subheader("Revenue exposure")
                st.plotly_chart(
                    make_revenue_chart(filtered),
                    width="stretch",
                    config={"displayModeBar": False},
                )

with performance_tab:
    render_model_performance(metadata)

with eda_tab:
    render_eda(metadata)

with about_tab:
    st.subheader("About this project")
    st.markdown(
        "This application follows a complete customer churn workflow: cleaning, feature "
        "engineering, model comparison, probability prediction, and an interactive review surface."
    )
    pipeline_columns = st.columns(3)
    pipeline_columns[0].metric("Stage 1", "Data cleaning", "TotalCharges and missing values")
    pipeline_columns[1].metric("Stage 2", "Classification", "Logistic regression and random forest")
    pipeline_columns[2].metric("Stage 3", "Deployment", "Streamlit prediction interface")
    st.markdown(
        """
        ### Run locally
        1. Install dependencies with `pip install -r requirements.txt`.
        2. Train or refresh the artifacts with `python train_model.py`.
        3. Launch the application with `streamlit run app.py`.
        """
    )
    st.caption(
        "The prediction is an estimate based on historical records and should not be treated "
        "as a guarantee of customer behavior."
    )
