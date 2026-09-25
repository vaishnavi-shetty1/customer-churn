from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Customer Churn Command Center", layout="wide")

DATA_PATH = Path(__file__).resolve().parent / "data" / "customer_churn.csv"
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
CATEGORY_ORDERS = {
    "Contract": CONTRACT_ORDER,
    "InternetService": INTERNET_ORDER,
    "PaymentMethod": PAYMENT_ORDER,
    "TenureBand": TENURE_ORDER,
    "SeniorCitizenLabel": ["Non-senior", "Senior"],
}
FILTER_KEYS = [
    "contract_filter",
    "internet_filter",
    "payment_filter",
    "senior_filter",
    "tenure_filter",
    "charges_filter",
]
DERIVED_COLUMNS = [
    "ChurnFlag",
    "SeniorCitizenLabel",
    "TenureBand",
    "RevenueAtRisk",
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
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["ChurnFlag"] = data["Churn"].eq("Yes").astype(int)
    data["SeniorCitizenLabel"] = data["SeniorCitizen"].map(
        {0: "Non-senior", 1: "Senior"}
    )
    data["TenureBand"] = pd.cut(
        data["tenure"],
        bins=[-1, 6, 12, 24, 48, 73],
        labels=TENURE_ORDER,
        ordered=True,
    )
    data["RevenueAtRisk"] = data["MonthlyCharges"] * data["ChurnFlag"]
    return data


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def segment_summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        data.groupby(column, observed=True, dropna=False)
        .agg(
            Customers=("ChurnFlag", "size"),
            Churned=("ChurnFlag", "sum"),
            AvgMonthlyCharge=("MonthlyCharges", "mean"),
            MonthlyRevenue=("MonthlyCharges", "sum"),
            RevenueAtRisk=("RevenueAtRisk", "sum"),
        )
        .reset_index()
    )
    summary["ChurnRate"] = (
        summary["Churned"] / summary["Customers"].replace(0, pd.NA)
    ).fillna(0)
    return summary


def ordered_categories(data: pd.DataFrame, column: str) -> list[str]:
    if column in CATEGORY_ORDERS:
        return CATEGORY_ORDERS[column]
    return sorted(data[column].dropna().astype(str).unique().tolist())


def churn_rate_bar(data: pd.DataFrame, column: str) -> go.Figure:
    summary = segment_summary(data, column)
    summary = summary.loc[summary["Customers"] > 0].sort_values("ChurnRate")
    figure = px.bar(
        summary,
        x="ChurnRate",
        y=column,
        orientation="h",
        color="ChurnRate",
        color_continuous_scale=["#14b8a6", "#f59e0b", "#ef4444"],
        custom_data=["Customers", "Churned", "AvgMonthlyCharge"],
        labels={column: "", "ChurnRate": "Churn rate"},
    )
    figure.update_traces(
        texttemplate="%{x:.1%}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>Churn rate: %{x:.1%}<br>"
            "Customers: %{customdata[0]:,.0f}<br>"
            "Churned: %{customdata[1]:,.0f}<br>"
            "Avg. monthly charge: $%{customdata[2]:,.2f}<extra></extra>"
        ),
    )
    figure.update_layout(
        height=max(360, 72 * len(summary) + 120),
        margin=dict(l=10, r=45, t=15, b=15),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(
        title="Churn rate",
        tickformat=".0%",
        range=[0, max(0.1, float(summary["ChurnRate"].max()) * 1.18)],
        gridcolor="#e2e8f0",
    )
    figure.update_yaxes(showgrid=False, categoryorder="array", categoryarray=summary[column].tolist())
    return figure


def segment_scatter(data: pd.DataFrame, column: str) -> go.Figure:
    summary = segment_summary(data, column)
    summary = summary.loc[summary["Customers"] > 0]
    figure = px.scatter(
        summary,
        x="AvgMonthlyCharge",
        y="ChurnRate",
        size="Customers",
        color="ChurnRate",
        text=column,
        color_continuous_scale=["#14b8a6", "#f59e0b", "#ef4444"],
        size_max=48,
        custom_data=[column, "Customers", "Churned"],
        labels={
            "AvgMonthlyCharge": "Average monthly charge",
            "ChurnRate": "Churn rate",
            "Customers": "Customer count",
        },
    )
    figure.update_traces(
        textposition="top center",
        marker_line_color="#ffffff",
        marker_line_width=1.5,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Churn rate: %{y:.1%}<br>"
            "Avg. monthly charge: $%{x:,.2f}<br>"
            "Customers: %{customdata[1]:,.0f}<br>"
            "Churned: %{customdata[2]:,.0f}<extra></extra>"
        ),
    )
    figure.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=25, b=20),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(title="Average monthly charge", tickprefix="$", gridcolor="#e2e8f0")
    figure.update_yaxes(title="Churn rate", tickformat=".0%", range=[0, 1], gridcolor="#e2e8f0")
    return figure


def churn_heatmap(
    data: pd.DataFrame, row_column: str, column_name: str
) -> go.Figure:
    rows = ordered_categories(data, row_column)
    columns = ordered_categories(data, column_name)
    rates = data.pivot_table(
        index=row_column,
        columns=column_name,
        values="ChurnFlag",
        aggfunc="mean",
        observed=False,
    ).reindex(index=rows, columns=columns, fill_value=0)
    labels = [[f"{value:.1%}" for value in row] for row in rates.to_numpy()]
    figure = go.Figure(
        go.Heatmap(
            z=rates.to_numpy(),
            x=columns,
            y=rows,
            text=labels,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#ccfbf1"],
                [0.25, "#5eead4"],
                [0.55, "#fbbf24"],
                [1.0, "#ef4444"],
            ],
            colorbar=dict(title="Churn", tickformat=".0%"),
            hovertemplate=(
                f"{row_column}: %{{y}}<br>{column_name}: %{{x}}<br>"
                "Churn rate: %{z:.1%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=max(330, 80 * len(rows) + 170),
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(side="bottom", tickangle=-20, showgrid=False)
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return figure


def tenure_charts(data: pd.DataFrame) -> go.Figure:
    summary = segment_summary(data, "TenureBand")
    summary = summary.set_index("TenureBand").reindex(TENURE_ORDER, fill_value=0).reset_index()
    figure = go.Figure()
    figure.add_bar(
        x=summary["TenureBand"],
        y=summary["Customers"],
        name="Customers",
        marker_color="#0f766e",
        customdata=summary[["RevenueAtRisk"]],
        hovertemplate=(
            "<b>%{x}</b><br>Customers: %{y:,.0f}<br>"
            "Revenue at risk: $%{customdata[0]:,.0f}<extra></extra>"
        ),
    )
    figure.add_scatter(
        x=summary["TenureBand"],
        y=summary["ChurnRate"],
        name="Churn rate",
        mode="lines+markers",
        marker=dict(color="#e11d48", size=9),
        line=dict(color="#e11d48", width=3),
        hovertemplate="<b>%{x}</b><br>Churn rate: %{y:.1%}<extra></extra>",
        yaxis="y2",
    )
    figure.update_layout(
        height=430,
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
    figure.update_xaxes(title="", showgrid=False)
    return figure


def reset_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


try:
    data = load_data(DATA_PATH)
except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Unable to load the customer dataset: {error}")
    st.stop()

st.sidebar.title("Filters")
st.sidebar.caption("Filter the entire dashboard")
st.sidebar.button(
    "Reset filters",
    on_click=reset_filters,
    width="stretch",
)

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
    min_value=int(data["tenure"].min()),
    max_value=int(data["tenure"].max()),
    value=(int(data["tenure"].min()), int(data["tenure"].max())),
    key="tenure_filter",
)
selected_charges = st.sidebar.slider(
    "Monthly charge",
    min_value=float(data["MonthlyCharges"].min()),
    max_value=float(data["MonthlyCharges"].max()),
    value=(float(data["MonthlyCharges"].min()), float(data["MonthlyCharges"].max())),
    step=1.0,
    key="charges_filter",
)
st.sidebar.divider()
st.sidebar.success(f"{len(data):,} customer records loaded")
st.sidebar.caption("Blank TotalCharges values are cleaned during loading.")

mask = (
    data["Contract"].isin(selected_contracts)
    & data["InternetService"].isin(selected_internet)
    & data["PaymentMethod"].isin(selected_payments)
    & data["tenure"].between(*selected_tenure)
    & data["MonthlyCharges"].between(*selected_charges)
)
if selected_senior != "All":
    mask &= data["SeniorCitizenLabel"].eq(selected_senior)

filtered = data.loc[mask].copy()
if filtered.empty:
    st.warning("No customers match the selected filters. Broaden a filter to continue.")
    st.stop()

overall_churn_rate = float(data["ChurnFlag"].mean())
churn_rate = float(filtered["ChurnFlag"].mean())
churned_customers = int(filtered["ChurnFlag"].sum())
monthly_revenue = float(filtered["MonthlyCharges"].sum())
revenue_at_risk = float(filtered["RevenueAtRisk"].sum())
rate_delta = (churn_rate - overall_churn_rate) * 100

st.title("Customer Churn Command Center")
st.caption(
    f"Explore {len(filtered):,} of {len(data):,} customers · "
    f"{len(filtered) / len(data):.1%} of the dataset · "
    "Observed relationships are descriptive, not causal."
)

kpi_columns = st.columns(5)
kpi_columns[0].metric(
    "Customers",
    f"{len(filtered):,}",
    f"{len(filtered) / len(data):.1%} of dataset",
    delta_color="off",
)
kpi_columns[1].metric(
    "Churn rate",
    format_percent(churn_rate),
    f"{rate_delta:+.1f} pp vs. all customers",
    delta_color="inverse",
)
kpi_columns[2].metric("Churned customers", f"{churned_customers:,}")
kpi_columns[3].metric("Monthly revenue", format_currency(monthly_revenue))
kpi_columns[4].metric("Revenue at risk", format_currency(revenue_at_risk))
st.caption(
    "Revenue at risk is the monthly charge associated with customers recorded as churned."
)

overview_tab, drivers_tab, customers_tab, quality_tab = st.tabs(
    ["Overview", "Segment drivers", "Customer explorer", "Data quality"]
)

with overview_tab:
    overview_left, overview_right = st.columns([2.2, 1], gap="large")
    with overview_left:
        st.subheader("Churn by tenure")
        st.caption("Customer volume and observed churn rate by tenure band")
        st.plotly_chart(
            tenure_charts(filtered),
            width="stretch",
            config={"displayModeBar": False},
        )
    with overview_right:
        st.subheader("Customer status")
        status_counts = filtered["Churn"].value_counts()
        status_figure = go.Figure(
            go.Pie(
                labels=["Retained", "Churned"],
                values=[
                    int(status_counts.get("No", 0)),
                    int(status_counts.get("Yes", 0)),
                ],
                hole=0.68,
                marker=dict(
                    colors=["#0f766e", "#e11d48"],
                    line=dict(color="#ffffff", width=3),
                ),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>",
            )
        )
        status_figure.add_annotation(
            text=f"<b>{churn_rate:.1%}</b><br>churn rate",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="#0f172a"),
        )
        status_figure.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=15, b=55),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.02),
        )
        st.plotly_chart(
            status_figure,
            width="stretch",
            config={"displayModeBar": False},
        )

    contract_summary = segment_summary(filtered, "Contract")
    contract_summary = contract_summary.set_index("Contract").reindex(CONTRACT_ORDER).dropna()
    contract_figure = px.bar(
        contract_summary.reset_index(),
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
            "value": "Revenue",
        },
    )
    contract_figure.update_traces(
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: $%{y:,.0f}<extra></extra>"
    )
    contract_figure.update_layout(
        height=360,
        barmode="group",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    contract_figure.update_yaxes(title="Monthly revenue", tickprefix="$", gridcolor="#e2e8f0")
    contract_figure.update_xaxes(showgrid=False)

    st.subheader("Revenue exposure by contract")
    st.caption("Total monthly revenue compared with monthly revenue at risk")
    st.plotly_chart(
        contract_figure,
        width="stretch",
        config={"displayModeBar": False},
    )

with drivers_tab:
    st.subheader("Compare customer segments")
    st.caption("Use a single segment dimension to identify high-churn customer groups.")

    driver_options = {
        "Contract": "Contract",
        "Internet service": "InternetService",
        "Payment method": "PaymentMethod",
        "Tech support": "TechSupport",
        "Online security": "OnlineSecurity",
        "Paperless billing": "PaperlessBilling",
        "Dependents": "Dependents",
        "Partner": "Partner",
    }
    driver_left, driver_right = st.columns(2, gap="large")
    with driver_left:
        selected_driver = st.selectbox("Segment dimension", list(driver_options))
        dimension = driver_options[selected_driver]
        st.plotly_chart(
            churn_rate_bar(filtered, dimension),
            width="stretch",
            config={"displayModeBar": False},
        )
    with driver_right:
        st.caption("Bubble size represents customer count")
        st.plotly_chart(
            segment_scatter(filtered, dimension),
            width="stretch",
            config={"displayModeBar": False},
        )

    st.divider()
    st.subheader("Cross-segment churn heatmap")
    heatmap_options = {
        "Contract": "Contract",
        "Internet service": "InternetService",
        "Payment method": "PaymentMethod",
        "Tenure band": "TenureBand",
        "Senior status": "SeniorCitizenLabel",
    }
    heat_left, heat_right = st.columns(2)
    with heat_left:
        row_label = st.selectbox(
            "Rows",
            list(heatmap_options),
            index=0,
            key="heatmap_rows",
        )
    with heat_right:
        column_label = st.selectbox(
            "Columns",
            list(heatmap_options),
            index=1,
            key="heatmap_columns",
        )
    row_column = heatmap_options[row_label]
    column_name = heatmap_options[column_label]
    if row_column == column_name:
        st.info("Choose two different dimensions for the heatmap.")
    else:
        st.plotly_chart(
            churn_heatmap(filtered, row_column, column_name),
            width="stretch",
            config={"displayModeBar": False},
        )

with customers_tab:
    st.subheader("Customer records")
    st.caption("Search and inspect the customers included in the current filters.")

    customer_columns = [
        "customerID",
        "gender",
        "SeniorCitizenLabel",
        "Partner",
        "Dependents",
        "tenure",
        "TenureBand",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]
    customer_view = filtered[customer_columns].rename(
        columns={
            "customerID": "Customer ID",
            "tenure": "Tenure (months)",
            "SeniorCitizenLabel": "Senior citizen",
            "TenureBand": "Tenure band",
            "InternetService": "Internet service",
            "PaymentMethod": "Payment method",
            "MonthlyCharges": "Monthly charge",
            "TotalCharges": "Total charges",
            "Churn": "Status",
        }
    )

    search_left, status_left, download_left = st.columns([2, 1, 1])
    with search_left:
        search_term = st.text_input("Search customer ID", placeholder="e.g. 7590-VHVEG")
    with status_left:
        status_filter = st.selectbox("Status", ["All", "Churned", "Retained"])
    with download_left:
        st.write("")
        st.download_button(
            "Download CSV",
            data=customer_view.to_csv(index=False).encode("utf-8"),
            file_name="filtered_customers.csv",
            mime="text/csv",
            width="stretch",
        )

    display_customers = customer_view
    if search_term:
        display_customers = display_customers.loc[
            display_customers["Customer ID"].str.contains(search_term, case=False, regex=False)
        ]
    if status_filter == "Churned":
        display_customers = display_customers.loc[display_customers["Status"].eq("Yes")]
    elif status_filter == "Retained":
        display_customers = display_customers.loc[display_customers["Status"].eq("No")]

    st.caption(f"Showing {len(display_customers):,} customer records")
    st.dataframe(
        display_customers,
        hide_index=True,
        width="stretch",
        height=560,
        column_config={
            "Tenure (months)": st.column_config.NumberColumn(format="%d"),
            "Monthly charge": st.column_config.NumberColumn(format="$%.2f"),
            "Total charges": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

with quality_tab:
    st.subheader("Dataset health")
    quality_columns = st.columns(4)
    quality_columns[0].metric("Rows", f"{len(data):,}")
    quality_columns[1].metric("Columns", f"{data.shape[1] - len(DERIVED_COLUMNS):,}")
    quality_columns[2].metric("Missing TotalCharges", f"{int(data['TotalCharges'].isna().sum()):,}")
    quality_columns[3].metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")

    st.warning(
        f"{int(data['TotalCharges'].isna().sum())} blank TotalCharges values were converted to "
        "missing values during loading. They correspond to customers with zero tenure."
    )

    profile = pd.DataFrame(
        {
            "Column": data.columns,
            "Data type": data.dtypes.astype(str).to_numpy(),
            "Missing": data.isna().sum().to_numpy(),
            "Unique values": data.nunique(dropna=True).to_numpy(),
        }
    )
    st.subheader("Column profile")
    st.dataframe(
        profile,
        hide_index=True,
        width="stretch",
        height=460,
    )
    st.download_button(
        "Download column profile",
        data=profile.to_csv(index=False).encode("utf-8"),
        file_name="column_profile.csv",
        mime="text/csv",
    )
