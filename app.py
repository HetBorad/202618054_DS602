import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical Insurance Statistical Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

insurance_df = pd.read_csv("insurance.csv")

# ============================================================
# TITLE
# ============================================================

st.title("🏥 Medical Insurance Statistical Dashboard")

st.write(
    "Interactive exploration and statistical analysis of medical "
    "insurance charges."
)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Data Filters")

# Age filter
age_min = int(insurance_df["age"].min())
age_max = int(insurance_df["age"].max())

age_range = st.sidebar.slider(
    "Age Range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

# Charges filter
charges_min = float(insurance_df["charges"].min())
charges_max = float(insurance_df["charges"].max())

charges_range = st.sidebar.slider(
    "Charges Range",
    min_value=charges_min,
    max_value=charges_max,
    value=(charges_min, charges_max)
)

# Sex filter
sex_options = insurance_df["sex"].unique().tolist()

selected_sex = st.sidebar.multiselect(
    "Sex",
    options=sex_options,
    default=sex_options
)

# Smoker filter
smoker_options = insurance_df["smoker"].unique().tolist()

selected_smoker = st.sidebar.multiselect(
    "Smoking Status",
    options=smoker_options,
    default=smoker_options
)

# Region filter
region_options = insurance_df["region"].unique().tolist()

selected_region = st.sidebar.multiselect(
    "Region",
    options=region_options,
    default=region_options
)

# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = insurance_df[
    (insurance_df["age"].between(age_range[0], age_range[1])) &
    (insurance_df["charges"].between(charges_range[0], charges_range[1])) &
    (insurance_df["sex"].isin(selected_sex)) &
    (insurance_df["smoker"].isin(selected_smoker)) &
    (insurance_df["region"].isin(selected_region))
]

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Data Exploration",
    "🧪 Hypothesis Testing Lab",
    "🔮 Live Prediction & Diagnostics"
])

# ============================================================
# TAB 1 — DATA EXPLORATION
# ============================================================

with tab1:

    st.header("📊 Data Exploration")

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(filtered_df):,}"
    )

    col2.metric(
        "Average Charges",
        f"${filtered_df['charges'].mean():,.2f}"
        if len(filtered_df) > 0 else "$0.00"
    )

    col3.metric(
        "Median Charges",
        f"${filtered_df['charges'].median():,.2f}"
        if len(filtered_df) > 0 else "$0.00"
    )

    col4.metric(
        "Average BMI",
        f"{filtered_df['bmi'].mean():.2f}"
        if len(filtered_df) > 0 else "0.00"
    )

    st.divider()

    # --------------------------------------------------------
    # FILTERED DATA
    # --------------------------------------------------------

    st.subheader("Filtered Dataset")

    st.write(
        f"Showing **{len(filtered_df):,}** records "
        f"out of **{len(insurance_df):,}** total records."
    )

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # DISTRIBUTION PLOT
    # --------------------------------------------------------

    st.subheader("Medical Charges Distribution")

    if len(filtered_df) > 0:

        fig_hist = px.histogram(
            filtered_df,
            x="charges",
            nbins=40,
            marginal="box",
            title="Distribution of Medical Insurance Charges"
        )

        fig_hist.update_layout(
            xaxis_title="Medical Charges",
            yaxis_title="Frequency"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    # --------------------------------------------------------
    # SCATTER PLOT
    # --------------------------------------------------------

    st.subheader("Age vs Medical Charges")

    if len(filtered_df) > 0:

        fig_scatter = px.scatter(
            filtered_df,
            x="age",
            y="charges",
            color="smoker",
            hover_data=[
                "bmi",
                "children",
                "region"
            ],
            title="Age vs Medical Charges"
        )

        fig_scatter.update_layout(
            xaxis_title="Age",
            yaxis_title="Medical Charges"
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )

    # --------------------------------------------------------
    # CORRELATION MATRIX
    # --------------------------------------------------------

    st.subheader("Correlation Matrix")

    if len(filtered_df) > 1:

        numerical_cols = [
            "age",
            "bmi",
            "children",
            "charges"
        ]

        correlation_matrix = filtered_df[
            numerical_cols
        ].corr()

        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True
        )

    # --------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # --------------------------------------------------------

    st.subheader("Descriptive Statistics")

    if len(filtered_df) > 0:

        numerical_cols = [
            "age",
            "bmi",
            "children",
            "charges"
        ]

        descriptive_stats = pd.DataFrame({
            "Mean": filtered_df[numerical_cols].mean(),
            "Median": filtered_df[numerical_cols].median(),
            "Standard Deviation": filtered_df[numerical_cols].std(),
            "IQR": (
                filtered_df[numerical_cols].quantile(0.75)
                -
                filtered_df[numerical_cols].quantile(0.25)
            ),
            "Skewness": filtered_df[numerical_cols].skew(),
            "Kurtosis": filtered_df[numerical_cols].kurtosis()
        })

        st.dataframe(
            descriptive_stats.round(4),
            use_container_width=True
        )

    else:

        st.warning(
            "No records match the selected filters."
        )
# ============================================================
# TAB 2: HYPOTHESIS TESTING LAB
# ============================================================

with tab2:

    st.header("🧪 Hypothesis Testing Lab")

    st.write(
        "Use this section to compare medical charges between different "
        "groups using statistical hypothesis tests."
    )

    st.divider()

    # --------------------------------------------------------
    # Select variables
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        categorical_variable = st.selectbox(
            "Select Categorical Variable",
            ["smoker", "sex", "region"]
        )

    with col2:
        numerical_variable = st.selectbox(
            "Select Numerical Variable",
            ["charges", "age", "bmi"]
        )

    st.divider()

    # --------------------------------------------------------
    # Group the data
    # --------------------------------------------------------

    groups = insurance_df[categorical_variable].dropna().unique()

    st.subheader("Selected Variables")

    st.write(
        f"**Categorical Variable:** `{categorical_variable}`"
    )

    st.write(
        f"**Numerical Variable:** `{numerical_variable}`"
    )

    st.write(
        f"**Number of Groups:** {len(groups)}"
    )

    # --------------------------------------------------------
    # Two-group hypothesis test
    # --------------------------------------------------------

    if len(groups) == 2:

        st.subheader("Two-Group Hypothesis Test")

        group1 = insurance_df[
            insurance_df[categorical_variable] == groups[0]
        ][numerical_variable].dropna()

        group2 = insurance_df[
            insurance_df[categorical_variable] == groups[1]
        ][numerical_variable].dropna()

        # Group summary
        summary = pd.DataFrame({
            "Group": [groups[0], groups[1]],
            "Count": [len(group1), len(group2)],
            "Mean": [group1.mean(), group2.mean()],
            "Median": [group1.median(), group2.median()],
            "Standard Deviation": [group1.std(), group2.std()]
        })

        st.dataframe(
            summary.round(2),
            use_container_width=True
        )

        # ----------------------------------------------------
        # Shapiro-Wilk Normality Test
        # ----------------------------------------------------

        st.subheader("1️⃣ Shapiro-Wilk Normality Test")

        shapiro1 = stats.shapiro(group1)
        shapiro2 = stats.shapiro(group2)

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**{groups[0]}**")
            st.write(f"Statistic: {shapiro1.statistic:.4f}")
            st.write(f"p-value: {shapiro1.pvalue:.6f}")

        with col2:
            st.write(f"**{groups[1]}**")
            st.write(f"Statistic: {shapiro2.statistic:.4f}")
            st.write(f"p-value: {shapiro2.pvalue:.6f}")

        alpha = 0.05

        if shapiro1.pvalue > alpha and shapiro2.pvalue > alpha:
            normal = True
            st.success(
                "Both groups appear normally distributed (p > 0.05)."
            )
        else:
            normal = False
            st.warning(
                "At least one group is not normally distributed (p ≤ 0.05)."
            )

        # ----------------------------------------------------
        # Levene's Test
        # ----------------------------------------------------

        st.subheader("2️⃣ Levene's Test for Equal Variance")

        levene_test = stats.levene(group1, group2)

        st.write(
            f"**Test Statistic:** {levene_test.statistic:.4f}"
        )

        st.write(
            f"**p-value:** {levene_test.pvalue:.6f}"
        )

        if levene_test.pvalue > alpha:
            equal_variance = True
            st.success(
                "The variances can be considered equal (p > 0.05)."
            )
        else:
            equal_variance = False
            st.warning(
                "The variances are significantly different (p ≤ 0.05)."
            )

        # ----------------------------------------------------
        # Select appropriate test
        # ----------------------------------------------------

        st.subheader("3️⃣ Final Hypothesis Test")

        if normal:

            # Welch t-test is safer when variances are unequal
            test_result = stats.ttest_ind(
                group1,
                group2,
                equal_var=equal_variance
            )

            test_name = (
                "Independent Two-Sample t-test"
            )

        else:

            test_result = stats.mannwhitneyu(
                group1,
                group2,
                alternative="two-sided"
            )

            test_name = "Mann-Whitney U Test"

        st.write(f"**Test Used:** {test_name}")

        st.write(
            f"**Test Statistic:** {test_result.statistic:.4f}"
        )

        st.write(
            f"**p-value:** {test_result.pvalue:.6g}"
        )

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        if test_result.pvalue < alpha:

            st.error(
                "❌ Reject H₀ — There is a statistically significant "
                "difference between the groups."
            )

            st.write(
                f"At α = 0.05, the p-value is less than 0.05. "
                f"Therefore, the {numerical_variable} differs "
                f"significantly between the two {categorical_variable} groups."
            )

        else:

            st.success(
                "✅ Fail to Reject H₀ — There is not enough evidence "
                "of a significant difference between the groups."
            )

            st.write(
                f"At α = 0.05, the p-value is greater than or equal "
                f"to 0.05. Therefore, there is not enough evidence "
                f"to conclude that {numerical_variable} differs "
                f"between the two groups."
            )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        st.subheader("📊 Group Comparison")

        fig = px.box(
            insurance_df,
            x=categorical_variable,
            y=numerical_variable,
            color=categorical_variable,
            points="outliers",
            title=f"{numerical_variable.title()} by {categorical_variable.title()}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # More than two groups → One-Way ANOVA
    # --------------------------------------------------------

    else:

        st.subheader("One-Way ANOVA Test")

        grouped_data = [
            insurance_df[
                insurance_df[categorical_variable] == group
            ][numerical_variable].dropna()
            for group in groups
        ]

        # Group summary
        summary = insurance_df.groupby(
            categorical_variable
        )[numerical_variable].agg(
            ["count", "mean", "median", "std"]
        )

        summary.columns = [
            "Count",
            "Mean",
            "Median",
            "Standard Deviation"
        ]

        st.dataframe(
            summary.round(2),
            use_container_width=True
        )

        # ----------------------------------------------------
        # One-Way ANOVA
        # ----------------------------------------------------

        anova_result = stats.f_oneway(*grouped_data)

        st.write(
            f"**F-statistic:** {anova_result.statistic:.4f}"
        )

        st.write(
            f"**p-value:** {anova_result.pvalue:.6g}"
        )

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        if anova_result.pvalue < alpha:

            st.error(
                "❌ Reject H₀ — At least one group has a "
                "significantly different mean."
            )

            st.write(
                f"The mean {numerical_variable} is significantly "
                f"different across the {len(groups)} "
                f"{categorical_variable} groups."
            )

        else:

            st.success(
                "✅ Fail to Reject H₀ — There is not enough evidence "
                "that the group means are different."
            )

            st.write(
                f"There is not enough evidence to conclude that "
                f"mean {numerical_variable} differs across the "
                f"{categorical_variable} groups."
            )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        st.subheader("📊 Group Comparison")

        fig = px.box(
            insurance_df,
            x=categorical_variable,
            y=numerical_variable,
            color=categorical_variable,
            points="outliers",
            title=f"{numerical_variable.title()} by {categorical_variable.title()}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
# ============================================================
# TAB 3: LIVE PREDICTION & DIAGNOSTICS
# ============================================================

with tab3:

    st.header("🔮 Live Prediction & Diagnostics")

    st.write(
        "Enter patient information to predict estimated medical "
        "insurance charges using the trained OLS regression model."
    )

    st.divider()

    # --------------------------------------------------------
    # Load saved model
    # --------------------------------------------------------

    import joblib

    model_data = joblib.load("models/ols_model.pkl")

    loaded_model = model_data["model"]
    model_features = model_data["features"]

    # --------------------------------------------------------
    # User inputs
    # --------------------------------------------------------

    st.subheader("👤 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=28.5,
            step=0.1
        )

    with col2:
        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=2
        )

        sex = st.selectbox(
            "Sex",
            ["female", "male"]
        )

    with col3:
        smoker = st.selectbox(
            "Smoking Status",
            ["no", "yes"]
        )

        region = st.selectbox(
            "Region",
            [
                "northeast",
                "northwest",
                "southeast",
                "southwest"
            ]
        )

    st.divider()

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "const": [1],
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0]
    })

    # Make sure columns are in exactly the same order
    input_data = input_data[model_features]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button("🔮 Predict Medical Charges"):

        prediction = loaded_model.get_prediction(input_data)

        prediction_summary = prediction.summary_frame(alpha=0.05)

        predicted_charge = prediction_summary["mean"].iloc[0]

        lower_prediction = prediction_summary["obs_ci_lower"].iloc[0]

        upper_prediction = prediction_summary["obs_ci_upper"].iloc[0]

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.subheader("💰 Prediction Result")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Charges",
            f"${predicted_charge:,.2f}"
        )

        col2.metric(
            "Lower 95% Prediction",
            f"${lower_prediction:,.2f}"
        )

        col3.metric(
            "Upper 95% Prediction",
            f"${upper_prediction:,.2f}"
        )

        st.success(
            f"Estimated medical charges: **${predicted_charge:,.2f}**"
        )

        st.info(
            f"95% prediction interval: "
            f"**${lower_prediction:,.2f} – ${upper_prediction:,.2f}**"
        )

    st.divider()

    # ========================================================
    # MODEL DIAGNOSTICS
    # ========================================================

    st.subheader("📈 Model Diagnostics")

    residuals = loaded_model.resid
    fitted_values = loaded_model.fittedvalues

    # --------------------------------------------------------
    # Residuals vs Fitted
    # --------------------------------------------------------

    st.write("### Residuals vs Fitted Values")

    diagnostic_df = pd.DataFrame({
        "Fitted Values": fitted_values,
        "Residuals": residuals
    })

    fig_residual = px.scatter(
        diagnostic_df,
        x="Fitted Values",
        y="Residuals",
        title="Residuals vs Fitted Values"
    )

    fig_residual.add_hline(
        y=0,
        line_dash="dash"
    )

    st.plotly_chart(
        fig_residual,
        use_container_width=True
    )

    st.write(
        "Residuals should ideally be randomly scattered around zero "
        "with approximately constant spread."
    )

    # --------------------------------------------------------
    # Q-Q Plot
    # --------------------------------------------------------

    st.write("### Q-Q Plot")

    import statsmodels.api as sm

    fig, ax = plt.subplots(figsize=(4, 4))
    
    qq_data = sm.qqplot(
        residuals,
        line="45",
        fit=True,
        ax=ax
    )
    ax.set_title("Q-Q Plot of Residuals")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.pyplot(fig, use_container_width=False)

    st.write(
        "The Q-Q plot is used to check whether the residuals "
        "approximately follow a normal distribution."
    )

    # --------------------------------------------------------
    # Diagnostic statistics
    # --------------------------------------------------------

    st.write("### Diagnostic Statistics")

    from statsmodels.stats.stattools import (
        omni_normtest,
        jarque_bera,
        durbin_watson
    )

    omnibus_stat, omnibus_p = omni_normtest(residuals)

    jb_stat, jb_p, skewness, kurtosis = jarque_bera(residuals)

    dw_stat = durbin_watson(residuals)

    diagnostic_stats = pd.DataFrame({
        "Statistic": [
            "Omnibus",
            "Omnibus p-value",
            "Jarque-Bera",
            "Jarque-Bera p-value",
            "Skewness",
            "Kurtosis",
            "Durbin-Watson"
        ],
        "Value": [
            omnibus_stat,
            omnibus_p,
            jb_stat,
            jb_p,
            skewness,
            kurtosis,
            dw_stat
        ]
    })

    st.dataframe(
        diagnostic_stats.round(4),
        use_container_width=True
    )