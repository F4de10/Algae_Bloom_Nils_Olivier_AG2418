"""
Nils Olivier
2025-03-21
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from src import feat, mod

# Set the style for all plots
plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12


def define_feature_sets():
    """
    Defines and returns sets of features used for algal bloom analysis models.

    Returns:
        dict: A dictionary where keys are feature set names and values are lists of feature names.
        The dictionary contains the following feature sets:
        - sat_500: Features derived from satellite data with 500m buffer
        - sat_1000: Features derived from satellite data with 1000m buffer
        - sat_2500: Features derived from satellite data with 2500m buffer
        - geographic: Location-based features (latitude, longitude, region, cluster)
        - topographic: Elevation-related features
        - temporal: Time-based feature (date)
        - sat_1025: Combined features from sat_1000 and sat_2500
        - all_satellite: Combined features from all satellite data (500m, 1000m, 2500m)
    """

    # Define feature sets
    feature_sets = {
        "sat_500": ["prop_lake_500", "r_500", "g_500", "b_500"],
        "sat_1000": ["prop_lake_1000", "r_1000", "g_1000", "b_1000"],
        "sat_2500": ["prop_lake_2500", "r_2500", "g_2500", "b_2500"],
        "geographic": ["latitude", "longitude", "region", "cluster"],
        "topographic": ["elevation", "maxe", "dife", "avge", "stde"],
        "temporal": ["date"],
    }

    # Derived combinations
    feature_sets["sat_1025"] = feature_sets["sat_1000"] + feature_sets["sat_2500"]
    feature_sets["all_satellite"] = (
        feature_sets["sat_500"] + feature_sets["sat_1000"] + feature_sets["sat_2500"]
    )

    return feature_sets


def create_model(train_data=None):
    """
    Create and train an ensemble model for algae bloom severity prediction.
    This function creates and trains an ensemble model consisting of three different
    regression models: CatBoost, LightGBM, and XGBoost. Each model is configured with
    different feature sets and hyperparameters to predict algae bloom severity.
    Parameters
    ----------
    train_data : pandas.DataFrame, optional
        The training data to fit the models. If None, the function will get the data
        using feat.get_data() with split_pred=True.
    Returns
    -------
    mod.EnsMod
        An ensemble model containing the trained CatBoost, LightGBM, and XGBoost models.
    Notes
    -----
    The function also saves the ensemble model to disk with a filename that includes
    the current date.
    Each regression model uses different feature sets:
    - CatBoost: Uses region, cluster, date, latitude, longitude, maxe, and dife.
    - LightGBM: Uses region, cluster, imtype, date, latitude, longitude, elevation,
      dife, and satellite features.
    - XGBoost: Uses region, cluster, and date.
    """

    if train_data is None:
        train_data = feat.get_data(split_pred=True)

    today = feat.today_str()

    feature_sets = define_feature_sets()
    sat_1025 = feature_sets["sat_1025"]

    cat = mod.RegMod(
        ord_vars=["region", "cluster"],
        dat_vars=["date"],
        ide_vars=["latitude", "longitude", "maxe", "dife"],
        y="severity",
        mod=mod.CatBoostRegressor(
            iterations=380, depth=6, allow_writing_files=False, verbose=False
        ),
    )
    cat.fit(train_data, weight=False, cat=False)

    lgbm = mod.RegMod(
        ord_vars=["region", "cluster", "imtype"],
        dat_vars=["date"],
        ide_vars=["latitude", "longitude", "elevation", "dife"] + sat_1025,
        y="severity",
        mod=mod.LGBMRegressor(n_estimators=470, max_depth=8),
    )
    lgbm.fit(train_data, weight=False, cat=True)

    xgb = mod.RegMod(
        ord_vars=["region", "cluster"],
        dat_vars=["date"],
        y="severity",
        mod=mod.XGBRegressor(n_estimators=70, max_depth=2),
    )
    xgb.fit(train_data, weight=False, cat=False)

    ensemble = mod.EnsMod(mods={"xgb": xgb, "cat": cat, "lig": lgbm})

    # Saving the data and model with today's date
    mod.save_model(ensemble, f"mod_{today}")

    return ensemble


def load_model_and_data():
    """
    Load the trained model and associated datasets.

    This function attempts to load a pre-trained model and relevant datasets for prediction.
    It first tries to load a model for the current date, and if that fails, it falls back to
    a predefined best-performing model.

    Returns:
        tuple: A tuple containing:
            - model (object): The loaded machine learning model, or None if no model is found
            - prediction_data (DataFrame): Data for making predictions, preprocessed with split_pred=True
            - test_data (DataFrame): Test dataset for evaluation

    Raises:
        No exceptions are raised, but prints error messages if models cannot be found.

    Note:
        If no model is found, the function will print an error message suggesting to run
        main_preds.py to generate a model first.
    """

    print("Loading model and data...")

    today = feat.today_str()
    model_path = f"./models/mod_{today}.pkl"
    fallback_model_path = "./models/mod_BESTRESULTS_0216.pkl"

    # Try today's model first, then fallback
    for path in [model_path, fallback_model_path]:
        if os.path.exists(path):
            print(f"Using model: {path}")
            with open(path, "rb") as f:
                return (
                    pickle.load(f),
                    feat.get_data(split_pred=True),
                    feat.get_data(data_type="test"),
                )

    print("ERROR: No model found!")
    print(f"  - Today's model not found at: {model_path}")
    print(f"  - Fallback model not found at: {fallback_model_path}")
    print(f"Please run main_preds.py first to generate a model")
    return None, None, None


def create_results_directory():
    """
    Creates a directory to store figure results if it doesn't already exist.

    The function creates a 'figures' directory in the current working directory
    if it does not already exist.

    Returns:
        str: The path to the figures directory.
    """
    figures_dir = "./figures"
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def visualize_temporal_patterns(train_data, figures_dir):
    """
    Visualizes temporal patterns in algal bloom severity data.

    This function creates and saves two visualizations:
    1. A bar chart showing the average severity of algal blooms across days of the week
    2. A line chart showing the monthly patterns of algal bloom severity over time

    Parameters
    ----------
    train_data : pandas.DataFrame
        DataFrame containing at minimum 'date' and 'severity' columns.
        The 'date' column should be able to be converted to datetime format.

    figures_dir : str
        Directory path where the generated visualization figures will be saved.

    Returns
    -------
    None
        The function saves the visualizations to the specified directory
        and does not return any values.

    Notes
    -----
    Generated files:
    - day_of_week_analysis.png: Bar chart of severity by day of week
    - monthly_patterns.png: Line chart of severity by month over time

    The function adds temporal features to the data, including:
    - day_of_week: Day of the week (0=Monday, 6=Sunday)
    - month: Month of the year (1-12)
    - year: Calendar year
    - year_month: Combined year-month periods for time series analysis
    """
    print("\nVisualizing temporal patterns...")

    # Add temporal features
    train_temporal = train_data.copy()
    train_temporal["date"] = pd.to_datetime(train_temporal["date"])
    train_temporal["day_of_week"] = train_temporal["date"].dt.dayofweek
    train_temporal["month"] = train_temporal["date"].dt.month
    train_temporal["year"] = train_temporal["date"].dt.year

    # Day of week analysis
    plt.figure(figsize=(12, 8))
    day_means = train_temporal.groupby("day_of_week")["severity"].mean()
    ax = sns.barplot(x=day_means.index, y=day_means.values, palette="viridis")

    # Add value labels on top of bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
        )

    plt.title(
        "Distribution of Algal Bloom Severity Across Days of the Week",
        pad=20,
        fontweight="bold",
    )
    plt.xlabel("Day of Week", labelpad=10)
    plt.ylabel("Average Severity", labelpad=10)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.xticks(
        range(7),
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    )
    plt.tight_layout()
    plt.savefig(
        f"{figures_dir}/day_of_week_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Monthly patterns
    plt.figure(figsize=(14, 8))
    # Group by year and month to create a time series
    train_temporal["year_month"] = train_temporal["date"].dt.to_period("M")
    month_means = train_temporal.groupby("year_month")["severity"].mean()

    # Convert PeriodIndex to datetime for plotting
    month_dates = month_means.index.to_timestamp()

    ax = sns.lineplot(
        x=month_dates, y=month_means.values, marker="o", linewidth=2, markersize=8
    )

    # Add value labels on data points (only for selected points to avoid clutter)
    for i, v in enumerate(month_means.values):
        if i % 6 == 0:  # Label every 6th point to reduce clutter
            ax.text(month_dates[i], v, f"{v:.2f}", ha="center", va="bottom")

    plt.title("Monthly Distribution of Bloom Severity", pad=20, fontweight="bold")
    plt.xlabel("Year-Month", labelpad=10)
    plt.ylabel("Average Severity", labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/monthly_patterns.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_regional_transfer(rm, train_data, figures_dir):
    """
    Analyzes and visualizes the transferability of algae bloom severity prediction models across different regions.

    This function performs two main visualization tasks:
    1. Creates a heatmap showing how well models trained on one region perform when applied to other regions (transfer performance)
    2. Generates violin plots depicting the error distribution of the primary model across different regions

    Parameters:
    ----------
    rm : RegMod or EnsMod
        The main regression model to evaluate against regional data
    train_data : pandas.DataFrame
        The training dataset containing features and target variable 'severity',
        must include a 'region' column to identify regional subsets
    figures_dir : str
        Directory path where generated visualization figures will be saved

    Returns:
    -------
    None
        The function saves visualization files to the specified directory and does not return any values

    Notes:
    -----
    - For each source region, a new ensemble model is trained consisting of CatBoost, LightGBM, and XGBoost models
    - The performance matrix shows RMSE values when applying models trained on one region (rows) to other regions (columns)
    - The violin plots show the distribution of prediction errors for the main model across all regions
    - Region names will be converted from numeric codes if feat.reg_ord_reverse mapping is available
    """
    print("\nVisualizing regional transfer...")

    # Split data by region
    regions = sorted(train_data["region"].unique())
    region_data = {
        region: train_data[train_data["region"] == region] for region in regions
    }

    # Create performance matrix
    performance_matrix = np.zeros((len(regions), len(regions)))

    # For each region, train a new ensemble model and test on other regions
    for i, source_region in enumerate(regions):
        source_data = region_data[source_region]

        print(f"Training ensemble model for region {source_region}...")
        region_train_data = source_data.copy()

        # Define feature sets
        feature_sets = define_feature_sets()
        sat_1025 = feature_sets["sat_1025"]

        # Create CatBoost model for this region
        cat = mod.RegMod(
            ord_vars=["cluster"],  # Removed "region" as we're using single region
            dat_vars=["date"],
            ide_vars=["latitude", "longitude", "maxe", "dife"],
            y="severity",
            mod=mod.CatBoostRegressor(
                iterations=380, depth=6, allow_writing_files=False, verbose=False
            ),
        )
        cat.fit(region_train_data, weight=False, cat=False)

        # Create LightGBM model for this region
        lig = mod.RegMod(
            ord_vars=[
                "cluster",
                "imtype",
            ],  # Removed "region" as we're using single region
            dat_vars=["date"],
            ide_vars=["latitude", "longitude", "elevation", "dife"] + sat_1025,
            y="severity",
            mod=mod.LGBMRegressor(n_estimators=470, max_depth=8),
        )
        lig.fit(region_train_data, weight=False, cat=True)

        # Create XGBoost model for this region
        xgb = mod.RegMod(
            ord_vars=["cluster"],  # Removed "region" as we're using single region
            dat_vars=["date"],
            y="severity",
            mod=mod.XGBRegressor(n_estimators=70, max_depth=2),
        )
        xgb.fit(region_train_data, weight=False, cat=False)

        # Create ensemble model for this region
        region_ensemble = mod.EnsMod(mods={"xgb": xgb, "cat": cat, "lig": lig})

        for j, target_region in enumerate(regions):
            target_data = region_data[target_region]
            y_pred = region_ensemble.predict(target_data)
            rmse = np.sqrt(np.mean((target_data["severity"] - y_pred) ** 2))
            performance_matrix[i, j] = rmse

    # Plot transfer performance matrix
    plt.figure(figsize=(14, 12))

    # Convert numeric regions to region names if possible
    region_names = []
    for region in regions:
        try:
            if hasattr(feat, "reg_ord_reverse") and isinstance(
                region, (int, np.integer)
            ):
                region_name = feat.reg_ord_reverse.get(int(region), f"Region {region}")
                region_names.append(str(region_name))
            else:
                region_names.append(str(region))
        except:
            # Fallback to string representation if conversion fails
            region_names.append(str(region))

    # Create a heatmap with region labels
    ax = sns.heatmap(
        performance_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=region_names,
        yticklabels=region_names,
        square=True,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "RMSE"},
    )

    # Improve readability of tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.title("Regional Transfer Performance (RMSE)", pad=20, fontweight="bold")
    plt.xlabel("Target Region", labelpad=10)
    plt.ylabel("Source Region", labelpad=10)
    plt.tight_layout()
    plt.savefig(
        f"{figures_dir}/regional_transfer_performance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create regional error distribution
    plt.figure(figsize=(12, 6))
    regional_errors = []
    region_labels = []

    for region in regions:
        region_data_subset = region_data[region]
        pred = rm.predict(region_data_subset)
        errors = region_data_subset["severity"] - pred
        regional_errors.append(errors)

        # Use region name if available
        if hasattr(feat, "reg_ord_reverse") and str(region).isdigit():
            region_name = feat.reg_ord_reverse.get(int(region), region)
            region_labels.append(str(region_name))
        else:
            region_labels.append(str(region))

    parts = plt.violinplot(regional_errors, showmeans=True)

    # Customize the plot
    for pc in parts["bodies"]:
        pc.set_facecolor("#3498db")
        pc.set_alpha(0.7)

    plt.title("Error Distribution Across Regions", pad=20, fontweight="bold")
    plt.xlabel("Region", labelpad=10)
    plt.ylabel("Prediction Error", labelpad=10)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.xticks(range(1, len(region_labels) + 1), region_labels)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_feature_importance(rm, train_data, figures_dir):
    """
    Creates and saves various feature importance visualizations using SHAP values.

    This function analyzes the trained model to understand which features have the
    most significant impact on algal bloom severity predictions. It generates:
    - A horizontal bar chart showing overall feature importance
    - Dependence plots for key features (elevation, latitude)
    - A SHAP summary plot

    The function uses stratified sampling to ensure representation across all regions
    in the dataset and focuses on numerical features for the SHAP analysis.

    Parameters
    ----------
    rm : object
        Trained regression model with a predict method.
    train_data : pandas.DataFrame
        Training dataset with features and target variable 'severity'.
        Must include a 'region' column for stratified sampling.
    figures_dir : str
        Directory path where the generated visualization figures will be saved.

    Notes
    -----
    The function requires the following dependencies:
    - shap
    - scipy.stats.gaussian_kde
    - statsmodels.nonparametric.smoothers_lowess.lowess
    - define_feature_sets() function must be available in the current scope

    The function generates the following files in the figures_dir:
    - feature_importance.png: Horizontal bar chart of feature importance
    - shap_dependence_elevation.png: Dependence plot for elevation (if available)
    - shap_dependence_latitude.png: Dependence plot for latitude (if available)
    - shap_summary.png: SHAP summary plot of all features
    """
    import shap
    from scipy.stats import gaussian_kde
    from statsmodels.nonparametric.smoothers_lowess import lowess

    print("\nVisualizing feature importance...")

    # Sample data for SHAP analysis
    sample_size = min(500, len(train_data))

    X_sample = train_data.drop(columns=["severity"]).sample(
        min(500, len(train_data)), random_state=42
    )

    # Stratified sampling to ensure representation across regions
    region_groups = []
    regions = train_data["region"].unique()
    for region in regions:
        region_data = train_data[train_data["region"] == region]
        # Sample proportionally from each region
        n_samples = max(10, int(sample_size * len(region_data) / len(train_data)))
        sampled = region_data.sample(min(n_samples, len(region_data)), random_state=42)
        region_groups.append(sampled)
    X_sample = pd.concat(region_groups)

    # Extract target and features
    y_sample = X_sample["severity"]
    X_sample = X_sample.drop(columns=["severity"])

    # Use numeric columns only for SHAP analysis
    numeric_cols = X_sample.select_dtypes(include=["number"]).columns
    X_numeric = X_sample[numeric_cols]

    # Create model wrapper for SHAP
    def model_wrapper(X_data):
        if isinstance(X_data, np.ndarray):
            # Create DataFrame with just the numeric columns we're analyzing
            X_for_model = pd.DataFrame(X_data, columns=X_numeric.columns)

            # Add any missing columns required by the model
            missing_cols = set(X_sample.columns) - set(X_for_model.columns)
            for col in missing_cols:
                X_for_model[col] = X_sample[col].iloc[
                    0
                ]  # Use first value as placeholder
        else:
            X_for_model = X_data.copy()

        if "date" not in X_for_model.columns and "date" in X_sample.columns:
            X_for_model["date"] = X_sample["date"].iloc[0]

        return rm.predict(X_for_model)

    # Create background dataset
    background_size = min(200, len(X_numeric))
    background_data = shap.sample(X_numeric, background_size)

    # Use numeric columns only for SHAP analysis
    numeric_cols = X_sample.select_dtypes(include=["number"]).columns
    X_numeric = X_sample[numeric_cols]

    print("Calculating SHAP values (this might take a while)...")
    # Calculate SHAP values - using KernelExplainer for a model-agnostic approach
    try:
        explainer = shap.KernelExplainer(model_wrapper, background_data)
        shap_values = explainer.shap_values(X_numeric)
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
        return

    # Define feature categories for better visualization
    feature_set = define_feature_sets()
    feature_categories = {
        "Geographic": feature_set["geographic"],
        "Topographic": feature_set["topographic"],
        "Lake Properties": [
            "prop_lake_500",
            "prop_lake_1000",
            "prop_lake_2500",
        ],
        "Satellite 1000m": feature_set["sat_1000"],
        "Satellite 2500m": feature_set["sat_2500"],
    }

    # Color palette for categories
    category_colors = {
        "Geographic": "#2ecc71",  # Green
        "Topographic": "#3498db",  # Blue
        "Lake Properties": "#e74c3c",  # Red
        "Satellite 1000m": "#9b59b6",  # Purple
        "Satellite 2500m": "#f1c40f",  # Yellow
        "Other": "#95a5a6",  # Gray
    }

    # Calculate feature importance
    feature_importance = {}
    for i, col in enumerate(numeric_cols):
        feature_importance[col] = np.abs(shap_values[:, i]).mean()

    # Create DataFrame for plotting
    df = pd.DataFrame(
        {
            "feature": list(feature_importance.keys()),
            "importance": list(feature_importance.values()),
        }
    )

    # Assign category and color to each feature
    df.loc[:, ["category", "color"]] = ["Other", category_colors["Other"]]
    for category, features in feature_categories.items():
        for feature in features:
            mask = df["feature"] == feature
            df.loc[mask, "category"] = category
            df.loc[mask, "color"] = category_colors[category]

    # Sort by importance
    df = df.sort_values("importance", ascending=True)

    # Create the plot
    plt.figure(figsize=(14, 10))

    # Create horizontal bar plot
    bars = plt.barh(range(len(df)), df["importance"], color=df["color"])

    # Customize the plot
    plt.yticks(range(len(df)), df["feature"], fontsize=10)
    plt.xlabel("SHAP Importance", fontsize=12, labelpad=10)
    plt.title(
        "Feature Importance in Algal Bloom Severity Predictions",
        pad=20,
        fontsize=14,
        fontweight="bold",
    )

    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,  # Offset for label position
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    # Create legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=category)
        for category, color in category_colors.items()
        if category in df["category"].values
    ]
    plt.legend(
        handles=legend_elements,
        title="Feature Categories",
        title_fontsize=12,
        fontsize=10,
        loc="lower right",
    )

    # Add grid for better readability
    plt.grid(True, axis="x", linestyle="--", alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    plt.savefig(f"{figures_dir}/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Function to create dependence plot
    def create_dependence_plot(feature_name, feature_idx, degree=1):
        plt.figure(figsize=(12, 8))

        # Get feature values and corresponding SHAP values
        x = X_numeric[feature_name].values
        y = shap_values[:, feature_idx]

        # Scatter plot with transparency based on point density
        if len(x) > 100:
            # Calculate point density for color gradient
            xy = np.vstack([x, y])
            # Use Gaussian KDE for density estimation (smoothed scatter plot)
            try:
                z = gaussian_kde(xy)(xy)
                plt.scatter(
                    x,
                    y,
                    c=z,
                    s=70,
                    alpha=0.7,
                    cmap="viridis",
                    edgecolor="k",
                    linewidth=0.5,
                )
            except:
                # Fallback if KDE fails
                plt.scatter(
                    x, y, c="lightblue", s=70, alpha=0.7, edgecolor="k", linewidth=0.5
                )
        else:
            plt.scatter(
                x, y, c="lightblue", s=70, alpha=0.7, edgecolor="k", linewidth=0.5
            )

        # Add LOWESS smoother or polynomial fit if it fails
        try:
            # Sort points for smooth line
            sorted_idx = np.argsort(x)
            x_sorted = x[sorted_idx]
            y_sorted = y[sorted_idx]

            # Apply LOWESS smoother (robust to outliers)
            smoothed = lowess(y_sorted, x_sorted, frac=0.6, it=3)
            plt.plot(
                smoothed[:, 0],
                smoothed[:, 1],
                color="red",
                linewidth=3,
                label="LOWESS trend",
            )

            # Add confidence band (approximated from scatter)
            residuals = y - np.interp(x, smoothed[:, 0], smoothed[:, 1])
            std_resid = np.std(residuals)
            plt.fill_between(
                smoothed[:, 0],
                smoothed[:, 1] - 1.96 * std_resid,
                smoothed[:, 1] + 1.96 * std_resid,
                alpha=0.2,
                color="red",
                label="95% Confidence Band",
            )
        except Exception as e:
            print(f"Error applying LOWESS smoother: {e}")
            # Fallback to simple polynomial fit if LOWESS fails
            coef = np.polyfit(x, y, degree)
            poly_line = np.poly1d(coef)
            x_range = np.linspace(min(x), max(x), 100)
            plt.plot(
                x_range,
                poly_line(x_range),
                color="red",
                linewidth=3,
                label=f"Degree {degree} polynomial",
            )

        # Add horizontal line at y=0 for reference
        plt.axhline(y=0, color="gray", linestyle="--", alpha=0.6)

        # Add a density histogram on top to show data distribution
        ax2 = plt.gca().twinx()
        ax2.hist(x, bins=30, alpha=0.2, color="gray", density=True)
        ax2.set_ylabel("Density", color="gray")
        ax2.tick_params(axis="y", colors="gray")
        ax2.set_ylim(bottom=0)
        # Hide density axis tick labels to reduce clutter
        ax2.set_yticks([])

        # Customize plot appearance
        plt.title(
            f"SHAP Dependence Plot for {feature_name.capitalize()}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel(f"{feature_name.capitalize()} Value", fontsize=14, labelpad=10)
        plt.ylabel("SHAP Value (Impact on Prediction)", fontsize=14, labelpad=10)
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend(loc="best")

        # Add annotations explaining the plot
        shap_mean = np.abs(y).mean()
        y_range = max(y) - min(y)
        pos_y = min(y) + 0.9 * y_range

        text = (
            f"Mean |SHAP| value: {shap_mean:.3f}\n"
            f"Positive values: Higher severity\n"
            f"Negative values: Lower severity"
        )

        plt.annotate(
            text,
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
            fontsize=11,
            ha="left",
            va="bottom",
        )

        plt.tight_layout()
        plt.savefig(
            f"{figures_dir}/shap_dependence_{feature_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Create dependence plots for Elevation and Latitude
    if "elevation" in X_numeric.columns:
        elevation_idx = list(X_numeric.columns).index("elevation")
        create_dependence_plot("elevation", elevation_idx, degree=2)

    if "latitude" in X_numeric.columns:
        latitude_idx = list(X_numeric.columns).index("latitude")
        create_dependence_plot("latitude", latitude_idx, degree=1)

    # Create a SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_numeric, show=False, plot_size=(12, 8))
    plt.title("SHAP Feature Importance Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def visualize_spatial_patterns(train_data, figures_dir):
    """
    Visualize the spatial patterns of algal bloom severity and observation density.

    This function creates two visualizations:
    1. A scatter plot showing the spatial distribution of algal bloom severity.
    2. A 2D histogram showing the density of algal bloom observations.

    Both visualizations are saved as PNG files in the specified directory.

    Parameters
    ----------
    train_data : pandas.DataFrame
        DataFrame containing the training data with columns:
        - 'longitude': Geographic longitude coordinates
        - 'latitude': Geographic latitude coordinates
        - 'severity': Algal bloom severity scores

    figures_dir : str
        Directory path where the generated figures will be saved.

    Returns
    -------
    None
        The function saves the visualizations to disk but does not return any values.

    Notes
    -----
    The visualizations use the 'YlOrRd' colormap to maintain consistency with the report.
    """
    print("\nVisualizing spatial patterns...")

    # Spatial distribution
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(
        train_data["longitude"],
        train_data["latitude"],
        c=train_data["severity"],
        cmap="YlOrRd",
        s=100,
        alpha=0.6,
    )
    plt.colorbar(scatter, label="Severity Score")
    plt.title("Spatial Distribution of Algal Bloom Severity", pad=20, fontweight="bold")
    plt.xlabel("Longitude", labelpad=10)
    plt.ylabel("Latitude", labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/spatial_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Observation density
    plt.figure(figsize=(15, 10))
    plt.hist2d(train_data["longitude"], train_data["latitude"], bins=50, cmap="YlOrRd")
    plt.colorbar(label="Number of Observations")
    plt.title("Density of Algal Bloom Observations", pad=20, fontweight="bold")
    plt.xlabel("Longitude", labelpad=10)
    plt.ylabel("Latitude", labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/observation_density.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_model_comparison_figures(train_data, rm, figures_dir):
    """
    Creates visualizations to compare different machine learning models for algal bloom severity prediction.

    This function trains multiple regression models (XGBoost, CatBoost, LightGBM, and an Ensemble),
    evaluates their performance, and generates visualizations for comparison. It also compares
    random vs. spatial validation approaches.

    Parameters:
    -----------
    train_data : pandas.DataFrame
        The training dataset containing features and target variable 'severity'.

    rm : object
        The reference model object (currently not used in the function).

    figures_dir : str
        Directory path where the generated figures will be saved.

    Generated Visualizations:
    ------------------------
    - Prediction scatter plots for each model
    - Performance comparison (RMSE and MAE) bar charts
    - Feature importance comparison across models
    - Comparison of random vs. spatial validation splits (RMSE and MAE)
    - Generalization gap visualization showing difference between train and test errors
    - Combined metrics visualization across all splits and datasets

    Returns:
    --------
    None
        Outputs visualizations to the specified figures_dir and prints progress messages.
    """
    """Create model comparison visualizations"""
    print("\nCreating model comparison visualizations...")

    # Import necessary libraries
    from src import feat, mod
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    # Define feature sets
    feature_sets = define_feature_sets()
    sat_1025 = feature_sets["sat_1025"]

    # Create validation split from training data
    train_subset, val_subset = train_test_split(
        train_data, test_size=0.3, random_state=42
    )
    print(
        f"Created validation split - Train: {len(train_subset)}, Validation: {len(val_subset)}"
    )

    # Initialize models for comparison
    xgb = mod.RegMod(
        ord_vars=["region", "cluster"],
        dat_vars=["date"],
        y="severity",
        mod=mod.XGBRegressor(n_estimators=70, max_depth=2),
    )

    cat = mod.RegMod(
        ord_vars=["region", "cluster"],
        dat_vars=["date"],
        ide_vars=["latitude", "longitude", "maxe", "dife"],
        y="severity",
        mod=mod.CatBoostRegressor(
            iterations=380, depth=6, allow_writing_files=False, verbose=False
        ),
    )

    lgbm = mod.RegMod(
        ord_vars=["region", "cluster", "imtype"],
        dat_vars=["date"],
        ide_vars=["latitude", "longitude", "elevation", "dife"] + sat_1025,
        y="severity",
        mod=mod.LGBMRegressor(n_estimators=470, max_depth=8),
    )

    # Fit the models
    print("Fitting models on training subset...")
    xgb.fit(train_subset, weight=False, cat=False)
    cat.fit(train_subset, weight=False, cat=False)
    lgbm.fit(train_subset, weight=False, cat=True)

    # Create ensemble model
    ensemble = mod.EnsMod(mods={"xgb": xgb, "cat": cat, "lgbm": lgbm})

    # Create predictions
    val_subset["XGBoost_pred"] = xgb.predict(val_subset)
    val_subset["CatBoost_pred"] = cat.predict(val_subset)
    val_subset["LightGBM_pred"] = lgbm.predict(val_subset)
    val_subset["Ensemble_pred"] = ensemble.predict(val_subset)

    model_names = ["XGBoost", "CatBoost", "LightGBM", "Ensemble"]

    # Prediction scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(
        "Model Predictions vs True Values", fontsize=16, fontweight="bold", y=1.02
    )

    for i, model in enumerate(model_names):
        ax = axes[i // 2, i % 2]
        true_values = val_subset["severity"]
        pred = val_subset[f"{model}_pred"]

        # Create scatter plot
        ax.scatter(true_values, pred, alpha=0.5, c="#3498db")

        # Add perfect prediction line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, "r--", alpha=0.8, zorder=0)

        # Calculate metrics
        rmse = np.sqrt(np.mean((true_values - pred) ** 2))
        r2 = np.corrcoef(true_values, pred)[0, 1] ** 2

        # Add metrics text
        ax.text(
            0.05,
            0.95,
            f"RMSE: {rmse:.3f}\nR²: {r2:.3f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

        ax.set_title(model)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/prediction_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Performance comparison
    performance_data = []
    for model in model_names:
        rmse = np.sqrt(
            np.mean((val_subset["severity"] - val_subset[f"{model}_pred"]) ** 2)
        )
        mae = np.mean(np.abs(val_subset["severity"] - val_subset[f"{model}_pred"]))
        performance_data.extend(
            [
                {"Model": model, "Metric Value": rmse, "Metric Type": "RMSE"},
                {"Model": model, "Metric Value": mae, "Metric Type": "MAE"},
            ]
        )

    performance_df = pd.DataFrame(performance_data)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=performance_df, x="Model", y="Metric Value", hue="Metric Type")
    plt.title("Model Performance Comparison", pad=20, fontweight="bold")
    plt.xlabel("Model", labelpad=10)
    plt.ylabel("Error", labelpad=10)

    # Add value labels on bars
    for i in plt.gca().containers:
        plt.gca().bar_label(i, fmt="%.3f")

    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(
        f"{figures_dir}/performance_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"model comparison graphs saved to {figures_dir}/")

    # ----- Feature Importance for the models -----
    feature_data = []

    # Extract feature names and importances from XGBoost
    xgb_features = xgb.feat_import()
    xgb_features = xgb_features.rename(columns={"Var": "Feature", "FI": "Importance"})
    xgb_features["Model"] = "XGBoost"
    feature_data.append(xgb_features)

    # Extract feature names and importances from CatBoost
    cat_features = cat.feat_import()
    cat_features = cat_features.rename(columns={"Var": "Feature", "FI": "Importance"})
    cat_features["Model"] = "CatBoost"
    feature_data.append(cat_features)

    # Extract feature names and importances from LightGBM
    lgbm_features = lgbm.feat_import()
    lgbm_features = lgbm_features.rename(columns={"Var": "Feature", "FI": "Importance"})
    lgbm_features["Model"] = "LightGBM"
    feature_data.append(lgbm_features)

    # Combine all feature data into a single DataFrame
    feature_data = pd.concat(feature_data, ignore_index=True)
    print(feature_data.head())

    feature_data["Importance"] = pd.to_numeric(
        feature_data["Importance"], errors="coerce"
    )

    # Normalize importances within each model
    for model in feature_data["Model"].unique():
        model_importances = feature_data.loc[
            feature_data["Model"] == model, "Importance"
        ]
        max_value = model_importances.max()
        if max_value > 0:  # Avoid division by zero
            feature_data.loc[feature_data["Model"] == model, "Importance"] = (
                model_importances / max_value
            )

    # Get top features (based on average importance across models)
    top_features = (
        feature_data.groupby("Feature")["Importance"].mean().nlargest(10).index
    )

    # Filter DataFrame to include only top features
    top_df = feature_data[feature_data["Feature"].isin(top_features)]

    # Create plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="Importance", y="Feature", hue="Model", data=top_df)

    # Customize plot
    plt.title(
        "Feature Importance Comparison Across Models", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Normalized Importance")
    plt.ylabel("Feature")
    plt.grid(True, axis="x", linestyle="--", alpha=0.3)

    # Add legend
    plt.legend(title="Model", loc="lower right")

    # Save figure
    plt.tight_layout()
    plt.savefig(
        f"{figures_dir}/feature_importance_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"feature importance graph saved to {figures_dir}/")

    # ----- Compare spatial split (competition approach) vs random split validation -----

    # Create a random train/test split for comparison
    random_train, random_test = train_test_split(
        train_data, test_size=0.3, random_state=42
    )

    # Get the original training data
    train_data = feat.get_data(split_pred=True)

    # Get the combined train/test data to build the split_pred model
    all_data = feat.get_both()  # Combined train/test data

    # Predicting probability of in test set
    cm = mod.CatMod(
        ord_vars=["region"],
        ide_vars=["latitude", "longitude", "maxe", "dife"],
        y="test",
    )
    cm.fit(all_data)

    # Predict probability in the train set
    split_pred2 = cm.predict(train_data)
    train_data["split_pred2"] = split_pred2

    # Create spatial train/test split based on split_pred2
    # Use threshold of 0.5 for simplicity
    spatial_train = train_data[train_data["split_pred2"] < 0.5].copy()
    spatial_test = train_data[train_data["split_pred2"] >= 0.5].copy()

    print(f"Random split - Train: {len(random_train)}, Test: {len(random_test)}")
    print(f"Spatial split - Train: {len(spatial_train)}, Test: {len(spatial_test)}")

    # --- Model for Random Split ---
    lgbm_random = mod.RegMod(
        ord_vars=["region", "cluster", "imtype"],
        dat_vars=["date"],
        ide_vars=["latitude", "longitude", "elevation", "dife"] + sat_1025,
        y="severity",
        mod=mod.LGBMRegressor(n_estimators=470, max_depth=8),
    )

    # Train on random_train
    lgbm_random.fit(random_train, weight=False, cat=True)

    # Make predictions
    random_train_pred = lgbm_random.predict(random_train)
    random_test_pred = lgbm_random.predict(random_test)

    # Calculate metrics for random split
    random_metrics = {
        "train_rmse": np.sqrt(
            mean_squared_error(random_train["severity"], random_train_pred)
        ),
        "test_rmse": np.sqrt(
            mean_squared_error(random_test["severity"], random_test_pred)
        ),
        "train_mae": mean_absolute_error(random_train["severity"], random_train_pred),
        "test_mae": mean_absolute_error(random_test["severity"], random_test_pred),
    }

    # --- Model for Spatial Split ---
    lgbm_spatial = mod.RegMod(
        ord_vars=["region", "cluster", "imtype"],
        dat_vars=["date"],
        ide_vars=["latitude", "longitude", "elevation", "dife"] + sat_1025,
        y="severity",
        weight="split_pred2",
        mod=mod.LGBMRegressor(n_estimators=470, max_depth=8),
    )

    # Train on spatial_train
    lgbm_spatial.fit(spatial_train, weight=True, cat=True)

    # Make predictions
    spatial_train_pred = lgbm_spatial.predict(spatial_train)
    spatial_test_pred = lgbm_spatial.predict(spatial_test)

    # Calculate metrics for spatial split
    spatial_metrics = {
        "train_rmse": np.sqrt(
            mean_squared_error(spatial_train["severity"], spatial_train_pred)
        ),
        "test_rmse": np.sqrt(
            mean_squared_error(spatial_test["severity"], spatial_test_pred)
        ),
        "train_mae": mean_absolute_error(spatial_train["severity"], spatial_train_pred),
        "test_mae": mean_absolute_error(spatial_test["severity"], spatial_test_pred),
    }

    # Print metrics
    print("\nRandom Split Metrics:")
    for metric, value in random_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nSpatial Split Metrics:")
    for metric, value in spatial_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # --- Create visualizations ---

    # Prepare data for plotting
    plot_data = []

    # Add random split metrics
    for metric, value in random_metrics.items():
        dataset = "Train" if "train" in metric else "Test"
        metric_type = "RMSE" if "rmse" in metric else "MAE"
        plot_data.append(
            {
                "Split Type": "Random Split",
                "Dataset": dataset,
                "Metric": metric_type,
                "Value": value,
            }
        )

        # Add spatial split metrics
    for metric, value in spatial_metrics.items():
        dataset = "Train" if "train" in metric else "Test"
        metric_type = "RMSE" if "rmse" in metric else "MAE"
        plot_data.append(
            {
                "Split Type": "Spatial Split",
                "Dataset": dataset,
                "Metric": metric_type,
                "Value": value,
            }
        )

    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)

    # --- Create separate bar charts for RMSE and MAE ---
    metrics = ["RMSE", "MAE"]

    for metric in metrics:
        plt.figure(figsize=(12, 8))
        metric_data = plot_df[plot_df["Metric"] == metric]

        ax = sns.barplot(
            data=metric_data,
            x="Split Type",
            y="Value",
            hue="Dataset",
            palette=["#3498db", "#e74c3c"],  # Blue for Train, Red for Test
        )

        plt.title(
            f"{metric} Comparison: Random Split vs Spatial Split",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.ylabel(f"{metric} (lower is better)", fontsize=14, labelpad=10)
        plt.xlabel("Validation Method", fontsize=14, labelpad=10)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f")

        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.legend(title="Dataset")
        plt.tight_layout()

        plt.savefig(
            f"{figures_dir}/{metric.lower()}_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # --- Create train/test difference plot to show generalization gap ---
    plt.figure(figsize=(12, 8))

    # Calculate differences between train and test errors
    diff_data = []

    for split_type in ["Random Split", "Spatial Split"]:
        for metric in ["RMSE", "MAE"]:
            train_val = plot_df[
                (plot_df["Split Type"] == split_type)
                & (plot_df["Dataset"] == "Train")
                & (plot_df["Metric"] == metric)
            ]["Value"].values[0]

            test_val = plot_df[
                (plot_df["Split Type"] == split_type)
                & (plot_df["Dataset"] == "Test")
                & (plot_df["Metric"] == metric)
            ]["Value"].values[0]

            diff = test_val - train_val

            diff_data.append(
                {"Split Type": split_type, "Metric": metric, "Generalization Gap": diff}
            )

    diff_df = pd.DataFrame(diff_data)

    # Plot the differences
    ax = sns.barplot(
        data=diff_df,
        x="Split Type",
        y="Generalization Gap",
        hue="Metric",
        palette=["#9b59b6", "#2ecc71"],  # Purple for RMSE, Green for MAE
    )

    plt.title(
        "Generalization Gap (Test Error - Train Error)\nLower values indicate better generalization",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.ylabel("Test Error - Train Error", fontsize=12, labelpad=10)
    plt.xlabel("Split Type", fontsize=12, labelpad=10)

    # Add value labels
    for container in ax.containers:
        plt.gca().bar_label(container, fmt="%.4f")

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(title="Metric Type")

    plt.tight_layout()
    plt.savefig(
        f"{figures_dir}/generalization_gap_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # --- Create combined metrics visualization ---
    # Reshape data for a grouped bar chart with all metrics
    plt.figure(figsize=(15, 10))

    # Use custom colors for better distinction
    colors = {
        "Random Split - Train": "#3498db",  # Light blue
        "Random Split - Test": "#2980b9",  # Dark blue
        "Spatial Split - Train": "#e74c3c",  # Light red
        "Spatial Split - Test": "#c0392b",  # Dark red
    }

    # Create new column combining split type and dataset
    plot_df["Split_Dataset"] = plot_df["Split Type"] + " - " + plot_df["Dataset"]

    # Create grouped bar chart
    g = sns.catplot(
        data=plot_df,
        kind="bar",
        x="Metric",
        y="Value",
        hue="Split_Dataset",
        palette=colors,
        height=8,
        aspect=1.5,
        legend_out=False,  # Keep the legend inside the plot
    )

    # Set axis labels and title
    g.set_axis_labels("Metric Type", "Error Value (lower is better)", 14)
    g.fig.suptitle(
        "Comparison of All Metrics Across Split Types and Datasets",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    # Get the current axes and add the legend properly
    ax = g.axes.flat[0]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Split Type - Dataset", loc="upper right")

    # Add grid
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{figures_dir}/all_metrics_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Visualizations saved to {figures_dir}/")


def main():
    """
    Main function to execute the algaebloom prediction and analysis workflow.

    This function performs the following tasks:
    1. Creates a directory for storing generated figures
    2. Attempts to create a new prediction model
    3. Loads an existing model and associated training/test data
    4. Generates various visualizations to analyze the model and data

    The visualization functions are modular and can be enabled/disabled by
    uncommenting/commenting the respective function calls.

    Returns:
        None

    Raises:
        Exception: If errors occur during model creation or visualization
    """

    # Create figures directory
    figures_dir = create_results_directory()
    print(f"Created figures directory: {figures_dir}")

    # Run prediction
    try:
        create_model()
        print("Model created successfully.")
    except Exception as e:
        print(f"Error creating model: {e}")
        # Continue with analysis of existing model

    # Load model and data
    rm, train_data, test_data = load_model_and_data()

    # Check if model and data loaded correctly
    if rm is None or train_data is None or test_data is None:
        print("Failed to load model or data. Exiting.")
        return

    # Run all visualizations
    try:
        visualize_temporal_patterns(train_data, figures_dir)
        visualize_regional_transfer(rm, train_data, figures_dir)
        visualize_feature_importance(rm, train_data, figures_dir)
        visualize_spatial_patterns(train_data, figures_dir)
        create_model_comparison_figures(train_data, rm, figures_dir)
        print(f"\nAll visualizations completed and saved to {figures_dir}")
    except Exception as e:
        print(f"Error during visualization: {e}")


if __name__ == "__main__":
    main()
