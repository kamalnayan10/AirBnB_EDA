# streamlit_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap # Added for heatmap
# from folium import Polygon # Polygon is available via folium.Polygon <-- This was in your original prompt but not used, folium.Polygon is correct way
from shapely.geometry import MultiPoint
from streamlit_folium import st_folium
import google.generativeai as genai

import scipy.stats as stats
import pingouin as pg
import numpy as np
import statsmodels.api as sm

# --- Initialize session state for filters and map ---
if "filters_changed" not in st.session_state:
    st.session_state.filters_changed = True # Initially true to build the map
if "cached_map" not in st.session_state:
    st.session_state.cached_map = None
if "prev_filters" not in st.session_state:
    st.session_state.prev_filters = {
        "borough": None,
        "room_type": None,
        "price_range": (0,0),
        "min_avail": -1,
        "heatmap_option": "None" # Added for heatmap selection
    }

# --- Load and Preprocess Data ---
@st.cache_data
def load_and_preprocess_data():
    try:
        df_loaded = pd.read_csv("AB_NYC_2019_cleaned.csv")
        required_cols = ['neighbourhood_group', 'room_type', 'price', 'availability_365',
                         'latitude', 'longitude', 'name', 'last_review',
                         'calculated_host_listings_count', 'minimum_nights', 'reviews_per_month', 'number_of_reviews']
        if not all(col in df_loaded.columns for col in required_cols):
            st.error(f"The CSV file is missing some required columns. Expected: {', '.join(required_cols)}. Using sample data instead.")
            raise FileNotFoundError # Trigger fallback to sample data
        # --- Data Preprocessing for new features ---
        # Convert 'last_review' to datetime and extract month/year
        df_loaded['last_review_date'] = pd.to_datetime(df_loaded['last_review'], errors='coerce')
        df_loaded['review_year_month'] = df_loaded['last_review_date'].dt.to_period('M')

        # Fill NaN reviews_per_month with 0 if appropriate for your analysis
        if 'reviews_per_month' in df_loaded.columns:
            df_loaded['reviews_per_month'].fillna(0, inplace=True)

        return df_loaded

    except FileNotFoundError:
        st.warning("Using sample data as 'AB_NYC_2019_cleaned.csv' was not found or was incomplete.")
        sample_data = {
            'neighbourhood_group': ['Manhattan', 'Brooklyn', 'Queens', 'Staten Island', 'Bronx'] * 200,
            'room_type': ['Entire home/apt', 'Private room', 'Shared room'] * (1000 // 3 + 1),
            'price': [150, 75, 50, 200, 60, 120, 90, 400, 30, 250] * 100,
            'availability_365': [365, 100, 20, 300, 50, 150, 0, 250, 10, 90] * 100,
            'latitude': [40.7128 + (i*0.001) for i in range(1000)],
            'longitude': [-74.0060 + (i*0.001) for i in range(1000)],
            'name': [f'Sample Listing {i+1}' for i in range(1000)],
            'id': list(range(1,1001)),
            'host_id': list(range(101,1101)),
            'neighbourhood': ['Midtown', 'Williamsburg', 'Astoria'] * (1000 // 3 +1 ),
            'minimum_nights': [1,2,1,3,1,2,4,1,2,1] * 100,
            'number_of_reviews': [10,5,2,20,0,1,15,3,6,9] * 100,
            'last_review': pd.to_datetime(['2019-01-01', '2019-02-15', '2018-03-20', '2019-04-10', '2017-05-05'] * 200).strftime('%Y-%m-%d'),
            'reviews_per_month': [1.0, 0.5, 0.2, 2.0, 0.0, 0.1, 1.5, 0.3, 0.6, 0.9] * 100,
            'calculated_host_listings_count': [1,1,1,2,1,1,3,1,1,1,5,10] * (1000//12 +1)
        }
        df_sample = pd.DataFrame(sample_data)
        df_sample = df_sample.iloc[:1000] # Ensure consistent length
        df_sample['last_review_date'] = pd.to_datetime(df_sample['last_review'], errors='coerce')
        df_sample['review_year_month'] = df_sample['last_review_date'].dt.to_period('M')
        df_sample['reviews_per_month'].fillna(0, inplace=True)
        return df_sample

df = load_and_preprocess_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Listings")
current_filters = {}
current_filters["borough"] = st.sidebar.selectbox("Select Borough", ["All"] + sorted(df['neighbourhood_group'].unique()), key="sb_borough")
current_filters["room_type"] = st.sidebar.selectbox("Select Room Type", ["All"] + sorted(df['room_type'].unique()), key="sb_room_type")

min_price_data = int(df['price'].min()) if not df.empty and 'price' in df.columns else 0
max_price_data = int(df['price'].max()) if not df.empty and 'price' in df.columns else 1000
default_price_low = 50 if 50 >= min_price_data else min_price_data
default_price_high = 500 if 500 <= max_price_data else max_price_data
if default_price_low > default_price_high: default_price_low, default_price_high = min_price_data, max_price_data

current_filters["price_range"] = st.sidebar.slider("Price Range ($)", min_value=min_price_data, max_value=max_price_data, value=(default_price_low, default_price_high), key="sl_price_range")
current_filters["min_avail"] = st.sidebar.slider("Minimum Availability (days)", 0, 365, 30, key="sl_min_avail")

# --- NEW: Heatmap selection for the map ---
current_filters["heatmap_option"] = st.sidebar.selectbox(
    "Select Heatmap Layer",
    ["None", "Price Heatmap", "Listing Density Heatmap"],
    key="sb_heatmap"
)

if st.session_state.prev_filters != current_filters:
    st.session_state.filters_changed = True
    st.session_state.prev_filters = current_filters.copy()

# --- Apply filters to the DataFrame ---
filtered_df = df.copy()
if current_filters["borough"] != "All":
    filtered_df = filtered_df[filtered_df['neighbourhood_group'] == current_filters["borough"]]
if current_filters["room_type"] != "All":
    filtered_df = filtered_df[filtered_df['room_type'] == current_filters["room_type"]]
if 'price' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['price'] >= current_filters["price_range"][0]) & (filtered_df['price'] <= current_filters["price_range"][1])]
if 'availability_365' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['availability_365'] >= current_filters["min_avail"]]


# --- Main Page Layout ---
st.title("🏙️ NYC Airbnb Strategic Insights Dashboard")

# --- NEW: Key Metrics Summary Cards ---
st.subheader("📈 Key Metrics (Based on Filters)")
if not filtered_df.empty:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Listings", f"{filtered_df.shape[0]:,}")
    col_m2.metric("Avg. Price", f"${filtered_df['price'].mean():.2f}" if 'price' in filtered_df.columns and filtered_df['price'].notna().any() else "N/A")
    col_m3.metric("Median Price", f"${filtered_df['price'].median():.2f}" if 'price' in filtered_df.columns and filtered_df['price'].notna().any() else "N/A")
    col_m4.metric("Avg. Availability", f"{filtered_df['availability_365'].mean():.0f} days" if 'availability_365' in filtered_df.columns and filtered_df['availability_365'].notna().any() else "N/A")
else:
    st.info("No listings match the current filters to display key metrics.")
st.markdown("---")


# --- Existing Charts Section (Price Distribution, Room Type, Borough vs Price) ---
st.subheader("📊 Core Data Visualizations")
if not filtered_df.empty:
    col1, col2 = st.columns(2)
    with col1:
        if 'price' in filtered_df.columns and filtered_df['price'].notna().any():
            st.markdown("##### Price Distribution")
            fig, ax = plt.subplots()
            sns.histplot(filtered_df["price"].dropna(), bins=30, kde=True, ax=ax) # Added .dropna()
            plt.xlabel("Price ($)")
            plt.ylabel("Number of Listings")
            st.pyplot(fig)
        else:
            st.markdown("##### Price Distribution")
            st.write("Price data not available for distribution plot.")

    with col2:
        if 'room_type' in filtered_df.columns and filtered_df['room_type'].notna().any():
            st.markdown("##### Room Type Breakdown")
            fig, ax = plt.subplots()
            sns.countplot(data=filtered_df.dropna(subset=['room_type']), y="room_type", order=filtered_df["room_type"].value_counts().index, ax=ax, palette="viridis")
            plt.xlabel("Count")
            plt.ylabel("Room Type")
            st.pyplot(fig)
        else:
            st.markdown("##### Room Type Breakdown")
            st.write("Room type data not available for breakdown plot.")


    if 'neighbourhood_group' in filtered_df.columns and 'price' in filtered_df.columns and \
       filtered_df['price'].notna().any() and filtered_df['neighbourhood_group'].notna().any():
        st.markdown("##### Price by Borough (Box Plot)")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Ensure price is numeric for quantile calculation
        numeric_prices = pd.to_numeric(filtered_df['price'], errors='coerce').dropna()
        if not numeric_prices.empty:
            sns.boxplot(data=filtered_df.dropna(subset=['neighbourhood_group', 'price']), x="neighbourhood_group", y="price", ax=ax, palette="pastel")
            q95 = numeric_prices.quantile(0.95)
            ax.set_ylim(0, q95 if q95 > 0 else (numeric_prices.max() if not numeric_prices.empty else 500) )
            plt.xlabel("Borough")
            plt.ylabel("Price ($)")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        else:
            st.write("Not enough valid price data to display borough comparison.")
    else:
        st.markdown("##### Price by Borough (Box Plot)")
        st.write("Borough or Price information is not available for this chart.")

else:
    st.info("No data matches the current filter criteria to display core charts.")
st.markdown("---")


# --- NEW: Temporal Analysis Section ---
with st.expander("📅 Temporal Analysis (Based on Last Review Date)"):
    if not filtered_df.empty and 'review_year_month' in filtered_df.columns and 'price' in filtered_df.columns:
        temporal_data = filtered_df.dropna(subset=['review_year_month', 'price'])

        if not temporal_data.empty:
            st.markdown("##### Average Price Over Time")
            # Ensure price is numeric before grouping
            temporal_data['price'] = pd.to_numeric(temporal_data['price'], errors='coerce')
            avg_price_time = temporal_data.dropna(subset=['price']).groupby('review_year_month')['price'].mean().sort_index()

            if not avg_price_time.empty:
                fig, ax = plt.subplots(figsize=(12, 5))
                avg_price_time.index = avg_price_time.index.to_timestamp()
                ax.plot(avg_price_time.index, avg_price_time.values, marker='o', linestyle='-')
                plt.title("Average Price by Review Month-Year")
                plt.xlabel("Month-Year of Last Review")
                plt.ylabel("Average Price ($)")
                plt.xticks(rotation=45)
                plt.grid(True)
                st.pyplot(fig)
            else:
                st.write("Not enough data to plot average price over time.")

            st.markdown("##### Listing Activity (Reviews) Over Time")
            # Assuming 'id' exists and is unique for counting reviews or listings associated with reviews
            if 'id' in temporal_data.columns:
                listings_time = temporal_data.groupby('review_year_month')['id'].count().sort_index()
                if not listings_time.empty:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    listings_time.index = listings_time.index.to_timestamp()
                    ax.plot(listings_time.index, listings_time.values, marker='o', linestyle='-', color='green')
                    plt.title("Number of Reviews by Review Month-Year")
                    plt.xlabel("Month-Year of Last Review")
                    plt.ylabel("Number of Reviews Logged")
                    plt.xticks(rotation=45)
                    plt.grid(True)
                    st.pyplot(fig)
                else:
                    st.write("Not enough data to plot listing activity over time.")
            else:
                st.write("'id' column missing for listing activity plot.")
        else:
            st.write("No review date data available for temporal analysis with current filters.")
    else:
        st.info("Filter data or ensure 'last_review' and 'price' columns are present and processed for temporal analysis.")
st.markdown("---")


# --- NEW: Host-Centric Analysis Section ---
with st.expander("👤 Host-Centric Analysis"):
    if not filtered_df.empty and 'calculated_host_listings_count' in filtered_df.columns and 'host_id' in filtered_df.columns:
        st.markdown("##### Distribution of Listings per Host")
        # Ensure 'calculated_host_listings_count' is numeric
        filtered_df['calculated_host_listings_count'] = pd.to_numeric(filtered_df['calculated_host_listings_count'], errors='coerce')
        host_listings_counts = filtered_df.dropna(subset=['host_id', 'calculated_host_listings_count']).drop_duplicates(subset=['host_id'])['calculated_host_listings_count']
        if not host_listings_counts.empty:
            fig, ax = plt.subplots()
            sns.histplot(host_listings_counts, bins=min(30, host_listings_counts.nunique()), kde=False, ax=ax)
            ax.set_yscale('log')
            plt.title("Distribution of Number of Listings per Host")
            plt.xlabel("Number of Listings Held by a Host")
            plt.ylabel("Number of Hosts (Log Scale)")
            st.pyplot(fig)
        else:
            st.write("No data for host listings distribution.")


        st.markdown("##### Single vs. Multi-Listing Hosts (Price & Availability)")
        # Create 'host_type' after ensuring 'calculated_host_listings_count' is numeric and handled NaNs
        temp_df_host_analysis = filtered_df.dropna(subset=['calculated_host_listings_count', 'price', 'availability_365']).copy()
        temp_df_host_analysis['host_type'] = temp_df_host_analysis['calculated_host_listings_count'].apply(lambda x: 'Single-Listing' if x == 1 else 'Multi-Listing')

        if len(temp_df_host_analysis['host_type'].unique()) > 1:
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                numeric_prices_host = pd.to_numeric(temp_df_host_analysis['price'], errors='coerce').dropna()
                if not numeric_prices_host.empty:
                    fig, ax = plt.subplots()
                    sns.boxplot(data=temp_df_host_analysis, x='host_type', y='price', ax=ax)
                    plt.title("Price Comparison: Single vs. Multi-Listing Hosts")
                    ax.set_ylim(0, numeric_prices_host.quantile(0.95) if not numeric_prices_host.empty else 500)
                    st.pyplot(fig)
                else:
                    st.write("Not enough price data for host type comparison.")
            with col_h2:
                numeric_avail_host = pd.to_numeric(temp_df_host_analysis['availability_365'], errors='coerce').dropna()
                if not numeric_avail_host.empty:
                    fig, ax = plt.subplots()
                    sns.boxplot(data=temp_df_host_analysis, x='host_type', y='availability_365', ax=ax)
                    plt.title("Availability: Single vs. Multi-Listing Hosts")
                    st.pyplot(fig)
                else:
                    st.write("Not enough availability data for host type comparison.")

        else:
            st.write("Not enough diversity in host types for comparison with current filters.")
    else:
        st.info("Ensure 'calculated_host_listings_count', 'host_id', 'price', 'availability_365' columns are present for host analysis.")
st.markdown("---")


# --- NEW: Advanced Price & Availability Insights ---
with st.expander("🔑 Advanced Price & Availability Insights"):
    plot_df_sample = filtered_df.sample(min(1000, len(filtered_df))) if not filtered_df.empty else pd.DataFrame()

    if not plot_df_sample.empty and 'price' in plot_df_sample.columns and 'availability_365' in plot_df_sample.columns:
        plot_df_sample['price'] = pd.to_numeric(plot_df_sample['price'], errors='coerce')
        plot_df_sample['availability_365'] = pd.to_numeric(plot_df_sample['availability_365'], errors='coerce')
        scatter_data = plot_df_sample.dropna(subset=['price', 'availability_365'])

        if not scatter_data.empty:
            st.markdown("##### Price vs. Availability Scatter Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=scatter_data,
                            x='availability_365', y='price',
                            hue='neighbourhood_group' if current_filters['borough'] == 'All' and 'neighbourhood_group' in scatter_data.columns else None,
                            alpha=0.5, ax=ax)
            plt.title("Price vs. Availability (Days)")
            plt.xlabel("Availability in Next 365 Days")
            plt.ylabel("Price ($)")
            ax.set_ylim(0, scatter_data['price'].quantile(0.98) if not scatter_data.empty else 500)
            plt.grid(True)
            st.pyplot(fig)
        else:
            st.markdown("##### Price vs. Availability Scatter Plot")
            st.write("Not enough valid data for Price vs. Availability scatter plot.")


    if not filtered_df.empty and 'minimum_nights' in filtered_df.columns and 'price' in filtered_df.columns:
        min_nights_data = filtered_df.copy()
        min_nights_data['minimum_nights'] = pd.to_numeric(min_nights_data['minimum_nights'], errors='coerce')
        min_nights_data['price'] = pd.to_numeric(min_nights_data['price'], errors='coerce')
        min_nights_data.dropna(subset=['minimum_nights', 'price'], inplace=True)

        if not min_nights_data.empty and min_nights_data['minimum_nights'].max() > 0:
            st.markdown("##### Impact of Minimum Nights on Price")
            bins = [0, 1, 2, 3, 7, 14, 30, min_nights_data['minimum_nights'].max() + 1] # Ensure max is covered
            labels = ['1', '2', '3', '4-7', '8-14', '15-30', f"31+"]
            # Adjust bins and labels if max nights is small
            if min_nights_data['minimum_nights'].max() < 30:
                 bins = [b for b in bins if b <= min_nights_data['minimum_nights'].max() +1 ]
                 labels = labels[:len(bins)-1]


            if len(bins) > 1 : # Check if any bins can be formed
                min_nights_data['min_nights_bin'] = pd.cut(min_nights_data['minimum_nights'], bins=bins, labels=labels, right=True, include_lowest=True)
                if not min_nights_data['min_nights_bin'].dropna().empty:
                    fig, ax = plt.subplots(figsize=(10,6))
                    sns.boxplot(data=min_nights_data.dropna(subset=['min_nights_bin']), x='min_nights_bin', y='price', ax=ax, order=[l for l in labels if l in min_nights_data['min_nights_bin'].unique()])
                    plt.title("Price by Minimum Nights Category")
                    plt.xlabel("Minimum Nights Required")
                    plt.ylabel("Price ($)")
                    ax.set_ylim(0, min_nights_data['price'].quantile(0.95) if not min_nights_data.empty else 500)
                    st.pyplot(fig)
                else:
                    st.markdown("##### Impact of Minimum Nights on Price")
                    st.write("Could not bin minimum nights for analysis with current data.")
            else:
                st.markdown("##### Impact of Minimum Nights on Price")
                st.write("Not enough range in minimum nights to create bins.")

        else:
            st.markdown("##### Impact of Minimum Nights on Price")
            st.write("No minimum nights data to analyze or all minimum nights are zero.")
    else:
        st.info("Ensure 'price', 'availability_365', and 'minimum_nights' columns are present for these insights.")
st.markdown("---")

# --- NEW: Review-Based Insights ---
with st.expander("📝 Review-Based Insights"):
    plot_df_sample_reviews = filtered_df.sample(min(1000, len(filtered_df))) if not filtered_df.empty else pd.DataFrame()

    if not plot_df_sample_reviews.empty and 'reviews_per_month' in plot_df_sample_reviews.columns and 'price' in plot_df_sample_reviews.columns:
        plot_df_sample_reviews['reviews_per_month'] = pd.to_numeric(plot_df_sample_reviews['reviews_per_month'], errors='coerce')
        plot_df_sample_reviews['price'] = pd.to_numeric(plot_df_sample_reviews['price'], errors='coerce')
        review_price_data = plot_df_sample_reviews.dropna(subset=['reviews_per_month', 'price'])

        if not review_price_data.empty:
            st.markdown("##### Reviews per Month vs. Price")
            fig, ax = plt.subplots(figsize=(10,6))
            sns.scatterplot(data=review_price_data,
                            x='reviews_per_month', y='price',
                            hue='room_type' if 'room_type' in review_price_data.columns else None,
                            alpha=0.5, ax=ax)
            plt.title("Reviews per Month vs. Price")
            plt.xlabel("Reviews per Month")
            plt.ylabel("Price ($)")
            ax.set_ylim(0, review_price_data['price'].quantile(0.98) if not review_price_data['price'].empty else 500)
            ax.set_xlim(0, review_price_data['reviews_per_month'].quantile(0.98) if not review_price_data['reviews_per_month'].empty and review_price_data['reviews_per_month'].quantile(0.98) > 0 else (review_price_data['reviews_per_month'].max() if not review_price_data['reviews_per_month'].empty else 10))
            st.pyplot(fig)
        else:
            st.markdown("##### Reviews per Month vs. Price")
            st.write("Not enough valid data for Reviews/Month vs. Price plot.")


    if not plot_df_sample_reviews.empty and 'reviews_per_month' in plot_df_sample_reviews.columns and 'availability_365' in plot_df_sample_reviews.columns:
        plot_df_sample_reviews['availability_365'] = pd.to_numeric(plot_df_sample_reviews['availability_365'], errors='coerce')
        # reviews_per_month already converted or use original df's preprocessed one
        review_avail_data = plot_df_sample_reviews.dropna(subset=['reviews_per_month', 'availability_365'])

        if not review_avail_data.empty:
            st.markdown("##### Reviews per Month vs. Availability")
            fig, ax = plt.subplots(figsize=(10,6))
            sns.scatterplot(data=review_avail_data,
                            x='availability_365', y='reviews_per_month',
                            alpha=0.5, ax=ax)
            plt.title("Reviews per Month vs. Availability")
            plt.xlabel("Availability in Next 365 Days")
            plt.ylabel("Reviews per Month")
            ax.set_ylim(0, review_avail_data['reviews_per_month'].quantile(0.98) if not review_avail_data['reviews_per_month'].empty and review_avail_data['reviews_per_month'].quantile(0.98) > 0 else (review_avail_data['reviews_per_month'].max() if not review_avail_data['reviews_per_month'].empty else 10) )
            st.pyplot(fig)
        else:
            st.markdown("##### Reviews per Month vs. Availability")
            st.write("Not enough valid data for Reviews/Month vs. Availability plot.")

    else:
        st.info("Ensure 'reviews_per_month', 'price', 'availability_365' columns are present for these insights.")
st.markdown("---")

# … after your Review-Based Insights expander, insert:

with st.expander("🔬 Statistical Analysis & Tests"):
    if filtered_df.empty:
        st.info("No data available for statistical testing.")
    else:
        # 1) ANOVA: Price by Borough
        st.markdown("### 1. ANOVA: Price by Borough")
        grouped = [
            (name, grp["price"].dropna())
            for name, grp in filtered_df.groupby("neighbourhood_group")
            if grp["price"].dropna().size > 1
        ]
        if len(grouped) > 1:
            labels = [name for name, _ in grouped]
            means  = [g.mean()      for _, g in grouped]
            sems   = [g.sem()       for _, g in grouped]

            fig, ax = plt.subplots()
            ax.bar(
                labels,
                means,
                yerr=sems,
                capsize=5,
                color=sns.color_palette("Set2", len(labels))
            )
            ax.set_title("Mean Price by Borough (±SEM)")
            ax.set_ylabel("Price ($)")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

            # ANOVA test
            f_stat, p_val = stats.f_oneway(*[g for _, g in grouped])
            col1, col2 = st.columns(2)
            col1.metric("F-statistic", f"{f_stat:.2f}")
            col2.metric("p-value",     f"{p_val:.4f}")

            if p_val < 0.05:
                st.success("Significant differences between boroughs (p < 0.05)")
                st.markdown("**Post-hoc: Tukey HSD**")
                tukey = pg.pairwise_tukey(data=filtered_df, dv="price", between="neighbourhood_group")
                # Use backticks around column name with hyphen
                sig = tukey.query("`p-tukey` < 0.05")[["A", "B", "diff", "p-tukey"]].round(4)
                if not sig.empty:
                    st.dataframe(sig)
                else:
                    st.write("No borough-pair differences reached significance.")
            else:
                st.info("No significant differences between boroughs (p ≥ 0.05).")
        else:
            st.warning("Not enough boroughs with >1 listing for ANOVA.")

        st.markdown("---")

        # 2) Welch’s t-test: Entire home vs Private room
        st.markdown("### 2. T-test: Entire home vs Private room")
        df_t = filtered_df.dropna(subset=["price","room_type"])
        grp1 = df_t.query("room_type=='Entire home/apt'")["price"]
        grp2 = df_t.query("room_type=='Private room'")["price"]
        if len(grp1) > 1 and len(grp2) > 1:
            t_stat, p_val = stats.ttest_ind(grp1, grp2, equal_var=False)
            col1, col2, col3 = st.columns(3)
            col1.metric("t-statistic", f"{t_stat:.2f}")
            col2.metric("p-value",     f"{p_val:.4f}")
            col3.metric("Mean Δ",       f"${(grp1.mean() - grp2.mean()):.2f}")
            fig, ax = plt.subplots()
            sns.kdeplot(grp1, label="Entire home", ax=ax)
            sns.kdeplot(grp2, label="Private room", ax=ax)
            ax.legend()
            ax.set_title("Price Distribution: Entire vs Private")
            st.pyplot(fig)
        else:
            st.warning("Not enough data for t-test between room types.")

        st.markdown("---")

        # 3) Chi-square: Room Type vs Availability
        st.markdown("### 3. Chi-square: Room Type vs Availability")
        avail_cat = pd.cut(
            filtered_df["availability_365"], 
            bins=[-1, 0, 180, 365],
            labels=["Not Available","Semi Available","Fully Available"]
        )
        ct = pd.crosstab(filtered_df["room_type"], avail_cat)
        st.write("Contingency Table:")
        st.dataframe(ct)
        if ct.values.sum() > 0:
            chi2, p_val, dof, _ = stats.chi2_contingency(ct)
            col1, col2 = st.columns(2)
            col1.metric("χ² statistic", f"{chi2:.2f}")
            col2.metric("p-value",      f"{p_val:.4f}")
            if p_val < 0.05:
                st.success("Dependency detected (p < 0.05)")
            else:
                st.info("No dependency detected (p ≥ 0.05).")
        else:
            st.warning("Insufficient data for Chi-square test.")

        st.markdown("---")

        # 4) Correlation matrix + Pearson tests
        st.markdown("### 4. Correlation Analysis")
        num_cols = ["price", "minimum_nights", "number_of_reviews", "reviews_per_month", "availability_365"]
        corr = filtered_df[num_cols].corr().round(2)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

        # Top 3 correlations
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
                .stack()
                .sort_values(key=lambda s: s.abs(), ascending=False)
                .head(3)
        )
        st.markdown("**Top 3 correlations:**")
        for (v1, v2), val in pairs.items():
            r, p = stats.pearsonr(filtered_df[v1].dropna(), filtered_df[v2].dropna())
            st.write(f"- {v1} vs {v2}: r = {val:.2f}, p = {p:.3f}")

        st.markdown("---")

        # 5) Linear regression: Price ~ Availability
        st.markdown("### 5. Regression: Price vs Availability")
        reg_df = filtered_df.dropna(subset=["price","availability_365"])
        if len(reg_df) > 10:
            X = sm.add_constant(reg_df["availability_365"])
            model = sm.OLS(reg_df["price"], X).fit()
            col1, col2 = st.columns(2)
            col1.metric("R²",       f"{model.rsquared:.2f}")
            col2.metric("p-value", f"{model.pvalues['availability_365']:.4f}")
            fig, ax = plt.subplots()
            sns.regplot(
                x="availability_365", y="price",
                data=reg_df,
                scatter_kws={"alpha":0.3},
                line_kws={"color":"red"},
                ax=ax
            )
            ax.set_title("Price vs Availability with Fit")
            st.pyplot(fig)
        else:
            st.warning("Not enough data points for regression.")
st.markdown("---")

# --- Map View Section (MODIFIED for Heatmap) ---
st.subheader("🗺️ Listing Map & Heatmaps")
map_key_suffix = f"{current_filters['borough']}_{current_filters['room_type']}_{current_filters['price_range'][0]}_{current_filters['price_range'][1]}_{current_filters['min_avail']}_{current_filters['heatmap_option']}"

if st.session_state.cached_map is None or st.session_state.filters_changed:
    if not filtered_df.empty and 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
        # Ensure latitude and longitude are numeric and drop NaNs for map operations
        map_df = filtered_df.copy()
        map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
        map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
        map_df.dropna(subset=['latitude', 'longitude'], inplace=True)

        if not map_df.empty:
            sample_df_map = map_df.sample(min(1000, len(map_df)))

            map_center_lat = sample_df_map["latitude"].mean()
            map_center_lon = sample_df_map["longitude"].mean()
            current_zoom = 11 if current_filters["borough"] == "All" else 12

            m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=current_zoom, tiles="CartoDB positron")

            # --- Add Heatmap Layer IF Selected ---
            if current_filters["heatmap_option"] != "None" and not sample_df_map.empty:
                heat_data = []
                if current_filters["heatmap_option"] == "Price Heatmap" and 'price' in sample_df_map.columns:
                    sample_df_map['price'] = pd.to_numeric(sample_df_map['price'], errors='coerce')
                    price_heat_df = sample_df_map.dropna(subset=['latitude', 'longitude', 'price'])
                    if not price_heat_df.empty:
                        max_price_heat = price_heat_df['price'].quantile(0.9)
                        heat_data = [[row['latitude'], row['longitude'], min(row['price'], max_price_heat)]
                                     for index, row in price_heat_df.iterrows()]
                        if heat_data:
                            HeatMap(heat_data, radius=15, blur=20,
                                    # ******** CORRECTED GRADIENT KEYS TO STRINGS ********
                                    gradient={'0.1': 'blue', '0.5': 'lime', '1.0': 'red'}
                                    ).add_to(m)
                            folium.LayerControl(collapsed=False).add_to(m)
                        else:
                            st.caption("No valid data points for Price Heatmap after filtering for NaN prices.")
                    else:
                        st.caption("No price data available for Price Heatmap.")


                elif current_filters["heatmap_option"] == "Listing Density Heatmap":
                    # lat/lon already filtered for NaNs in sample_df_map
                    heat_data = [[row['latitude'], row['longitude']]
                                 for index, row in sample_df_map.iterrows()] # No further NaN check needed here for lat/lon
                    if heat_data: # Ensure heat_data is not empty
                        HeatMap(heat_data, radius=10, blur=15).add_to(m)
                        folium.LayerControl(collapsed=False).add_to(m)
                    else:
                        st.caption("No valid data points for Listing Density Heatmap.")


            # --- Add Listing Markers ---
            if current_filters["heatmap_option"] == "None": # Only show markers if no heatmap
                # For markers, ensure name and price are strings, handle missing data
                markers_df = sample_df_map.sample(min(200, len(sample_df_map))).copy()
                markers_df['name_str'] = markers_df['name'].astype(str).fillna('N/A')
                markers_df['price_str'] = markers_df['price'].astype(str).fillna('N/A')

                marker_cluster = folium.plugins.MarkerCluster().add_to(m)
                for _, row in markers_df.iterrows():
                    popup_text = f"{row['name_str']}<br>Price: ${row['price_str']}"
                    folium.Marker(
                        location=[row["latitude"], row["longitude"]],
                        popup=folium.Popup(popup_text, max_width=250),
                        tooltip=row['name_str']
                    ).add_to(marker_cluster)

            # Add borough boundary polygon
            if current_filters["borough"] != "All" and len(map_df) >= 3: # Use map_df which has lat/lon NaNs dropped
                unique_points_df = map_df[map_df['neighbourhood_group'] == current_filters["borough"]][['latitude', 'longitude']].drop_duplicates()
                if len(unique_points_df) >= 3:
                    try:
                        points = MultiPoint(unique_points_df.values)
                        if points.convex_hull.geom_type == 'Polygon':
                            polygon_shape = points.convex_hull
                            poly_coords = [[lat, lon] for lon, lat in polygon_shape.exterior.coords] # Folium expects [lat, lon]
                            folium.Polygon(locations=poly_coords, color='blue', weight=2, fill=True, fill_opacity=0.1, tooltip=f"{current_filters['borough']} Boundary").add_to(m)
                    except Exception as e:
                        st.warning(f"Could not generate borough boundary for {current_filters['borough']}: {e}")
                # else: # Optional: inform if not enough unique points for boundary
                #     st.caption(f"Not enough unique data points in {current_filters['borough']} to draw a boundary after filtering.")


            st.session_state.cached_map = m
            st.session_state.filters_changed = False
            st_folium(m, width=700, height=500, key=f"map_display_new_{map_key_suffix}", returned_objects=[])

        else: # if map_df is empty after dropping lat/lon NaNs
            st.info("No listings with valid geolocation data match the current filters for map display.")
            default_map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
            st.session_state.cached_map = default_map
            st.session_state.filters_changed = False
            st_folium(default_map, width=700, height=500, key=f"map_display_empty_geo_{map_key_suffix}", returned_objects=[])

    elif filtered_df.empty: # if original filtered_df was already empty
        st.info("No listings match the current filters to display on the map.")
        default_map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
        st.session_state.cached_map = default_map # Cache this default view
        st.session_state.filters_changed = False
        st_folium(default_map, width=700, height=500, key=f"map_display_empty_all_{map_key_suffix}", returned_objects=[])
    else: # Should not be reached if filtered_df has rows but no lat/lon columns, but as a fallback
        st.info("Location data ('latitude', 'longitude') is missing in the filtered data.")
        default_map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
        st.session_state.cached_map = default_map
        st.session_state.filters_changed = False
        st_folium(default_map, width=700, height=500, key=f"map_display_empty_cols_{map_key_suffix}", returned_objects=[])


elif st.session_state.cached_map:
    st_folium(st.session_state.cached_map, width=700, height=500, key=f"map_display_cached_{map_key_suffix}", returned_objects=[])
else:
    st.error("Map could not be loaded. Please try refreshing or changing filters.")
    default_map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
    st_folium(default_map, width=700, height=500, key="map_display_fallback_generic", returned_objects=[])
st.markdown("---")

# --- NEW: Interactive Data Table View ---
with st.expander("📄 View Filtered Data Table"):
    if not filtered_df.empty:
        # Display a sample, ensuring columns are mostly simple types for display
        display_df = filtered_df.copy()
        # Convert complex types like period to string for display if they cause issues
        if 'review_year_month' in display_df.columns:
            display_df['review_year_month'] = display_df['review_year_month'].astype(str)
        if 'last_review_date' in display_df.columns:
            display_df['last_review_date'] = display_df['last_review_date'].astype(str)

        st.dataframe(display_df.head(min(1000, len(display_df)))) # Show head for consistency
        st.caption(f"Showing the first {min(1000, len(display_df))} of {len(filtered_df):,} filtered listings.")
        # Provide download for the full filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Filtered Data as CSV",
            data=csv,
            file_name='filtered_airbnb_data.csv',
            mime='text/csv',
        )
    else:
        st.write("No data to display based on current filters.")
st.markdown("---")


# --- Agent using Gemini Section ---
st.subheader("🤖 Strategy Assistant")
st.markdown("""
Ask a strategic question based on the data, for example:
- *What pricing strategy would you recommend for new 'Private room' listings in Brooklyn during summer?*
- *How can hosts in Queens with low availability improve their occupancy rates?*
- *Suggest marketing points for listings in Manhattan with prices over $300.*
""")

if "gemini_response" not in st.session_state:
    st.session_state.gemini_response = ""

# IMPORTANT: Replace with your actual API key retrieval from st.secrets
# gemini_api_key = st.secrets.get("GEMINI_API_KEY") # Recommended
gemini_api_key = "AIzaSyCqZy8Ow0ZHgK3wAMK9ymwk5b82CLQyWuo" # User provided key
model = None

if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash") # Using a common current model
        prompt_disabled = False
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}. The Strategy Assistant may not work.")
        prompt_disabled = True
else:
    st.warning("GEMINI_API_KEY not found. The Strategy Assistant will be unavailable.")
    prompt_disabled = True

filter_summary = f"Current filters: Borough: {current_filters['borough']}, Room Type: {current_filters['room_type']}, Price: ${current_filters['price_range'][0]}-${current_filters['price_range'][1]}, Min. Availability: {current_filters['min_avail']} days. Heatmap: {current_filters['heatmap_option']}."
num_listings_summary = f"Number of listings matching these filters: {len(filtered_df):,}."

user_prompt = st.text_input("Enter your question for the AI assistant:", placeholder="e.g., How can we increase listings in Queens?", disabled=prompt_disabled)

if st.button("Get Suggestion", disabled=prompt_disabled or not user_prompt):
    if model and user_prompt:
        full_prompt = f"""
        You are an AI assistant for analyzing NYC Airbnb data.
        The user is looking at a dashboard with the following filters applied:
        {filter_summary}
        {num_listings_summary}

        User's question: "{user_prompt}"

        Please provide a concise, actionable strategic suggestion (around 3-5 bullet points or a short paragraph) based on this context.
        If the question is too broad or unrelated to Airbnb strategy with the given data, provide general advice or ask for clarification.
        Focus on business strategy for Airbnb hosts or for understanding the NYC market.
        """
        try:
            with st.spinner("🤖 Thinking..."):
                response = model.generate_content(full_prompt)
                # Check for safety ratings or blocked content if applicable to your GenAI model version
                if response.parts:
                    st.session_state.gemini_response = response.text
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                    st.session_state.gemini_response = f"Could not generate a response. Reason: {response.prompt_feedback.block_reason}"
                else: # Fallback if parts is empty but no explicit block
                    st.session_state.gemini_response = "Received an empty response from the AI. Please try rephrasing your question."

        except Exception as e:
            st.error(f"Error communicating with Gemini: {e}")
            st.session_state.gemini_response = "Sorry, I couldn't retrieve a suggestion at this time."
    elif not user_prompt:
        st.info("Please enter a question.")


if st.session_state.gemini_response:
    st.markdown("### 💡 Suggested Strategy")
    st.markdown(st.session_state.gemini_response)