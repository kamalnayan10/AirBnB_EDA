# NYC Airbnb Strategic Insights Dashboard

An **interactive Streamlit application** for exploring, analyzing, and deriving strategic insights from the New York City Airbnb dataset. This dashboard combines **data visualizations**, **statistical tests**, **geospatial mapping**, and an **AI-powered strategy assistant** to help Airbnb stakeholders make data-driven decisions about pricing, availability, and marketing campaigns.

---

## 🚀 Features

* **Interactive Filters**: Filter listings by borough, room type, price range, availability, and heatmap layer.
* **Core Visualizations**:

  * Price distribution (histogram + KDE)
  * Room type breakdown (bar chart)
  * Borough vs. price (boxplot)
* **Temporal Analysis**:

  * Average price over time
  * Review activity trends
* **Host-Centric Analysis**:

  * Listings per host distribution
  * Price & availability comparison: single vs multi-listing hosts
* **Advanced Insights**:

  * Price vs. availability scatterplots
  * Minimum nights impact on price
  * Reviews per month vs price and availability
* **Statistical Analysis & Tests**:

  * One-way ANOVA with post-hoc Tukey HSD
  * Welch’s t-test (entire home vs private room)
  * Chi-square test (room type vs availability)
  * Correlation matrix & Pearson tests
  * Linear regression (price \~ availability)
* **Geospatial Mapping**:

  * Interactive Folium map with clustering
  * Optional heatmaps (price density or listing density)
  * Highlight selected borough boundary with convex hull
* **AI Strategy Assistant**:

  * Powered by Google Gemini API
  * Suggests marketing campaigns, host programs, and pricing strategies based on current filters

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
