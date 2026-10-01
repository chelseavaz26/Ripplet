# Section (b): Trending Topics Detection (dashboard/app.py: lines 538–568)

## Overview
This code section in `dashboard/app.py` handles the setup, batch processing, and execution of **Section (b): Trending Topics Detection** using the `detect_trends` analytic module:

1. **Section Header & Metadata**:
   - Displays the section title (`🔥 (b) Trending Topics (detect_trends)`) and caption explaining the breakout threshold (keywords/hashtags spiking >= 2.0x above rolling baseline).

2. **Daily Batch Generator (`generate_window_batches`)**:
   - Groups the unified time-window DataFrame (`df_window`) by day (`freq="1D"`) on its timestamp.
   - Yields non-empty daily slices as an iterator, streaming batches derived strictly from the shared window.

3. **Trend Detection (`detect_trends`)**:
   - Runs `detect_trends()` inside a user feedback spinner (`st.spinner`).
   - Evaluates terms using a 7-day rolling window (`window_size=7`), a 2.0x spike ratio threshold (`threshold=2.0`), and a minimum count cutoff (`min_frequency=3`).

4. **Dashboard Layout & Trend Filtering**:
   - Allocates a two-column Streamlit layout (`col_t1`, `col_t2` with a 3:2 ratio) for displaying top trending terms alongside the trend feed.
   - Initializes `trending_terms_set` used to cross-reference trends across the dashboard (such as in the Influencer Spotlight).
   - Filters `trends_df` to extract active breakout trends (`is_trending == True`).
