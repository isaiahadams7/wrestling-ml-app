import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.title("🦼️ Wrestling Match Analyzer & ML Insights")

# Load data
df = pd.read_csv("expanded_wrestling_data.csv")
df['shot_type_encoded'] = LabelEncoder().fit_transform(df['shot_type'])
df['wrestler'] = df['wrestler'].astype(str)

# Sidebar: Match and wrestler filters
match_id_filter = st.sidebar.multiselect("Filter by Match ID(s):", df['match_id'].unique(), default=df['match_id'].unique())
wrestler_filter = st.sidebar.multiselect("Filter by Wrestler(s):", df['wrestler'].unique(), default=df['wrestler'].unique())

df_filtered = df[(df['match_id'].isin(match_id_filter)) & (df['wrestler'].isin(wrestler_filter))].copy()
df_shots = df_filtered[df_filtered['shot_attempted'] == 1].copy()

st.subheader("📋 Filtered Match Data")
st.dataframe(df_filtered)

# Shot Success Over Time - Binned
st.subheader("📈 Shot Success Rate Over Time (Binned)")
bins = np.arange(0, 390, 30)
df_shots['time_bin'] = pd.cut(df_shots['time_sec'], bins=bins, labels=[f"{b}-{b+30}" for b in bins[:-1]])
binned = df_shots.groupby(['time_bin', 'shot_type'])['shot_success'].mean().reset_index()

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.lineplot(data=binned, x='time_bin', y='shot_success', hue='shot_type', marker='o', ax=ax1)
ax1.set_title("Shot Success Rate Over Time (Binned)")
ax1.set_ylabel("Avg Shot Success Rate")
ax1.set_xlabel("Time Bin (seconds)")
ax1.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig1)

# Takedowns Given - Cumulative
st.subheader("📊 Cumulative Takedowns Given Over Time")
df_filtered = df_filtered.sort_values(['match_id', 'time_sec'])
df_filtered['cumulative_takedowns'] = df_filtered.groupby('match_id')['takedown_given'].cumsum()

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df_filtered, x='time_sec', y='cumulative_takedowns', hue='match_id', marker='o', ax=ax2)
ax2.set_title("Cumulative Takedowns Given Over Time")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Total Takedowns Given")
ax2.grid(True)
st.pyplot(fig2)

# Back Points Summary
st.subheader("🏅 Back Points Scored vs. Given by Match")
df_grouped = df_filtered.groupby('match_id')[['back_points_scored', 'back_points_given']].sum().reset_index()
df_melted = df_grouped.melt(id_vars='match_id', var_name='point_type', value_name='total')

fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.barplot(data=df_melted, x='match_id', y='total', hue='point_type', ax=ax3)
ax3.set_title("Back Points Scored vs. Given")
ax3.set_xlabel("Match ID")
ax3.set_ylabel("Total Points")
ax3.grid(True)
st.pyplot(fig3)

# ML Model: Predict Shot Success
st.subheader("🤖 Predicting Shot Success with ML")
features = ['time_sec', 'shot_type_encoded', 'score_diff', 'fatigue_proxy']
target = 'shot_success'
X = df_shots[features]
y = df_shots[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
st.text("📄 Classification Report:")
st.json(report)

# --- User Input Form ---
st.subheader("👥 Try Your Own Shot Prediction")
with st.form("user_input_form"):
    input_time = st.slider("Time into Match (seconds)", 0, 360, 120)
    input_score_diff = st.slider("Score Difference (Your score - Opponent)", -10, 10, 0)
    input_fatigue = st.slider("Fatigue Proxy (0 = fresh, 30 = exhausted)", 0.0, 30.0, 10.0)
    input_shot_type = st.selectbox("Shot Type", df['shot_type'].unique())
    input_shot_encoded = LabelEncoder().fit(df['shot_type']).transform([input_shot_type])[0]

    submitted = st.form_submit_button("Predict Shot Success")
    if submitted:
        user_features = pd.DataFrame([[input_time, input_shot_encoded, input_score_diff, input_fatigue]],
                                     columns=features)
        prediction = model.predict(user_features)[0]
        probability = model.predict_proba(user_features)[0][1]
        st.success(f"Predicted Success: {'Yes' if prediction else 'No'} (Confidence: {probability:.2f})")
