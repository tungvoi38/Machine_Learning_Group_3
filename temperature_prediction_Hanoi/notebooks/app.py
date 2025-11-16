import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import plotly.express as px
import gradio as gr

# PHẦN A: BACKEND - MODEL
print("--- KHỞI ĐỘNG: Đang huấn luyện mô hình ---")

# --- 1. Đọc dữ liệu ---
try:
    read_dir = '../data/processed/'
    train_data = pd.read_excel(read_dir + 'train_data.xlsx')
    val_data = pd.read_excel(read_dir + 'val_data.xlsx')
    test_data = pd.read_excel(read_dir + 'test_data.xlsx')
except FileNotFoundError:
    print("LỖI: Không tìm thấy file trong '../data/processed/'.")
    exit()

# --- 2. Hàm tạo feature và target ---
def create_features_and_split(data):
    df = data.copy()
    
    # PET
    temp_diff = (df['tempmax'] - df['tempmin']).clip(lower=0)
    df['PET'] = (0.0023 * df['solarenergy'] * 0.408 * 
                 np.sqrt(temp_diff) * (df['temp'] + 17.8))
    
    # Feature mới
    df['daylight_duration_hours'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600
    df['wind_U'] = df['windspeed'] * np.sin(2 * np.pi * df['winddir'] / 360)
    df['wind_V'] = df['windspeed'] * np.cos(2 * np.pi * df['winddir'] / 360)
    df['pressure_daily_change'] = df['sealevelpressure'].diff(3)
    df['solar_cloud_interaction'] = df['solarradiation'] * (1 - (df['cloudcover'] / 100))

    if 'datetime' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)

    # --- Rolling features ---
    roll_cols = ['dew', 'humidity', 'precip', 'precipcover', 'windgust',
                 'windspeed', 'sealevelpressure', 'pressure_daily_change', 
                 'cloudcover', 'solarradiation', 'solarenergy', 
                 'solar_cloud_interaction','PET', 'daylight_duration_hours', 
                 'wind_U', 'wind_V']
    windows = [7,28,56,84]

    all_features_list = []

    for col in roll_cols:
        if col not in df.columns: continue
        col_upper = col.upper()
        for w in windows:
            mean_series = df[col].shift(1).rolling(window=w).mean().rename(f"{w}D_AVG_{col_upper}")
            var_series = df[col].shift(1).rolling(window=w).var().rename(f"{w}D_VAR_{col_upper}")
            all_features_list.extend([mean_series, var_series])

    # --- Các feature gốc ---
    cols = ['month_cos','month_sin','humidity','dew','precip','sealevelpressure','solar_cloud_interaction',
            'precipcover','solarradiation','pressure_daily_change','PET','daylight_duration_hours',
            'windspeed','winddir','solarenergy','windgust','cloudcover',
            'conditions_Clear','conditions_Overcast','conditions_Partially cloudy',
            'conditions_Rain','conditions_Rain, Overcast','conditions_Rain, Partially cloudy']
    for col in cols:
        if col in df.columns:
            all_features_list.append(df[col])

    # --- Polynomial / sqrt / log1p features ---
    poly_candidates = ['humidity','dew','precip','precipcover','windspeed','windgust',
                       'solarenergy','solarradiation','cloudcover','PET',
                       'daylight_duration_hours','sealevelpressure']
    for col in poly_candidates:
        if col not in df.columns: continue
        base = df[col]
        all_features_list.extend([
            (base ** 2).rename(f"{col.upper()}_SQ"),
            (base ** 3).rename(f"{col.upper()}_CUBE"),
            np.sqrt(base.clip(lower=0)).rename(f"{col.upper()}_SQRT"),
            np.log1p(base.clip(lower=0)).rename(f"{col.upper()}_LOG1P")
        ])

    features_df = pd.concat(all_features_list, axis=1)

    # --- Target 5 bước ---
    target_data = {f'y_temp_{i}': df['temp'].shift(-i) for i in range(1,6)}
    y = pd.DataFrame(target_data, index=df.index)

    # --- Gộp, dropna ---
    full_df = pd.concat([features_df, y], axis=1).dropna()
    target_cols = list(target_data.keys())
    X = full_df.drop(columns=target_cols)
    y = full_df[target_cols]

    return X, y

# --- 3. Tách train/val/test ---
X_train, y_train = create_features_and_split(train_data)
X_val, y_val = create_features_and_split(val_data)
X_test, y_test = create_features_and_split(test_data)

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# --- 4. Huấn luyện 5 RandomForest ---
models = {}
n_features = X_train.shape[1]
target_cols = list(y_train.columns)

for target_col in target_cols:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features=0.3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train[target_col])
    models[target_col] = model

print("--- HUẤN LUYỆN HOÀN TẤT ---")

# PHẦN B: FRONTEND - GRADIO
def get_weather_icon(temp):
    if temp > 32:
        return "☀️"
    elif temp > 25:
        return "🌤️"
    elif temp > 18:
        return "☁️"
    else:
        return "❄️"

# --- 2. Hàm dự đoán 5 ngày ---
def create_forecast():
    """
    Dự đoán 5 ngày tiếp theo dựa trên mô hình đã huấn luyện và X_test gần nhất
    """
    future_predictions = []

    # Lấy dữ liệu cuối cùng từ X_test để làm input
    last_row = X_test.iloc[-1]
    last_known_features = pd.DataFrame([last_row], columns=X_test.columns)
    last_known_date = test_data['datetime'].max()

    for i, target_col in enumerate(target_cols):
        model_i = models[target_col]
        # Dự đoán từng bước (trong RandomForest, mỗi target riêng)
        pred = model_i.predict(last_known_features)[0]
        future_predictions.append(pred)

    # --- 3. Tạo DataFrame kết quả ---
    future_dates = [last_known_date + timedelta(days=i+1) for i in range(5)]
    future_df = pd.DataFrame({
        'datetime': future_dates,
        'predicted_temperature': future_predictions
    })

    # --- 4. Tạo biểu đồ Plotly ---
    fig = px.area(
        future_df,
        x='datetime',
        y='predicted_temperature',
        title="Biểu đồ dự đoán 5 ngày tới",
        markers=True,
        labels={'datetime': 'Ngày', 'predicted_temperature': 'Nhiệt độ (°C)'},
        color_discrete_sequence=['#0056b3']  # màu xanh nước biển
    )
    fig.update_traces(
        text=future_df['predicted_temperature'].apply(lambda x: f'{x:.1f}°'),
        textposition='top center',
        hovertemplate='<b>Ngày</b>: %{x|%d-%m-%Y}<br><b>Nhiệt độ</b>: %{y:.1f}°C<extra></extra>'
    )
    fig.update_layout(
        yaxis_range=[future_df['predicted_temperature'].min()-2, future_df['predicted_temperature'].max()+2],
        title_x=0.5,
        xaxis_title=None,
        yaxis_title="Nhiệt độ (°C)",
        plot_bgcolor='#fcfcfc',
        paper_bgcolor='white',
        xaxis=dict(gridcolor='#eee'),
        yaxis=dict(gridcolor='#eee')
    )

    # --- 5. Tạo thẻ HTML dự báo ---
    html_output = "<div style='display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px;'>"
    DAYS_VN = ["Thứ Hai","Thứ Ba","Thứ Tư","Thứ Năm","Thứ Sáu","Thứ Bảy","Chủ Nhật"]

    for i in range(5):
        date = future_df.iloc[i]['datetime']
        temp = future_df.iloc[i]['predicted_temperature']
        day_of_week = DAYS_VN[date.weekday()]
        day_str = date.strftime('%d-%m')
        icon = get_weather_icon(temp)

        html_output += f"""
        <div style='border: 1px solid #ddd; border-radius: 12px; padding: 15px; min-width: 120px; 
                    text-align: center; background-color: #f9f9f9; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05);'>
          <h3 style='margin: 0; color: #0056b3;'>{day_of_week}</h3>
          <p style='font-size: 1.1em; color: #555; margin: 5px 0;'>{day_str}</p>
          <p style='font-size: 2.5em; margin: 10px 0;'>{icon}</p>
          <p style='font-size: 2.2em; font-weight: bold; color: #0056b3; margin: 5px 0;'>{temp:.1f}°C</p>
        </div>
        """

    html_output += "</div>"

    return fig, html_output

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange")) as iface:
    
    gr.Markdown("<h1 style='text-align:center;color:#0056b3;'>🌦️ Ứng dụng Dự báo Nhiệt độ Hà Nội</h1>")
    gr.Markdown("---")

    # 📅 DAILY FORECAST (TRÊN)
    gr.Markdown("## 📅 Dự báo 5 ngày tới (Daily)")

    daily_cards = gr.HTML()
    daily_plot = gr.Plot()

    iface.load(
        fn=create_forecast,
        inputs=None,
        outputs=[daily_plot, daily_cards]
    )

    gr.Markdown("---")

    # ⏰ HOURLY FORECAST (DƯỚI)
    gr.Markdown("## ⏰ Dự báo theo giờ cho 5 ngày tới (Hourly)")

    hourly_cards = gr.HTML()
    hourly_plot = gr.Plot()

    iface.load(
        fn=create_forecast,
        inputs=None,
        outputs=[hourly_plot, hourly_cards]   # DÙNG CHUNG HÀM
    )

    gr.Markdown("---")
    gr.Markdown("© 2025 – Nhóm Machine Learning Hà Nội")

print("Đang chạy giao diện... http://127.0.0.1:7860")
iface.launch()