import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import altair as alt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(file):
    return pd.read_csv(file)

def clean_data(df):
    data_clean = df.copy()
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    data_clean = data_clean[(data_clean['wl_up'] >= 100) & (data_clean['wl_up'] <= 450)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]
    return data_clean

def create_time_features(data_clean):
    if not pd.api.types.is_datetime64_any_dtype(data_clean['datetime']):
        data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')

    data_clean['year'] = data_clean['datetime'].dt.year
    data_clean['month'] = data_clean['datetime'].dt.month
    data_clean['day'] = data_clean['datetime'].dt.day
    data_clean['hour'] = data_clean['datetime'].dt.hour
    data_clean['minute'] = data_clean['datetime'].dt.minute
    data_clean['day_of_week'] = data_clean['datetime'].dt.dayofweek
    data_clean['day_of_year'] = data_clean['datetime'].dt.dayofyear
    data_clean['week_of_year'] = data_clean['datetime'].dt.isocalendar().week
    data_clean['days_in_month'] = data_clean['datetime'].dt.days_in_month

    return data_clean

def prepare_features(data_clean):
    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'day_of_week', 'day_of_year', 'week_of_year',
        'days_in_month'
    ]
    X = data_clean[feature_cols]
    y = data_clean['wl_up']
    return X, y

def train_model(X_train, y_train):
    param_distributions = {
        'n_estimators': [100, 200, 500, 1000],  # ขยายช่วงของ n_estimators
        'max_depth': [None, 10, 20, 50],  # ปรับ max_depth ให้กว้างขึ้น
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],  # ทดสอบการใช้ข้อมูลทั้งหมดในแต่ละต้นไม้หรือบางส่วน
        'max_samples': [None, 0.8, 0.9]  # กำหนด max_samples เพื่อสุ่มตัวอย่างในแต่ละต้นไม้
    }

    rf = RandomForestRegressor(random_state=42)

    n_splits = min(3, len(X_train) // 2)
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=20, cv=n_splits, n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    return random_search.best_estimator_

def generate_missing_dates(data):
    full_date_range = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='15T')
    all_dates = pd.DataFrame(full_date_range, columns=['datetime'])
    data_with_all_dates = pd.merge(all_dates, data, on='datetime', how='left')
    return data_with_all_dates

def fill_code_column(data):
    data['code'] = data['code'].fillna(method='ffill').fillna(method='bfill')
    return data

def smooth_filled_values(data_with_all_dates, window_size=3):
    """Apply smoothing technique to reduce the sudden jumps in the filled values."""
    data_with_all_dates['wl_up'] = data_with_all_dates['wl_up'].interpolate(method='linear')
    data_with_all_dates['wl_up'] = data_with_all_dates['wl_up'].rolling(window=window_size, min_periods=1).mean()
    return data_with_all_dates

def handle_missing_values_by_week(data_clean, start_date, end_date):
    feature_cols = ['year', 'month', 'day', 'hour', 'minute',
                    'day_of_week', 'day_of_year', 'week_of_year', 'days_in_month']

    data = data_clean.copy()
    
    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data based on the datetime range
    data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]

    # Generate all missing dates within the selected range
    data_with_all_dates = generate_missing_dates(data)
    data_with_all_dates.index = pd.to_datetime(data_with_all_dates['datetime'])
    data_missing = data_with_all_dates[data_with_all_dates['wl_up'].isnull()]
    data_not_missing = data_with_all_dates.dropna(subset=['wl_up'])

    # เพิ่มคอลัมน์ timestamp และ wl_forecast
    data_with_all_dates['timestamp'] = pd.NaT  # กำหนดค่าเริ่มต้นเป็น NaT (Not a Timestamp)
    data_with_all_dates['wl_forecast'] = np.nan  # สร้างคอลัมน์สำหรับเก็บค่าที่ถูกเติม

    if len(data_missing) == 0:
        st.write("No missing values to predict.")
        return data_with_all_dates

    # Train initial model with all available data
    X_train, y_train = prepare_features(data_not_missing)
    model = train_model(X_train, y_train)

    # Separate weeks with fewer and more missing rows
    weeks_with_fewer_missing = []
    weeks_with_more_missing = []

    weeks_with_missing = data_missing['week_of_year'].unique()

    for week in weeks_with_missing:
        group = data_missing[data_missing['week_of_year'] == week]
        missing_count = group['wl_up'].isnull().sum()
        if missing_count <= 288:
            weeks_with_fewer_missing.append(week)
        else:
            weeks_with_more_missing.append(week)

    # Handle weeks with fewer than 288 missing rows by predicting missing values row-by-row
    for week in weeks_with_fewer_missing:
        group = data_missing[data_missing['week_of_year'] == week]
        week_data = data_not_missing[data_not_missing['week_of_year'] == week]

        if len(week_data) > 0:
            X_train_week, y_train_week = prepare_features(week_data)
            model_week = train_model(X_train_week, y_train_week)

            for idx, row in group.iterrows():
                X_missing = row[feature_cols].values.reshape(1, -1)
                predicted_value = model_week.predict(X_missing)
                
                # บันทึกค่าที่เติมในคอลัมน์ wl_forecast และ timestamp
                data_with_all_dates.loc[idx, 'wl_forecast'] = predicted_value
                data_with_all_dates.loc[idx, 'timestamp'] = pd.Timestamp.now()

    # Update data_not_missing after filling values
    data_not_missing = data_with_all_dates.dropna(subset=['wl_up'])

    # Handle weeks with more than 288 missing rows using data from adjacent weeks
    for week in weeks_with_more_missing:
        group = data_missing[data_missing['week_of_year'] == week]
        prev_week = week - 1 if week > min(weeks_with_missing) else week
        next_week = week + 1 if week < max(weeks_with_missing) else week

        prev_data = data_not_missing[data_not_missing['week_of_year'] == prev_week]
        next_data = data_not_missing[data_not_missing['week_of_year'] == next_week]
        previous_month_data = data_not_missing[data_not_missing['month'] == group['month'].iloc[0] - 1]

        combined_data = pd.concat([prev_data, next_data, previous_month_data])

        if data_missing[data_missing['week_of_year'] == next_week]['wl_up'].isnull().sum() > 288:
            current_month = group['month'].iloc[0]
            non_missing_month_data = data_clean[(data_clean['month'] == current_month) & (~data_clean['wl_up'].isnull())]

            X_train_month, y_train_month = prepare_features(non_missing_month_data)
            model_month = train_model(X_train_month, y_train_month)

            for idx, row in group.iterrows():
                X_missing = row[feature_cols].values.reshape(1, -1)
                predicted_value = model_month.predict(X_missing)

                # บันทึกค่าที่เติมในคอลัมน์ wl_forecast และ timestamp
                data_with_all_dates.loc[idx, 'wl_forecast'] = predicted_value
                data_with_all_dates.loc[idx, 'timestamp'] = pd.Timestamp.now()
        else:
            combined_train_X, combined_train_y = prepare_features(combined_data)
            model_combined = train_model(combined_train_X, combined_train_y)

            for idx, row in group.iterrows():
                X_missing = row[feature_cols].values.reshape(1, -1)
                predicted_value = model_combined.predict(X_missing)

                # บันทึกค่าที่เติมในคอลัมน์ wl_forecast และ timestamp
                data_with_all_dates.loc[idx, 'wl_forecast'] = predicted_value
                data_with_all_dates.loc[idx, 'timestamp'] = pd.Timestamp.now()

    # สร้างคอลัมน์ wl_up2 ที่รวมข้อมูลเดิมกับค่าที่เติม
    data_with_all_dates['wl_up2'] = data_with_all_dates['wl_up'].combine_first(data_with_all_dates['wl_forecast'])

    data_with_all_dates.reset_index(drop=True, inplace=True)
    return data_with_all_dates

def delete_data_by_date_range(data, delete_start_date, delete_end_date):
    # Convert delete_start_date and delete_end_date to datetime
    delete_start_date = pd.to_datetime(delete_start_date)
    delete_end_date = pd.to_datetime(delete_end_date)

    # ตรวจสอบว่าช่วงวันที่ต้องการลบข้อมูลอยู่ในช่วงของ data หรือไม่
    data_to_delete = data[(data['datetime'] >= delete_start_date) & (data['datetime'] <= delete_end_date)]

    # เพิ่มการตรวจสอบว่าถ้าจำนวนข้อมูลที่ถูกลบมีมากเกินไป
    if len(data_to_delete) == 0:
        st.warning(f"ไม่พบข้อมูลระหว่าง {delete_start_date} และ {delete_end_date}.")
    elif len(data_to_delete) > (0.3 * len(data)):  # ตรวจสอบว่าถ้าลบเกิน 30% ของข้อมูล
        st.warning("คำเตือน: มีข้อมูลมากเกินไปที่จะลบ การดำเนินการลบถูกยกเลิก")
    else:
        # ลบข้อมูลโดยตั้งค่า wl_up เป็น NaN
        data.loc[data_to_delete.index, 'wl_up'] = np.nan

    return data

def calculate_accuracy_metrics(original, filled):
    # ผสานข้อมูลโดยใช้ datetime เป็นพื้นฐาน โดยจะใช้ wl_up จาก original และ wl_up2 จาก filled
    merged_data = pd.merge(original[['datetime', 'wl_up']], filled[['datetime', 'wl_up2']], on='datetime')
    
    # คำนวณค่าความแม่นยำจาก wl_up ของ original และ wl_up2 ของ filled
    mse = mean_squared_error(merged_data['wl_up'], merged_data['wl_up2'])
    mae = mean_absolute_error(merged_data['wl_up'], merged_data['wl_up2'])
    r2 = r2_score(merged_data['wl_up'], merged_data['wl_up2'])

    # แสดงค่าความแม่นยำบนหน้าจอ
    st.header("ผลค่าความแม่นยำ", divider='gray')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")

    with col2:
        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")

    with col3:
        st.metric(label="R-squared (R²)", value=f"{r2:.4f}")

def plot_results(data_before, data_filled, data_deleted):
    data_before_filled = pd.DataFrame({
        'วันที่': data_before['datetime'],
        'ข้อมูลเดิม': data_before['wl_up']
    })

    data_after_filled = pd.DataFrame({
        'วันที่': data_filled['datetime'],
        'ข้อมูลหลังเติมค่า': data_filled['wl_up2']
    })

    data_after_deleted = pd.DataFrame({
        'วันที่': data_deleted['datetime'],
        'ข้อมูลหลังลบ': data_deleted['wl_up']
    })

    combined_data = pd.merge(data_before_filled, data_after_filled, on='วันที่', how='outer')
    combined_data = pd.merge(combined_data, data_after_deleted, on='วันที่', how='outer')

    min_y = combined_data[['ข้อมูลเดิม', 'ข้อมูลหลังเติมค่า', 'ข้อมูลหลังลบ']].min().min()
    max_y = combined_data[['ข้อมูลเดิม', 'ข้อมูลหลังเติมค่า', 'ข้อมูลหลังลบ']].max().max()

    chart = alt.Chart(combined_data).transform_fold(
        ['ข้อมูลเดิม', 'ข้อมูลหลังเติมค่า', 'ข้อมูลหลังลบ'],
        as_=['ข้อมูล', 'ระดับน้ำ']
    ).mark_line().encode(
        x='วันที่:T',
        y=alt.Y('ระดับน้ำ:Q', scale=alt.Scale(domain=[min_y, max_y])),
        color=alt.Color('ข้อมูล:N', scale=alt.Scale(scheme='reds'), legend=alt.Legend(orient='right', title='ข้อมูล'))
    ).properties(
        height=400
    ).interactive()

    st.header("ข้อมูลหลังจากการเติมค่าที่หายไป", divider='gray')
    st.altair_chart(chart, use_container_width=True)

    st.header("ตารางแสดงข้อมูลหลังเติมค่า", divider='gray')
    data_filled_selected = data_filled[['code', 'datetime', 'wl_up', 'wl_forecast', 'rf_15m', 'timestamp']]
    st.dataframe(data_filled_selected, use_container_width=True)

    # เรียกฟังก์ชันคำนวณค่าความแม่นยำ
    calculate_accuracy_metrics(data_before, data_filled)

def plot_data_preview(data1, data2):
    data_pre1 = pd.DataFrame({
        'วันที่': data1['datetime'],
        'สถานีที่ต้องการเติมค่า': data1['wl_up']
    })

    data_pre2 = pd.DataFrame({
        'วันที่': data2['datetime'],
        'สถานีก่อนหน้า': data2['wl_up']
    })

    combined_data_pre = pd.merge(data_pre1, data_pre2, on='วันที่', how='outer')
    min_y = combined_data_pre[['สถานีที่ต้องการเติมค่า', 'สถานีก่อนหน้า']].min().min()
    max_y = combined_data_pre[['สถานีที่ต้องการเติมค่า', 'สถานีก่อนหน้า']].max().max()

    chart = alt.Chart(combined_data_pre).transform_fold(
        ['สถานีที่ต้องการเติมค่า', 'สถานีก่อนหน้า'],
        as_=['ข้อมูล', 'ระดับน้ำ']
    ).mark_line().encode(
        x='วันที่:T',
        y=alt.Y('ระดับน้ำ:Q', scale=alt.Scale(domain=[min_y, max_y])),
        color=alt.Color('ข้อมูล:N', scale=alt.Scale(scheme='reds'), legend=alt.Legend(orient='right', title='ข้อมูล'))  # ใช้พาเลต magma และปรับตำแหน่ง Legend ไปทางขวา
    ).properties(
        height=400,
        title='ข้อมูลสถานี'
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# Streamlit UI
st.set_page_config(
    page_title="RandomForest",
    page_icon="🌲",
    layout="wide"
)
'''
# การจัดการข้อมูลระดับน้ำด้วย Random Forest
แอป Streamlit สำหรับจัดการข้อมูลระดับน้ำ โดยใช้โมเดล Random Forest เพื่อเติมค่าที่ขาดหายไป 
ข้อมูลถูกประมวลผลและแสดงผลผ่านกราฟและการวัดค่าความแม่นยำ ผู้ใช้สามารถเลือกอัปโหลดไฟล์, 
กำหนดช่วงเวลาลบข้อมูล, และดูผลลัพธ์ของการเติมค่าได้
'''
st.markdown("---")

# Sidebar: Upload files and choose date ranges
with st.sidebar:
    st.header("อัปโหลดไฟล์ CSV")
    
    with st.sidebar.expander("อัปโหลดข้อมูลสถานีวัดระดับน้ำ", expanded=False):
        uploaded_file = st.file_uploader("สถานีที่ต้องการเติมค่า", type="csv", key="uploader1")
        uploaded_file2 = st.file_uploader("สถานีที่ใช้ฝึกโมเดล", type="csv", key="uploader2")

    # เลือกช่วงวันที่ใน sidebar
    st.header("เลือกช่วงที่ต้องการข้อมูล")
    start_date = st.date_input("วันที่เริ่มต้น", value=pd.to_datetime("2024-05-01"))
    end_date = st.date_input("วันที่สิ้นสุด", value=pd.to_datetime("2024-05-31"))
    
    # เพิ่มตัวเลือกว่าจะลบข้อมูลหรือไม่
    delete_data_option = st.checkbox("ต้องการเลือกลบข้อมูล", value=False)

    if delete_data_option:
        # แสดงช่องกรอกข้อมูลสำหรับการลบข้อมูลเมื่อผู้ใช้ติ๊กเลือก
        st.header("เลือกช่วงที่ต้องการลบข้อมูล")
        delete_start_date = st.date_input("กำหนดเริ่มต้นลบข้อมูล", value=start_date, key='delete_start')
        delete_start_time = st.time_input("เวลาเริ่มต้น", value=pd.Timestamp("00:00:00").time(), key='delete_start_time')
        delete_end_date = st.date_input("กำหนดสิ้นสุดลบข้อมูล", value=end_date, key='delete_end')
        delete_end_time = st.time_input("เวลาสิ้นสุด", value=pd.Timestamp("23:45:00").time(), key='delete_end_time')

    process_button = st.button("ประมวลผล")

# Main content: Display results after file uploads and date selection
if uploaded_file and uploaded_file2:
    df = load_data(uploaded_file)
    df_pre = clean_data(df)

    df2 = load_data(uploaded_file2)
    df2_pre = clean_data(df2)

    plot_data_preview(df_pre, df2_pre)

    if process_button:
        processing_placeholder = st.empty()
        processing_placeholder.text("กำลังประมวลผลข้อมูล...")

        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

        # ปรับค่า end_date เฉพาะถ้าเลือกช่วงเวลาแล้ว
        end_date = end_date + pd.DateOffset(days=1)

        # กรองข้อมูลตามช่วงวันที่เลือก
        df_filtered = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))]

        # Clean data
        df_clean = clean_data(df_filtered)

        # ตรวจสอบว่าผู้ใช้เลือกที่จะลบข้อมูลหรือไม่
        if delete_data_option:
            delete_start_datetime = pd.to_datetime(f"{delete_start_date} {delete_start_time}")
            delete_end_datetime = pd.to_datetime(f"{delete_end_date} {delete_end_time}")
            df_deleted = delete_data_by_date_range(df_clean, delete_start_datetime, delete_end_datetime)
        else:
            df_deleted = df_clean.copy()  # ถ้าไม่เลือกลบก็ใช้ข้อมูลเดิมแทน

        # Generate all dates
        df_clean = generate_missing_dates(df_clean)

        # Fill NaN values in 'code' column
        df_clean = fill_code_column(df_clean)

        # Create time features
        df_clean = create_time_features(df_clean)

        # เก็บข้อมูลก่อนการสุ่มลบ
        df_before_random_deletion = df_filtered.copy()

        # Handle missing values by week
        df_handled = handle_missing_values_by_week(df_clean, start_date, end_date)

        # Remove the processing message after the processing is complete
        processing_placeholder.empty()

        # Plot the results using Streamlit's line chart
        plot_results(df_before_random_deletion, df_handled, df_deleted)
    st.markdown("---")