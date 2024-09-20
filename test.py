import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title='การพยากรณ์ด้วย RandomForest', page_icon=':ocean:')

# ชื่อของแอป
st.title("การจัดการค่าระดับน้ำและการพยากรณ์ด้วย RandomForest")

# ฟังก์ชันสำหรับการแสดงกราฟข้อมูลทั้งไฟล์ (ตัวอย่าง)
def plot_full_data(data):
    fig = px.line(data, x=data.index, y='wl_up', title='Water Level Over Time (Full Data)', labels={'x': 'Date', 'wl_up': 'Water Level (wl_up)'})
    fig.update_layout(xaxis_title="Date", yaxis_title="Water Level (wl_up)")
    return fig

# ฟังก์ชันสำหรับการแสดงกราฟข้อมูลช่วงที่เลือกก่อนการตัด
def plot_selected_time_range(data, start_date, end_date):
    selected_data = data[(data.index.date >= start_date) & (data.index.date <= end_date)]
    fig = px.line(selected_data, x=selected_data.index, y='wl_up', title=f'Water Level from {start_date} to {end_date}', labels={'x': 'Date', 'wl_up': 'Water Level (wl_up)'})
    fig.update_layout(xaxis_title="Date", yaxis_title="Water Level (wl_up)")
    return fig

# ฟังก์ชันสำหรับการแสดงกราฟข้อมูลหลังตัดค่า (ใช้ plotly)
def plot_original_data(data, original_nan_indexes=None):
    data = data.sort_index()
    fig = px.line(data, x=data.index, y='wl_up', title='Water Level Over Time (After Cutting)', labels={'x': 'Date', 'wl_up': 'Water Level (wl_up)'})
    if original_nan_indexes is not None:
        fig.add_scatter(x=original_nan_indexes, y=data.loc[original_nan_indexes, 'wl_up'], mode='markers', name='Missing Values (Cut)', marker=dict(color='orange'))
    fig.update_layout(xaxis_title="Date", yaxis_title="Water Level (wl_up)")
    return fig

# ฟังก์ชันสำหรับการแสดงกราฟที่ถูกเติมค่าแล้ว (ใช้ plotly)
def plot_filled_data(original_data, filled_data, original_nan_indexes):
    original_data = original_data.sort_index()
    filled_data = filled_data.sort_index()
    
    # กราฟของค่าจริง
    fig = px.line(original_data, x=original_data.index, y='wl_up', title='Water Level Over Time (After Filling)', labels={'x': 'Date', 'wl_up': 'Water Level (wl_up)'})

    # Plot ค่าที่ถูกตัดออก (สีส้ม)
    if original_nan_indexes is not None:
        fig.add_scatter(x=original_nan_indexes, y=original_data.loc[original_nan_indexes, 'wl_up'], mode='markers', name='Cut Values', marker=dict(color='orange'))

    # Plot ค่าที่เติมด้วยโมเดล (สีเขียว)
    if original_nan_indexes is not None:
        fig.add_scatter(x=filled_data.loc[original_nan_indexes].index, y=filled_data.loc[original_nan_indexes, 'wl_up'], mode='lines', name='Filled Values (Model)', line=dict(color='green'))

    fig.update_layout(xaxis_title="Date", yaxis_title="Water Level (wl_up)")
    return fig

# ฟังก์ชันสำหรับการเติมค่าด้วย RandomForestRegressor
def fill_missing_values(full_data):
    filled_data = full_data.copy()
    filled_data['week'] = filled_data.index.to_period("W")
    filled_data['month'] = filled_data.index.to_period("M")
    missing_weeks = filled_data[filled_data['wl_up'].isna()]['week'].unique()
    for week in missing_weeks:
        week_data = filled_data[filled_data['week'] == week]
        missing_idx = week_data[week_data['wl_up'].isna()].index
        train_data = week_data.dropna(subset=['wl_up', 'hour', 'day_of_week', 'minute', 'lag_1', 'lag_2'])
        if len(train_data) > 1:
            X_train = train_data[['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
            y_train = train_data['wl_up']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
        else:
            prev_week = pd.Period(week, freq='W') - 1
            prev_week_data = filled_data[filled_data['week'] == prev_week.strftime('%Y-%m-%d')]
            if len(prev_week_data.dropna(subset=['wl_up', 'hour', 'day_of_week', 'minute', 'lag_1', 'lag_2'])) > 1:
                X_train = prev_week_data[['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
                y_train = prev_week_data['wl_up']
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
            else:
                current_month = filled_data.loc[week_data.index[0], 'month']
                month_data = filled_data[(filled_data['month'] == current_month) & (filled_data['week'] != week)]
                month_train_data = month_data.dropna(subset=['wl_up', 'hour', 'day_of_week', 'minute', 'lag_1', 'lag_2'])
                if len(month_train_data) > 1:
                    X_train = month_train_data[['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
                    y_train = month_train_data['wl_up']
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
        X_missing = week_data.loc[missing_idx, ['hour', 'day_of_week', 'minute', 'lag_1', 'lag_2']]
        X_missing_clean = X_missing.dropna()
        if not X_missing_clean.empty:
            filled_values = model.predict(X_missing_clean)
            filled_data.loc[X_missing_clean.index, 'wl_up'] = filled_values
    return filled_data

# ฟังก์ชันคำนวณความแม่นยำ
def calculate_accuracy(filled_data, original_data, original_nan_indexes):
    actual_values = original_data.loc[original_nan_indexes, 'wl_up']
    predicted_values = filled_data.loc[original_nan_indexes, 'wl_up']
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Root Mean Square Error (RMSE): {rmse:.4f}")

# ใช้ session state เพื่อเก็บค่าการเลือกวันที่
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = None
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = None
if 'selected_graph' not in st.session_state:
    st.session_state['selected_graph'] = None

# อัปโหลดไฟล์ CSV ข้อมูลจริง
uploaded_file = st.file_uploader("เลือกไฟล์ CSV ข้อมูลจริง", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['datetime'] = data['datetime'].dt.tz_localize(None)
    data.set_index('datetime', inplace=True)
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['minute'] = data.index.minute
    data['lag_1'] = data['wl_up'].shift(1)
    data['lag_2'] = data['wl_up'].shift(2)
    data['lag_1'].ffill(inplace=True)
    data['lag_2'].ffill(inplace=True)

    # ตัดข้อมูลที่มีค่า wl_up น้อยกว่า 100 ออก
    filtered_data = data[data['wl_up'] >= 100]

    # แสดงกราฟข้อมูลทั้งไฟล์ก่อน
    st.subheader('กราฟตัวอย่างข้อมูลทั้งไฟล์หลังจากตัดค่าที่น้อยกว่า 100 ออก')
    st.plotly_chart(plot_full_data(filtered_data))

    st.subheader("เลือกช่วงวันที่ที่สนใจก่อนการตัดข้อมูล")
    start_date = st.date_input("เลือกวันเริ่มต้น (ดูข้อมูล)", pd.to_datetime(filtered_data.index.min()).date())
    end_date = st.date_input("เลือกวันสิ้นสุด (ดูข้อมูล)", pd.to_datetime(filtered_data.index.max()).date())

    if st.button("ตกลง (แสดงข้อมูลช่วงที่สนใจ)"):
        st.session_state['start_date'] = start_date
        st.session_state['end_date'] = end_date
        st.session_state['selected_graph'] = plot_selected_time_range(filtered_data, start_date, end_date)
        st.plotly_chart(st.session_state['selected_graph'])

    if st.session_state['start_date'] is not None and st.session_state['end_date'] is not None:
        st.subheader("เลือกช่วงวันที่และเวลาที่ต้องการตัดข้อมูล")
        start_date_cut = st.date_input("เลือกวันเริ่มต้น (ตัดข้อมูล)", pd.to_datetime(filtered_data.index.min()).date(), key="start_date_cut")
        start_time_cut = st.time_input("เลือกเวลาเริ่มต้น (ตัดข้อมูล)", value=pd.to_datetime(filtered_data.index.min()).time(), key="start_time_cut")
        end_date_cut = st.date_input("เลือกวันสิ้นสุด (ตัดข้อมูล)", pd.to_datetime(filtered_data.index.max()).date(), key="end_date_cut")
        end_time_cut = st.time_input("เลือกเวลาสิ้นสุด (ตัดข้อมูล)", value=pd.to_datetime(filtered_data.index.max()).time(), key="end_time_cut")

        start_datetime_cut = pd.to_datetime(f"{start_date_cut} {start_time_cut}")
        end_datetime_cut = pd.to_datetime(f"{end_date_cut} {end_time_cut}")

        if st.button("ตัดข้อมูล"):
            original_data = filtered_data.copy()
            date_mask = (filtered_data.index >= start_datetime_cut) & (filtered_data.index <= end_datetime_cut)
            if date_mask.any():
                filtered_data.loc[date_mask, 'wl_up'] = np.nan
                original_nan_indexes = filtered_data[filtered_data['wl_up'].isna()].index

                # แสดงกราฟช่วงที่สนใจก่อน (คงไว้แค่กราฟเดียว)
                st.plotly_chart(st.session_state['selected_graph'])

                # แสดงกราฟข้อมูลที่ถูกตัด
                st.subheader('กราฟข้อมูลหลังจากตัดค่าออก')
                cut_graph = plot_original_data(filtered_data, original_nan_indexes)
                st.plotly_chart(cut_graph)

                # เติมค่าด้วย RandomForest
                filled_data = fill_missing_values(filtered_data)

                # คำนวณความแม่นยำ
                st.subheader('ผลการคำนวณความแม่นยำ')
                calculate_accuracy(filled_data, original_data, original_nan_indexes)

                # แสดงกราฟข้อมูลที่เติมค่าแล้ว
                st.subheader('กราฟผลลัพธ์การเติมค่า')
                st.plotly_chart(plot_filled_data(original_data, filled_data, original_nan_indexes))

                # แสดงผลลัพธ์เป็นตาราง
                st.subheader('ตารางข้อมูลที่เติมค่า (datetime, wl_up)')
                st.write(filled_data[['wl_up']])
            else:
                st.error("ไม่พบข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกวันที่ใหม่")