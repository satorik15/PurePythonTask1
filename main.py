import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression
import concurrent.futures
import time
import asyncio
import aiohttp

# Настройка страницы
st.set_page_config(page_title="Анализ температур по городам", layout="wide")
st.title("Анализ температурных данных и мониторинг текущей температуры по городам")

# Секция 1: Загрузка данных
st.header("1. Загрузка исторических данных о температуре")
uploaded_file = st.file_uploader("Загрузите файл temperature_data.csv", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['city', 'timestamp'])

    # Показать основные статистики
    st.subheader("Основная информация о данных")
    st.write(f"Всего записей: {len(df)}")
    st.write(f"Период данных: с {df['timestamp'].min().date()} по {df['timestamp'].max().date()}")
    st.write(f"Города в данных: {', '.join(df['city'].unique())}")

    # Секция 2: Выбор города
    st.header("2. Выбор города для анализа температур")
    cities = df['city'].unique()
    selected_city = st.selectbox("Выберите город:", cities)

    # Фильтрация данных по выбранному городу
    city_df = df[df['city'] == selected_city].copy()
    city_df = city_df.sort_values('timestamp')
    city_df.set_index('timestamp', inplace=True)


    # Функция для анализа данных с возможностью использовать параллельность
    def analyze_city_data(city_data):
        # Скользящее среднее (30 дней) и стандартное отклонение
        city_data['rolling_mean'] = city_data['temperature'].rolling(window=30, min_periods=1).mean()
        city_data['rolling_std'] = city_data['temperature'].rolling(window=30, min_periods=1).std()

        # Определение аномалий
        city_data['anomaly'] = np.where(
            abs(city_data['temperature'] - city_data['rolling_mean']) > (2 * city_data['rolling_std']),
            'Аномалия',
            'Норма'
        )

        # Линейный тренд
        X = np.arange(len(city_data)).reshape(-1, 1)
        y = city_data['temperature'].values
        model = LinearRegression()
        model.fit(X, y)
        city_data['trend'] = model.predict(X)

        # Сезонная статистика
        seasonal_stats = city_data.groupby('season').agg({
            'temperature': ['mean', 'std']
        }).round(2)
        seasonal_stats.columns = ['Средняя температура', 'Стандартное отклонение']

        return city_data, seasonal_stats


    # Сравнение скорости с распараллеливанием и без
    st.subheader("Анализ скорости обработки")

    # Без распараллеливания
    start_time = time.time()
    for city in cities:
        temp_df = df[df['city'] == city].copy()
        analyze_city_data(temp_df)
    serial_time = time.time() - start_time

    # С распараллеливанием
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for city in cities:
            temp_df = df[df['city'] == city].copy()
            futures.append(executor.submit(analyze_city_data, temp_df))

        for future in concurrent.futures.as_completed(futures):
            future.result()
    parallel_time = time.time() - start_time

    st.write(f"Время обработки без распараллеливания: {serial_time} сек")
    st.write(f"Время обработки с распараллеливанием: {parallel_time} сек")
    st.write(f"Ускорение/замедление линейной обработки по сравнению с параллельной: {serial_time / parallel_time}x")

    # Применение анализа к выбранному городу
    analyzed_data, seasonal_stats = analyze_city_data(city_df.copy())

    # Визуализация исторических данных
    st.header("3. Анализ исторических данных")

    # Температура с трендом и аномалиями
    fig1 = make_subplots(rows=2, cols=1,
                         subplot_titles=('Температура и тренд', 'Сезонные профили'),
                         vertical_spacing=0.2)

    # Основной график температуры
    fig1.add_trace(
        go.Scatter(x=analyzed_data.index, y=analyzed_data['temperature'],
                   mode='lines', name='Температура', line=dict(color='lightgreen', width=1)),
        row=1, col=1
    )

    # Тренд
    fig1.add_trace(
        go.Scatter(x=analyzed_data.index, y=analyzed_data['trend'],
                   mode='lines', name='Линейный тренд', line=dict(color='red', width=3)),
        row=1, col=1
    )

    # Аномалии
    anomalies = analyzed_data[analyzed_data['anomaly'] == 'Аномалия']
    if not anomalies.empty:
        fig1.add_trace(
            go.Scatter(x=anomalies.index, y=anomalies['temperature'],
                       mode='markers', name='Аномалии', marker=dict(color='red', size=5)),
            row=1, col=1
        )

    # Сезонные профили
    seasons_order = ['winter', 'spring', 'summer', 'autumn']
    seasonal_stats = seasonal_stats.reindex(seasons_order)

    fig1.add_trace(
        go.Bar(x=seasonal_stats.index,
               y=seasonal_stats['Средняя температура'],
               name='Средняя температура',
               error_y=dict(type='data',
                            array=seasonal_stats['Стандартное отклонение'],
                            visible=True)),
        row=2, col=1
    )

    fig1.update_layout(height=800, showlegend=True)
    fig1.update_xaxes(title_text="Дата", row=1, col=1)
    fig1.update_yaxes(title_text="Температура в градусах цельсия", row=1, col=1)
    fig1.update_xaxes(title_text="Сезон", row=2, col=1)
    fig1.update_yaxes(title_text="Температура в градусах цельсия", row=2, col=1)

    st.plotly_chart(fig1, use_container_width=True)

    # Статистика по аномалиям
    st.subheader("Статистика аномалий")
    anomaly_count = len(analyzed_data[analyzed_data['anomaly'] == 'Аномалия'])
    total_count = len(analyzed_data)
    st.write(f"Всего аномалий: {anomaly_count} ({anomaly_count / total_count * 100:.1f}% от всех данных)")

    # Секция 4: Мониторинг текущей температуры
    st.header("4. Мониторинг текущей температуры")

    # Ввод API ключа
    api_key = st.text_input("Введите ваш OpenWeatherMap API ключ:", type="password")

    if api_key:
        # функции API запросов
        def sync_request(city, key):
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
            start = time.time()
            try:
                r = requests.get(url, timeout=5)
                return {
                    'time': time.time() - start,
                    'status': r.status_code,
                    'data': r.json() if r.status_code == 200 else {'message': r.text}
                }
            except Exception as e:
                return {'time': time.time() - start, 'error': str(e)}


        async def async_request(city, key):
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
            start = time.time()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as r:
                        data = await r.json() if r.status == 200 else {'message': await r.text()}
                        return {
                            'time': time.time() - start,
                            'status': r.status,
                            'data': data
                        }
            except Exception as e:
                return {'time': time.time() - start, 'error': str(e)}


        if st.button("Сравнить запросы к API"):
            col1, col2 = st.columns(2)

            # Синхронный запрос
            with col1:
                st.write("**Синхронный запрос:**")
                sync = sync_request(selected_city, api_key)
                if 'error' in sync:
                    st.error(f"Ошибка: {sync['error']}")
                elif sync['status'] == 401:
                    st.error(f"Ошибка 401: {sync['data'].get('message', 'Неверный API ключ')}")
                elif sync['status'] == 200:
                    temp = sync['data']['main']['temp']
                    st.write(f"Температура: {temp:.1f} Градусов Цельсия")
                    st.write(f"Время: {sync['time']:.3f}с")
                else:
                    st.error(f"Ошибка {sync['status']}")

            # Асинхронный запрос
            with col2:
                st.write("**Асинхронный запрос:**")
                try:
                    async def get_async():
                        return await async_request(selected_city, api_key)

                    async_start = time.time()
                    async_res = asyncio.run(get_async())
                    total_time = time.time() - async_start

                    if 'error' in async_res:
                        st.error(f"Ошибка: {async_res['error']}")
                    elif async_res['status'] == 401:
                        st.error(f"Ошибка 401: {async_res['data'].get('message', 'Неверный API ключ')}")
                    elif async_res['status'] == 200:
                        temp = async_res['data']['main']['temp']
                        st.write(f"Температура: {temp:.1f} Градусов Цельсия")
                        st.write(f"Время запроса: {total_time:.3f}с")
                    else:
                        st.error(f"Ошибка {async_res['status']}")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

            # Проверка аномальности
            if 'sync' in locals() and sync.get('status') == 200:
                current_temp = sync['data']['main']['temp']
                current_month = datetime.now().month
                season = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
                          6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 11: 'autumn'}[current_month]

                if season in seasonal_stats.index:
                    mean, std = seasonal_stats.loc[season, 'Средняя температура'], seasonal_stats.loc[season, 'Стандартное отклонение']
                    is_normal = abs(current_temp - mean) <= 2 * std

                    st.write(f"**Проверка аномальности:**")
                    st.write(f"Сезон: {season}, Норма: {mean:.1f} +/- {2 * std:.1f}°C")
                    if is_normal:
                        st.success(f"Температура {current_temp:.1f}°C - НОРМАЛЬНАЯ")
                    else:
                        st.warning(f"Температура {current_temp:.1f}°C - АНОМАЛЬНАЯ!")

    # Секция 5: Дополнительная информация
    st.header("5. Технические детали")
    st.info(f"""
                       **Вывод:** Для одиночного запроса к одному городу асинхронный подход не дает 
                       значительного преимущества. Более того, из-за накладных расходов на создание 
                       асинхронной сессии общее время выполнения может быть даже больше.

                       Асинхронный подход становится эффективным только при множественных 
                       одновременных запросах к разным API или при работе с медленными соединениями.
                       """)

else:
    st.info("Пожалуйста, загрузите файл temperature_data.csv для начала анализа")
