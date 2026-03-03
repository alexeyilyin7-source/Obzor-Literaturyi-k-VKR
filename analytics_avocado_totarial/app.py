import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd

# --- Загрузка данных с дополнительной обработкой ---
data = pd.read_csv("avocado.csv", delimiter=';')  # Явно указываем разделитель

# Выведем первые несколько строк и названия колонок для отладки (можно удалить после проверки)
print("Колонки в данных:", data.columns.tolist())
print("Первые 5 строк:")
print(data.head())

# Переименуем колонки, чтобы избавиться от возможных проблем с пробелами
data.columns = data.columns.str.strip()

# Теперь работаем с колонкой 'Date'
# Формат даты в файле: день.месяц.год (27.12.2015)
data["Date"] = pd.to_datetime(data["Date"], format="%d.%m.%Y", errors='coerce')

# Удалим строки с некорректными датами
data = data.dropna(subset=['Date'])

data.sort_values("Date", inplace=True)

# --- Инициализация приложения ---
app = dash.Dash(__name__)
server = app.server

# --- Макет приложения (Layout) ---
app.layout = html.Div(
    children=[
        # Шапка
        html.Div(
            className="header",
            children=[
                html.H1("🥑 Avocado Analytics"),
                html.P("Анализ цен и объемов продаж авокадо в США (2015-2018)")
            ]
        ),
        # Панель фильтров
        html.Div(
            className="filters",
            children=[
                html.Div([
                    html.Label("Регион:"),
                    dcc.Dropdown(
                        id='region-dropdown',
                        options=[{'label': region, 'value': region} for region in sorted(data['region'].unique())],
                        value='Albany',
                        clearable=False
                    )
                ], className="filter-item"),
                html.Div([
                    html.Label("Тип авокадо:"),
                    dcc.Dropdown(
                        id='type-dropdown',
                        options=[{'label': atype, 'value': atype} for atype in data['type'].unique()],
                        value='conventional',
                        clearable=False
                    )
                ], className="filter-item"),
                html.Div([
                    html.Label("Диапазон дат:"),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=data['Date'].min(),
                        max_date_allowed=data['Date'].max(),
                        start_date=data['Date'].min(),
                        end_date=data['Date'].max()
                    )
                ], className="filter-item"),
            ]
        ),
        # Контейнер для графиков
        html.Div(
            className="graph-container",
            children=[
                html.Div(
                    className="graph-card",
                    children=[dcc.Graph(id='price-graph')]
                ),
                html.Div(
                    className="graph-card",
                    children=[dcc.Graph(id='volume-graph')]
                )
            ]
        )
    ]
)


# --- Функции обратного вызова (Callbacks) ---
@app.callback(
    [Output('price-graph', 'figure'),
     Output('volume-graph', 'figure')],
    [Input('region-dropdown', 'value'),
     Input('type-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graphs(selected_region, selected_type, start_date, end_date):
    # Фильтруем данные
    filtered_data = data[
        (data['region'] == selected_region) &
        (data['type'] == selected_type) &
        (data['Date'] >= start_date) &
        (data['Date'] <= end_date)
    ]

    # Проверяем, есть ли данные после фильтрации
    if filtered_data.empty:
        # Возвращаем пустые графики с сообщением
        empty_figure = {
            'data': [],
            'layout': {'title': 'Нет данных для выбранных параметров'}
        }
        return empty_figure, empty_figure

    price_figure = {
        'data': [{'x': filtered_data["Date"], 'y': filtered_data["AveragePrice"],
                  'type': 'line', 'name': 'Средняя цена', 'line': {'color': '#2ecc71'}}],
        'layout': {
            'title': f'Динамика средней цены в {selected_region} ({selected_type})',
            'xaxis': {'title': 'Дата'},
            'yaxis': {'title': 'Цена ($)'}
        }
    }

    volume_figure = {
        'data': [{'x': filtered_data["Date"], 'y': filtered_data["Total Volume"],
                  'type': 'line', 'name': 'Объем продаж', 'line': {'color': '#e74c3c'}}],
        'layout': {
            'title': f'Динамика объема продаж в {selected_region} ({selected_type})',
            'xaxis': {'title': 'Дата'},
            'yaxis': {'title': 'Объем'}
        }
    }
    return price_figure, volume_figure


# --- Запуск приложения ---
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=8050)