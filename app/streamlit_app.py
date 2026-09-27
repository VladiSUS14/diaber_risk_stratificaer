from pathlib import Path
import sys
from io import BytesIO

import pandas as pd
import streamlit as st
from openpyxl.utils import get_column_letter


# ------------------------------------------------------------
# Настройка путей
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.predict import (
    predict_single_patient_all,
    predict_all_complications_from_features,
    create_single_patient_features,
    load_feature_columns,
    load_all_metadata,
)

from src.risk_interpretation import (
    interpret_single_patient_factors,
    interpret_patient_row,
    warnings_to_text,
)

from src.preprocessing import clean_data
from src.feature_engineering import aggregate_by_patient


# ------------------------------------------------------------
# Настройка страницы
# ------------------------------------------------------------

st.set_page_config(
    page_title="Стратификация рисков осложнений СД",
    page_icon="🩺",
    layout="wide",
)


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------

def parse_optional_float(value: str):
    """
    Преобразует текстовое поле в число.

    Если поле пустое — возвращает None.
    Позволяет вводить числа как с точкой, так и с запятой.
    """
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    value = value.replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return None


def risk_level_color(risk_level: str) -> str:
    """
    Возвращает цвет для уровня риска.
    """
    if "Низкий" in risk_level:
        return "green"
    if "Умеренный" in risk_level:
        return "orange"
    if "Высокий" in risk_level:
        return "red"
    return "gray"


def show_single_complication_result(result: dict):
    """
    Отображает результат прогноза по одному осложнению.
    """
    complication_name = result["complication"]

    if result.get("status") == "experimental":
        complication_name += " (экспериментальная модель)"

    st.markdown(f"### {complication_name}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Расчётная вероятность риска",
            value=f"{result['percent']:.1f}%"
        )

    with col2:
        st.metric(
            label="Уровень риска",
            value=result["risk_level"]
        )

    with col3:
        st.metric(
            label="Решение модели",
            value=result["binary_result"]
        )

    color = risk_level_color(result["risk_level"])

    st.markdown(
        f"""
        <div style="
            padding: 16px;
            border-radius: 10px;
            border: 1px solid #ddd;
            background-color: #f8f9fa;
            color: #111111;
            font-size: 16px;
            line-height: 1.5;
            margin-bottom: 12px;
        ">
            <b>Итоговая интерпретация:</b><br>
            Пациент относится к категории: 
            <span style="color:{color}; font-weight:700;">{result["risk_level"]}</span>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.get("status") == "experimental":
        st.info(
            "Данная модель является экспериментальной. "
            "Её результат следует интерпретировать особенно осторожно."
        )


def show_prediction_results(results: list[dict], warnings: list[str]):
    """
    Отображает результаты прогноза по всем реализованным осложнениям.
    """
    st.subheader("Результаты стратификации риска")

    for result in results:
        show_single_complication_result(result)

    st.markdown("### Краткая расшифровка факторов риска")

    for warning in warnings:
        st.write(f"- {warning}")

    st.warning(
        "Результат является исследовательской оценкой и не является медицинским диагнозом. "
        "Окончательное решение должно приниматься врачом с учётом полной клинической картины."
    )


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Читает загруженный Excel или CSV-файл.
    """
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    raise ValueError("Поддерживаются только Excel и CSV-файлы.")


def build_features_from_single_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт признаки модели из файла, где каждая строка — один пациент или один текущий визит.

    Используется, если файл не является полным исходным датасетом с несколькими визитами.
    """
    feature_rows = []

    for _, row in df.iterrows():
        patient_features = create_single_patient_features(
            age=row.get("Возраст", None),
            sex=row.get("Пол", None),
            diabetes_duration=row.get("Стаж СД", None),
            hba1c=row.get("HbA1c", None),
            hypoglycemia_per_year=row.get("Гипогликемии/год", None),
            ldl=row.get("ЛПНП", None),
            hdl=row.get("ЛПВП", None),
            triglycerides=row.get("ТГ", None),
            total_cholesterol=row.get("Общий холестерин", None),
            acr=row.get("ACR", None),
            albuminuria=row.get("Альбуминурия", None),
            egfr=row.get("СКФ", None),
            systolic_bp=row.get("САД", None),
            diastolic_bp=row.get("ДАД", None),
            therapy=row.get("Терапия", 0),
            ketoacidosis=row.get("Кетоацидоз", 0),
            angiopathy=row.get("Ангиопатия", 0),
            cataract=row.get("Катаракта", 0),
            polyneuropathy=row.get("Полинейропатия", 0),
            steatosis=row.get("Стеатоз", 0),
            lipodystrophy=row.get("Липодистрофии", 0),
        )

        feature_rows.append(patient_features.iloc[0])

    return pd.DataFrame(feature_rows)


def interpret_aggregated_patient_row(row: pd.Series) -> list[str]:
    """
    Интерпретация факторов риска для агрегированной строки пациента.
    """
    return interpret_single_patient_factors(
        hba1c=row.get("hba1c_last", row.get("hba1c_mean", None)),
        diabetes_duration=row.get("diabetes_duration_last", row.get("diabetes_duration_max", None)),
        systolic_bp=row.get("systolic_bp_last", row.get("systolic_bp_mean", None)),
        diastolic_bp=row.get("diastolic_bp_last", row.get("diastolic_bp_mean", None)),
        systolic_bp_percentile=row.get("systolic_bp_percentile", row.get("Процентиль САД", None)),
        diastolic_bp_percentile=row.get("diastolic_bp_percentile", row.get("Процентиль ДАД", None)),
        ldl=row.get("ldl_last", row.get("ldl_mean", None)),
        hdl=row.get("hdl_last", row.get("hdl_mean", None)),
        triglycerides=row.get("triglycerides_last", row.get("triglycerides_mean", None)),
        total_cholesterol=row.get("total_cholesterol_last", row.get("total_cholesterol_mean", None)),
        acr=row.get("acr_last", row.get("acr_max", None)),
        albuminuria=row.get("albuminuria_last", row.get("albuminuria_max", None)),
        egfr=row.get("egfr_last", row.get("egfr_min", None)),
        hypoglycemia_per_year=row.get("hypoglycemia_mean", None),
        ketoacidosis=row.get("ketoacidosis_any", 0),
        angiopathy=row.get("angiopathy_any", 0),
        cataract=row.get("cataract_any", 0),
        polyneuropathy=row.get("polyneuropathy_any", 0),
        steatosis=row.get("steatosis_any", 0),
        lipodystrophy=row.get("lipodystrophy_any", 0),
    )


def prepare_uploaded_data_for_prediction(df: pd.DataFrame):
    """
    Подготавливает загруженные данные к прогнозу.

    Поддерживает три варианта:
    1. файл уже содержит признаки модели;
    2. файл имеет исходный формат с визитами пациентов;
    3. файл содержит одну строку на пациента с русскими названиями столбцов.
    """
    feature_columns = load_feature_columns()

    # Вариант 1: файл уже содержит признаки модели.
    if all(col in df.columns for col in feature_columns):
        features_df = df[feature_columns].copy()
        display_df = df.copy()
        mode = "model_features"
        return features_df, display_df, mode

    # Вариант 2: файл похож на исходный датасет с визитами.
    raw_required_cols = {"№", "Дата визита", "Возраст", "Пол", "Стаж СД", "HbA1c"}

    if raw_required_cols.issubset(set(df.columns)):
        cleaned_df = clean_data(df)
        aggregated_df = aggregate_by_patient(cleaned_df)

        features_df = aggregated_df.copy()
        display_df = aggregated_df.copy()
        mode = "aggregated_raw"
        return features_df, display_df, mode

    # Вариант 3: файл содержит одну строку на пациента.
    single_row_required_cols = {"Возраст", "Пол", "Стаж СД", "HbA1c"}

    if single_row_required_cols.issubset(set(df.columns)):
        features_df = build_features_from_single_rows(df)
        display_df = df.copy()
        mode = "single_rows"
        return features_df, display_df, mode

    raise ValueError(
        "Не удалось распознать структуру файла. "
        "В файле должны быть как минимум столбцы: Возраст, Пол, Стаж СД, HbA1c. "
        "Для исходного формата также нужны: № и Дата визита."
    )


def add_interpretation_to_results(result_df: pd.DataFrame, display_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Добавляет краткую расшифровку факторов риска к таблице результатов.
    """
    interpretations = []

    for idx in range(len(display_df)):
        row = display_df.iloc[idx]

        if mode in ["aggregated_raw", "model_features"]:
            warnings = interpret_aggregated_patient_row(row)
        else:
            warnings = interpret_patient_row(row)

        interpretations.append(warnings_to_text(warnings))

    result_df["Краткая расшифровка"] = interpretations

    return result_df


def convert_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Преобразует DataFrame в CSV для скачивания.
    """
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

def convert_df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Преобразует DataFrame в Excel-файл для скачивания.

    Excel-формат удобнее CSV, потому что:
    - корректно отображает русские заголовки;
    - сохраняет таблицу по столбцам;
    - лучше подходит для длинной расшифровки факторов риска.
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Результаты")

        worksheet = writer.sheets["Результаты"]

        # Закрепляем первую строку с заголовками
        worksheet.freeze_panes = "A2"

        # Автоматически подбираем ширину столбцов
        for column_index, column_name in enumerate(df.columns, start=1):
            values = df[column_name].astype(str).fillna("").head(200).tolist()
            max_length = max(
                [len(str(column_name))] + [len(value) for value in values]
            )

            adjusted_width = min(max(max_length + 2, 12), 60)

            worksheet.column_dimensions[
                get_column_letter(column_index)
            ].width = adjusted_width

    return output.getvalue()


# ------------------------------------------------------------
# Основной интерфейс
# ------------------------------------------------------------

st.title("Программный модуль стратификации рисков осложнений сахарного диабета")

st.write(
    "Приложение рассчитывает риск осложнений сахарного диабета на основе "
    "клинико-лабораторных данных пациента с использованием модели машинного обучения."
)

all_metadata = load_all_metadata()

with st.sidebar:
    st.header("Информация о моделях")

    for model_key, model_info in all_metadata.items():
        st.write(f"**{model_info['display_name']}**")
        st.write(f"Модель: {model_info.get('model_name', 'Не указано')}")
        st.write(f"Порог классификации: {float(model_info.get('threshold', 0.5)):.2f}")

        if model_info.get("status") == "experimental":
            st.caption("Статус: экспериментальная модель")
        else:
            st.caption("Статус: основная модель")

        st.divider()

    st.write("**Реализованные риски:**")
    st.write("- Диабетическая нефропатия")
    st.write("- Диабетическая ретинопатия / ангиопатия")

    st.info(
        "Модель нефропатии является основной. "
        "Модель ретинопатии/ангиопатии добавлена как экспериментальная "
        "из-за ограниченного объёма данных."
    )

mode = st.radio(
    "Выберите режим работы:",
    [
        "Ввод одного пациента",
        "Загрузка файла с пациентами",
    ],
)


# ------------------------------------------------------------
# Режим 1. Ручной ввод одного пациента
# ------------------------------------------------------------

if mode == "Ввод одного пациента":
    st.header("Ввод данных одного пациента")

    st.write(
        "Заполните известные показатели пациента. "
        "Необязательные поля можно оставить пустыми."
    )

    with st.form("single_patient_form"):
        st.subheader("Основные данные")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            age = st.number_input("Возраст", min_value=0.0, max_value=120.0, value=16.0, step=0.1)

        with col2:
            sex_label = st.selectbox("Пол", ["0 — женский", "1 — мужской"])
            sex = 0 if sex_label.startswith("0") else 1

        with col3:
            diabetes_duration = st.number_input("Стаж СД, лет", min_value=0.0, max_value=80.0, value=5.0, step=0.1)

        with col4:
            hba1c = st.number_input("HbA1c, %", min_value=0.0, max_value=25.0, value=8.0, step=0.1)

        st.subheader("Дополнительные лабораторные показатели")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            hypoglycemia_text = st.text_input("Гипогликемии/год", value="")
            ldl_text = st.text_input("ЛПНП", value="")

        with col2:
            hdl_text = st.text_input("ЛПВП", value="")
            triglycerides_text = st.text_input("ТГ", value="")

        with col3:
            total_cholesterol_text = st.text_input("Общий холестерин", value="")
            acr_text = st.text_input("ACR", value="")

        with col4:
            albuminuria_text = st.text_input("Альбуминурия", value="")
            egfr_text = st.text_input("СКФ", value="")

        st.subheader("Артериальное давление")

        st.info(
            "Для детей артериальное давление рекомендуется оценивать по процентилям "
            "с учетом возраста, пола и роста. Если процентили известны, врач может "
            "ввести их вручную."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            systolic_bp_text = st.text_input("САД", value="")

        with col2:
            diastolic_bp_text = st.text_input("ДАД", value="")

        with col3:
            systolic_bp_percentile_text = st.text_input("Процентиль САД", value="")

        with col4:
            diastolic_bp_percentile_text = st.text_input("Процентиль ДАД", value="")

        st.subheader("Осложнения и состояния")

        col1, col2, col3 = st.columns(3)

        with col1:
            therapy = st.selectbox("Терапия", [0, 1])
            ketoacidosis = st.selectbox("Кетоацидоз", [0, 1])

        with col2:
            cataract = st.selectbox("Катаракта", [0, 1])
            polyneuropathy = st.selectbox("Полинейропатия", [0, 1])

        with col3:
            steatosis = st.selectbox("Стеатоз", [0, 1])
            lipodystrophy = st.selectbox("Липодистрофии", [0, 1])

        # Ангиопатия не запрашивается у пользователя,
        # так как теперь она является прогнозируемым осложнением.
        angiopathy = 0

        submitted = st.form_submit_button("Рассчитать риски")

    if submitted:
        hypoglycemia_per_year = parse_optional_float(hypoglycemia_text)
        ldl = parse_optional_float(ldl_text)
        hdl = parse_optional_float(hdl_text)
        triglycerides = parse_optional_float(triglycerides_text)
        total_cholesterol = parse_optional_float(total_cholesterol_text)
        acr = parse_optional_float(acr_text)
        albuminuria = parse_optional_float(albuminuria_text)
        egfr = parse_optional_float(egfr_text)
        systolic_bp = parse_optional_float(systolic_bp_text)
        diastolic_bp = parse_optional_float(diastolic_bp_text)
        systolic_bp_percentile = parse_optional_float(systolic_bp_percentile_text)
        diastolic_bp_percentile = parse_optional_float(diastolic_bp_percentile_text)

        results = predict_single_patient_all(
            age=age,
            sex=sex,
            diabetes_duration=diabetes_duration,
            hba1c=hba1c,
            hypoglycemia_per_year=hypoglycemia_per_year,
            ldl=ldl,
            hdl=hdl,
            triglycerides=triglycerides,
            total_cholesterol=total_cholesterol,
            acr=acr,
            albuminuria=albuminuria,
            egfr=egfr,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            therapy=therapy,
            ketoacidosis=ketoacidosis,
            angiopathy=angiopathy,
            cataract=cataract,
            polyneuropathy=polyneuropathy,
            steatosis=steatosis,
            lipodystrophy=lipodystrophy,
        )

        optional_fields = {
            "Гипогликемии/год": hypoglycemia_per_year,
            "ЛПНП": ldl,
            "ЛПВП": hdl,
            "ТГ": triglycerides,
            "Общий холестерин": total_cholesterol,
            "ACR": acr,
            "Альбуминурия": albuminuria,
            "СКФ": egfr,
            "САД": systolic_bp,
            "ДАД": diastolic_bp,  
        }

        missing_fields = [
            field_name
            for field_name, field_value in optional_fields.items()
            if field_value is None
        ]

        if missing_fields:
            st.warning(
                "Некоторые показатели не заполнены. Расчёт выполнен на основе доступных данных; "
                "результат следует интерпретировать осторожно."
            )
        
            with st.expander("Показать незаполненные показатели"):
                for field_name in missing_fields:
                    st.write(f"- {field_name}")

        warnings = interpret_single_patient_factors(
            hba1c=hba1c,
            diabetes_duration=diabetes_duration,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            systolic_bp_percentile=systolic_bp_percentile,
            diastolic_bp_percentile=diastolic_bp_percentile,
            ldl=ldl,
            hdl=hdl,
            triglycerides=triglycerides,
            total_cholesterol=total_cholesterol,
            acr=acr,
            albuminuria=albuminuria,
            egfr=egfr,
            hypoglycemia_per_year=hypoglycemia_per_year,
            ketoacidosis=ketoacidosis,
            angiopathy=angiopathy,
            cataract=cataract,
            polyneuropathy=polyneuropathy,
            steatosis=steatosis,
            lipodystrophy=lipodystrophy,
        )

        show_prediction_results(results, warnings)


# ------------------------------------------------------------
# Режим 2. Загрузка файла
# ------------------------------------------------------------

if mode == "Загрузка файла с пациентами":
    st.header("Загрузка файла с пациентами")

    st.write(
        "Загрузите Excel или CSV-файл. "
        "Приложение автоматически рассчитает риски нефропатии "
        "и ретинопатии/ангиопатии для пациентов."
    )

    uploaded_file = st.file_uploader(
        "Выберите Excel/CSV-файл",
        type=["xlsx", "xls", "csv"]
    )

    if uploaded_file is not None:
        try:
            uploaded_df = read_uploaded_file(uploaded_file)

            st.subheader("Предпросмотр загруженных данных")

            st.write(
                f"Размер загруженной таблицы: "
                f"{uploaded_df.shape[0]} строк × {uploaded_df.shape[1]} столбцов"
            )

            st.dataframe(
                uploaded_df,
                use_container_width=True,
                height=350
            )

            if st.button("Рассчитать риски для файла"):
                features_df, display_df, detected_mode = prepare_uploaded_data_for_prediction(uploaded_df)

                prediction_df = predict_all_complications_from_features(features_df)

                output_df = pd.DataFrame()

                if "patient_id" in display_df.columns:
                    output_df["Пациент"] = display_df["patient_id"]
                elif "№" in display_df.columns:
                    output_df["Пациент"] = display_df["№"]
                else:
                    output_df["Пациент"] = range(1, len(prediction_df) + 1)

                output_df["Нефропатия: вероятность, %"] = prediction_df["nephropathy_percent"]
                output_df["Нефропатия: уровень риска"] = prediction_df["nephropathy_risk_level"]
                output_df["Нефропатия: решение модели"] = prediction_df["nephropathy_binary_result"]

                output_df["Ретинопатия/ангиопатия: вероятность, %"] = prediction_df["angiopathy_percent"]
                output_df["Ретинопатия/ангиопатия: уровень риска"] = prediction_df["angiopathy_risk_level"]
                output_df["Ретинопатия/ангиопатия: решение модели"] = prediction_df["angiopathy_binary_result"]

                output_df = add_interpretation_to_results(
                    output_df,
                    display_df,
                    detected_mode
                )

                st.subheader("Результаты расчёта")
                st.dataframe(
                    output_df,
                    use_container_width=True,
                    height=350
                )

                st.info(
                    "Рекомендуется скачивать результаты в формате Excel, так как он лучше подходит "
                    "для просмотра таблиц с русскими заголовками и длинной расшифровкой факторов риска."
                )

                col1, col2 = st.columns(2)

                with col1:
                    excel_bytes = convert_df_to_excel_bytes(output_df)

                    st.download_button(
                        label="Скачать результаты в Excel",
                        data=excel_bytes,
                        file_name="diabetes_complication_risk_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                with col2:
                    csv_bytes = convert_df_to_csv_bytes(output_df)

                    st.download_button(
                        label="Скачать результаты в CSV",
                        data=csv_bytes,
                        file_name="diabetes_complication_risk_predictions.csv",
                        mime="text/csv",
                    )

                st.success("Расчёт выполнен успешно.")

        except Exception as e:
            st.error("При обработке файла произошла ошибка.")
            st.exception(e)