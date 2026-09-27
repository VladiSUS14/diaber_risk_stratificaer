from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"


def load_clean_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """
    Загружает очищенный CSV-файл.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_csv(path)

    # Дату визита снова приводим к формату даты,
    # потому что после сохранения в CSV она читается как строка.
    df["Дата визита"] = pd.to_datetime(df["Дата визита"], errors="coerce")

    return df


def first_valid(series: pd.Series):
    """
    Возвращает первое непустое значение.
    """
    values = series.dropna()
    if len(values) == 0:
        return np.nan
    return values.iloc[0]


def last_valid(series: pd.Series):
    """
    Возвращает последнее непустое значение.
    """
    values = series.dropna()
    if len(values) == 0:
        return np.nan
    return values.iloc[-1]


def any_positive(series: pd.Series):
    """
    Для бинарных признаков:
    возвращает 1, если хотя бы один раз было значение 1.
    возвращает 0, если были только 0.
    возвращает NaN, если данных нет.
    """
    values = series.dropna()

    if len(values) == 0:
        return np.nan

    if (values == 1).any():
        return 1

    return 0


def aggregate_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует данные до уровня пациента.

    Было:
        1 строка = 1 визит

    Станет:
        1 строка = 1 пациент
    """
    df = df.copy()

    # Сортируем, чтобы last_valid действительно брал последнее значение по времени.
    df = df.sort_values(["№", "Дата визита"])

    grouped = df.groupby("№", as_index=False)

    patient_df = grouped.agg(
        patient_id=("№", "first"),
        n_visits=("№", "size"),

        first_visit_date=("Дата визита", "min"),
        last_visit_date=("Дата визита", "max"),

        sex=("Пол", first_valid),
        age_last=("Возраст", last_valid),

        diabetes_duration_last=("Стаж СД", last_valid),
        diabetes_duration_max=("Стаж СД", "max"),

        hba1c_mean=("HbA1c", "mean"),
        hba1c_last=("HbA1c", last_valid),
        hba1c_max=("HbA1c", "max"),

        hypoglycemia_mean=("Гипогликемии/год", "mean"),
        hypoglycemia_max=("Гипогликемии/год", "max"),

        ldl_mean=("ЛПНП", "mean"),
        ldl_last=("ЛПНП", last_valid),

        hdl_mean=("ЛПВП", "mean"),
        hdl_last=("ЛПВП", last_valid),

        triglycerides_mean=("ТГ", "mean"),
        triglycerides_last=("ТГ", last_valid),

        total_cholesterol_mean=("Общий холестерин", "mean"),
        total_cholesterol_last=("Общий холестерин", last_valid),

        albuminuria_max=("Альбуминурия", "max"),
        albuminuria_last=("Альбуминурия", last_valid),

        acr_max=("ACR", "max"),
        acr_last=("ACR", last_valid),

        egfr_min=("СКФ", "min"),
        egfr_last=("СКФ", last_valid),

        systolic_bp_mean=("САД", "mean"),
        systolic_bp_last=("САД", last_valid),

        diastolic_bp_mean=("ДАД", "mean"),
        diastolic_bp_last=("ДАД", last_valid),

        therapy_last=("Терапия", last_valid),

        ketoacidosis_any=("Кетоацидоз", any_positive),
        angiopathy_any=("Ангиопатия", any_positive),
        cataract_any=("Катаракта", any_positive),
        polyneuropathy_any=("Полинейропатия", any_positive),
        steatosis_any=("Стеатоз", any_positive),
        lipodystrophy_any=("Липодистрофии", any_positive),
    )

    return patient_df


def create_nephropathy_target(patient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт целевую переменную риска диабетической нефропатии.

    В рамках исследовательского прототипа используем правило:

    nephropathy_risk = 1, если:
    - ACR >= 3
    или
    - альбуминурия >= 30
    или
    - СКФ < 60

    nephropathy_risk = 0, если признаки риска не выявлены.

    Если по пациенту нет ни ACR, ни альбуминурии, ни СКФ,
    то целевая переменная остаётся пропущенной.
    """
    patient_df = patient_df.copy()

    has_acr = patient_df["acr_max"].notna()
    has_albuminuria = patient_df["albuminuria_max"].notna()
    has_egfr = patient_df["egfr_min"].notna()

    has_any_kidney_data = has_acr | has_albuminuria | has_egfr

    acr_risk = patient_df["acr_max"] >= 3
    albuminuria_risk = patient_df["albuminuria_max"] >= 30
    egfr_risk = patient_df["egfr_min"] < 60

    patient_df["target_acr_risk"] = acr_risk.astype(int)
    patient_df["target_albuminuria_risk"] = albuminuria_risk.astype(int)
    patient_df["target_egfr_risk"] = egfr_risk.astype(int)

    patient_df["nephropathy_risk"] = np.where(
        has_any_kidney_data,
        (acr_risk | albuminuria_risk | egfr_risk).astype(int),
        np.nan,
    )

    return patient_df


def create_angiopathy_target(patient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт целевую переменную риска диабетической ретинопатии / ангиопатии.

    В рамках исследовательского прототипа используем признак angiopathy_any.

    angiopathy_risk = 1, если ангиопатия была отмечена хотя бы на одном визите.
    angiopathy_risk = 0, если ангиопатия не была отмечена.
    """
    patient_df = patient_df.copy()

    if "angiopathy_any" not in patient_df.columns:
        raise KeyError("В датасете отсутствует столбец angiopathy_any")

    patient_df["angiopathy_risk"] = patient_df["angiopathy_any"]

    return patient_df


def create_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт итоговый датасет для обучения модели.
    """
    patient_df = aggregate_by_patient(df)
    patient_df = create_nephropathy_target(patient_df)
    patient_df = create_angiopathy_target(patient_df)

    return patient_df


def save_model_dataset(patient_df: pd.DataFrame, path: Path = MODEL_DATA_PATH) -> None:
    """
    Сохраняет датасет для машинного обучения.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    patient_df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    print("Загрузка очищенных данных...")
    df_clean = load_clean_data()

    print("Размер очищенной таблицы:", df_clean.shape)
    print("Количество уникальных пациентов:", df_clean["№"].nunique())

    print("Агрегация данных по пациентам...")
    model_df = create_model_dataset(df_clean)

    print("Размер обучающего датасета:", model_df.shape)

    print("\nРаспределение целевой переменной nephropathy_risk:")
    print(model_df["nephropathy_risk"].value_counts(dropna=False))

    print("\nРаспределение целевой переменной angiopathy_risk:")
    print(model_df["angiopathy_risk"].value_counts(dropna=False))

    print("\nПервые строки обучающего датасета:")
    print(model_df.head())

    print("\nСохранение обучающего датасета...")
    save_model_dataset(model_df)

    print(f"Обучающий датасет сохранён: {MODEL_DATA_PATH}")


if __name__ == "__main__":
    main()