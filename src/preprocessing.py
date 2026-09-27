from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes_data.xlsx"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"


NUMERIC_COLUMNS = [
    "№",
    "Возраст",
    "Пол",
    "Стаж СД",
    "HbA1c",
    "Гипогликемии/год",
    "ЛПНП",
    "ЛПВП",
    "ТГ",
    "Общий холестерин",
    "Альбуминурия",
    "ACR",
    "СКФ",
    "САД",
    "ДАД",
    "Терапия",
    "Кетоацидоз",
    "Ангиопатия",
    "Катаракта",
    "Полинейропатия",
    "Стеатоз",
    "Липодистрофии",
]


MISSING_VALUES = [
    "",
    " ",
    "-",
    "—",
    "–",
    "НД",
    "нд",
    "Нет данных",
    "нет данных",
    "nan",
    "NaN",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Загружает исходный Excel-файл.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    df = pd.read_excel(path)
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет пустые технические столбцы, которые появились из Excel.
    Например: Unnamed: 23.
    """
    df = df.copy()

    columns_to_drop = [
        col for col in df.columns
        if str(col).startswith("Unnamed")
    ]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    return df


def replace_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит разные обозначения пропусков к NaN.
    """
    df = df.copy()
    df = df.replace(MISSING_VALUES, np.nan)
    return df


def convert_visit_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует дату визита в нормальный формат даты.

    В Excel даты могут храниться двумя способами:
    1. как обычный текст;
    2. как серийное число Excel.

    Поэтому используем отдельную функцию обработки.
    """
    df = df.copy()

    def parse_excel_or_text_date(value):
        if pd.isna(value):
            return pd.NaT

        # Если дата хранится как Excel-число
        if isinstance(value, (int, float, np.integer, np.floating)):
            return pd.to_datetime(value, unit="D", origin="1899-12-30", errors="coerce")

        # Если дата хранится как текст
        return pd.to_datetime(value, errors="coerce", dayfirst=True)

    df["Дата визита"] = df["Дата визита"].apply(parse_excel_or_text_date)

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует числовые столбцы в числовой формат.
    Если значение невозможно преобразовать, оно станет NaN.
    """
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fix_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Исправляет клинически невозможные или некорректные значения.
    Такие значения заменяются на NaN.
    """
    df = df.copy()

    # Пол должен быть только 0 или 1.
    # Значение 1.5 является ошибочным.
    if "Пол" in df.columns:
        df.loc[~df["Пол"].isin([0, 1]), "Пол"] = np.nan

    # HbA1c не может быть равен 0.
    if "HbA1c" in df.columns:
        df.loc[df["HbA1c"] <= 0, "HbA1c"] = np.nan

    # Диастолическое давление не может быть равно 0.
    if "ДАД" in df.columns:
        df.loc[df["ДАД"] <= 0, "ДАД"] = np.nan

    # Систолическое давление тоже не должно быть 0 или отрицательным.
    if "САД" in df.columns:
        df.loc[df["САД"] <= 0, "САД"] = np.nan

    # ЛПВП не может быть равен 0.
    if "ЛПВП" in df.columns:
        df.loc[df["ЛПВП"] <= 0, "ЛПВП"] = np.nan

    # Возраст не должен быть отрицательным или нулевым.
    if "Возраст" in df.columns:
        df.loc[df["Возраст"] <= 0, "Возраст"] = np.nan

    # Стаж СД не должен быть отрицательным.
    if "Стаж СД" in df.columns:
        df.loc[df["Стаж СД"] < 0, "Стаж СД"] = np.nan

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полный пайплайн очистки данных.
    """
    df = drop_empty_columns(df)
    df = replace_missing_values(df)
    df = convert_visit_date(df)
    df = convert_numeric_columns(df)
    df = fix_invalid_values(df)

    return df


def save_clean_data(df: pd.DataFrame, path: Path = CLEAN_DATA_PATH) -> None:
    """
    Сохраняет очищенные данные.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    print("Загрузка исходных данных...")
    df_raw = load_raw_data()

    print("Размер исходной таблицы:", df_raw.shape)

    print("Очистка данных...")
    df_clean = clean_data(df_raw)

    print("Размер очищенной таблицы:", df_clean.shape)

    print("Сохранение очищенного файла...")
    save_clean_data(df_clean)

    print(f"Очищенные данные сохранены: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()