from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


MODEL_CONFIGS = {
    "nephropathy": {
        "display_name": "Диабетическая нефропатия",
        "model_path": MODELS_DIR / "nephropathy_model.joblib",
        "features_path": MODELS_DIR / "feature_columns.json",
        "metadata_path": MODELS_DIR / "model_metadata.json",
        "prefix": "nephropathy",
        "default_threshold": 0.5,
        "status": "main",
    },
    "angiopathy": {
        "display_name": "Диабетическая ретинопатия / ангиопатия",
        "model_path": MODELS_DIR / "angiopathy_model.joblib",
        "features_path": MODELS_DIR / "angiopathy_feature_columns.json",
        "metadata_path": MODELS_DIR / "angiopathy_metadata.json",
        "prefix": "angiopathy",
        "default_threshold": 0.5,
        "status": "experimental",
    },
}


def load_json(path: Path):
    """
    Загружает JSON-файл.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON-файл не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_bundle(complication_key: str) -> dict:
    """
    Загружает комплект файлов для конкретного осложнения:
    - модель;
    - список признаков;
    - метаданные.
    """
    if complication_key not in MODEL_CONFIGS:
        raise KeyError(f"Неизвестное осложнение: {complication_key}")

    config = MODEL_CONFIGS[complication_key]

    model_path = config["model_path"]
    features_path = config["features_path"]
    metadata_path = config["metadata_path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    model = joblib.load(model_path)
    feature_columns = load_json(features_path)

    if metadata_path.exists():
        metadata = load_json(metadata_path)
    else:
        metadata = {}

    return {
        "key": complication_key,
        "config": config,
        "model": model,
        "feature_columns": feature_columns,
        "metadata": metadata,
    }


def get_model_threshold(bundle: dict) -> float:
    """
    Получает порог классификации из метаданных модели.

    Для старой модели нефропатии используется ключ decision_threshold.
    Для новой модели ангиопатии используется ключ threshold.
    """
    metadata = bundle["metadata"]
    config = bundle["config"]

    threshold = metadata.get(
        "decision_threshold",
        metadata.get("threshold", config.get("default_threshold", 0.5))
    )

    return float(threshold)


def get_model_display_name(bundle: dict) -> str:
    """
    Возвращает человекочитаемое название осложнения.
    """
    metadata = bundle["metadata"]
    config = bundle["config"]

    return metadata.get("display_name", config["display_name"])


def get_model_name(bundle: dict) -> str:
    """
    Возвращает название ML-модели.
    """
    metadata = bundle["metadata"]
    return metadata.get("model_name", "Unknown model")


def get_model_status(bundle: dict) -> str:
    """
    Возвращает статус модели: основная / экспериментальная.
    """
    metadata = bundle["metadata"]
    config = bundle["config"]

    return metadata.get("status", config.get("status", "unknown"))


def load_model():
    """
    Совместимость со старым кодом.
    Загружает модель нефропатии.
    """
    return load_model_bundle("nephropathy")["model"]


def load_feature_columns():
    """
    Совместимость со старым кодом.
    Загружает признаки модели нефропатии.
    """
    return load_model_bundle("nephropathy")["feature_columns"]


def load_metadata():
    """
    Совместимость со старым кодом.
    Загружает метаданные модели нефропатии.
    """
    return load_model_bundle("nephropathy")["metadata"]


def load_all_metadata() -> dict:
    """
    Загружает метаданные всех доступных моделей.
    Используется для отображения информации в интерфейсе.
    """
    result = {}

    for key in MODEL_CONFIGS:
        bundle = load_model_bundle(key)
        result[key] = {
            "display_name": get_model_display_name(bundle),
            "model_name": get_model_name(bundle),
            "threshold": get_model_threshold(bundle),
            "status": get_model_status(bundle),
            "metrics": bundle["metadata"].get("metrics", {}),
            "note": bundle["metadata"].get("note", ""),
        }

    return result


def get_risk_level(probability: float) -> str:
    """
    Переводит вероятность риска в текстовый уровень риска.

    0–30%   — низкий риск
    31–70%  — умеренный риск
    71–100% — высокий риск
    """
    if probability < 0.30:
        return "Низкий риск"
    elif probability < 0.70:
        return "Умеренный риск"
    else:
        return "Высокий риск"


def get_binary_risk_class(probability: float, threshold: float) -> str:
    """
    Переводит вероятность риска в бинарное решение.
    """
    if probability >= threshold:
        return "Признаки риска выявлены"
    return "Признаки риска не выявлены"


def prepare_feature_dataframe(input_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Подготавливает таблицу признаков для модели.

    Модель ожидает строго определённый список столбцов.
    Если какого-то столбца нет, он добавляется как пустой.
    Если есть лишние столбцы, они не передаются в модель.
    """
    df = input_df.copy()

    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_columns]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def predict_complication_from_features(
    input_df: pd.DataFrame,
    complication_key: str,
) -> pd.DataFrame:
    """
    Делает прогноз риска для одного осложнения.
    """
    bundle = load_model_bundle(complication_key)

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    threshold = get_model_threshold(bundle)
    prefix = bundle["config"]["prefix"]

    X = prepare_feature_dataframe(input_df, feature_columns)

    probabilities = model.predict_proba(X)[:, 1]

    result_df = input_df.copy()

    result_df[f"{prefix}_probability"] = probabilities
    result_df[f"{prefix}_percent"] = (probabilities * 100).round(1)
    result_df[f"{prefix}_risk_level"] = [
        get_risk_level(prob) for prob in probabilities
    ]
    result_df[f"{prefix}_binary_result"] = [
        get_binary_risk_class(prob, threshold) for prob in probabilities
    ]

    return result_df


def predict_nephropathy_from_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Делает прогноз риска диабетической нефропатии.
    Оставлено для совместимости со старым Streamlit-кодом.
    """
    return predict_complication_from_features(input_df, "nephropathy")


def predict_angiopathy_from_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Делает прогноз риска диабетической ретинопатии / ангиопатии.
    """
    return predict_complication_from_features(input_df, "angiopathy")


def predict_all_complications_from_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Делает прогноз сразу по всем реализованным осложнениям.
    """
    result_df = input_df.copy()

    for complication_key in MODEL_CONFIGS:
        prediction_df = predict_complication_from_features(
            result_df,
            complication_key,
        )

        prefix = MODEL_CONFIGS[complication_key]["prefix"]

        result_df[f"{prefix}_probability"] = prediction_df[f"{prefix}_probability"]
        result_df[f"{prefix}_percent"] = prediction_df[f"{prefix}_percent"]
        result_df[f"{prefix}_risk_level"] = prediction_df[f"{prefix}_risk_level"]
        result_df[f"{prefix}_binary_result"] = prediction_df[f"{prefix}_binary_result"]

    return result_df


def create_single_patient_features(
    age: float,
    sex: int,
    diabetes_duration: float,
    hba1c: float,
    hypoglycemia_per_year: float | None = None,
    ldl: float | None = None,
    hdl: float | None = None,
    triglycerides: float | None = None,
    total_cholesterol: float | None = None,
    acr: float | None = None,
    albuminuria: float | None = None,
    egfr: float | None = None,
    systolic_bp: float | None = None,
    diastolic_bp: float | None = None,
    therapy: int = 0,
    ketoacidosis: int = 0,
    angiopathy: int = 0,
    cataract: int = 0,
    polyneuropathy: int = 0,
    steatosis: int = 0,
    lipodystrophy: int = 0,
) -> pd.DataFrame:
    """
    Создаёт строку признаков для одного пациента.

    Так как модели обучались на агрегированных данных,
    для одного введённого пациента mean/last/max заполняются
    одним и тем же текущим значением.
    """
    patient_data = {
        "n_visits": 1,

        "sex": sex,
        "age_last": age,

        "diabetes_duration_last": diabetes_duration,
        "diabetes_duration_max": diabetes_duration,

        "hba1c_mean": hba1c,
        "hba1c_last": hba1c,
        "hba1c_max": hba1c,

        "hypoglycemia_mean": hypoglycemia_per_year,
        "hypoglycemia_max": hypoglycemia_per_year,

        "ldl_mean": ldl,
        "ldl_last": ldl,

        "hdl_mean": hdl,
        "hdl_last": hdl,

        "triglycerides_mean": triglycerides,
        "triglycerides_last": triglycerides,

        "total_cholesterol_mean": total_cholesterol,
        "total_cholesterol_last": total_cholesterol,

        "albuminuria_max": albuminuria,
        "albuminuria_last": albuminuria,

        "acr_max": acr,
        "acr_last": acr,

        "egfr_min": egfr,
        "egfr_last": egfr,

        "systolic_bp_mean": systolic_bp,
        "systolic_bp_last": systolic_bp,

        "diastolic_bp_mean": diastolic_bp,
        "diastolic_bp_last": diastolic_bp,

        "therapy_last": therapy,

        "ketoacidosis_any": ketoacidosis,
        "angiopathy_any": angiopathy,
        "cataract_any": cataract,
        "polyneuropathy_any": polyneuropathy,
        "steatosis_any": steatosis,
        "lipodystrophy_any": lipodystrophy,
    }

    return pd.DataFrame([patient_data])


def row_to_single_result(row: pd.Series, complication_key: str) -> dict:
    """
    Преобразует строку результата в словарь для отображения в интерфейсе.
    """
    bundle = load_model_bundle(complication_key)
    prefix = bundle["config"]["prefix"]

    result = {
        "complication": get_model_display_name(bundle),
        "model_name": get_model_name(bundle),
        "status": get_model_status(bundle),
        "probability": float(row[f"{prefix}_probability"]),
        "percent": float(row[f"{prefix}_percent"]),
        "risk_level": row[f"{prefix}_risk_level"],
        "binary_result": row[f"{prefix}_binary_result"],
    }

    return result


def predict_single_patient(**kwargs) -> dict:
    """
    Делает прогноз нефропатии для одного пациента.

    Оставлено для совместимости со старым кодом Streamlit.
    """
    patient_df = create_single_patient_features(**kwargs)
    prediction_df = predict_nephropathy_from_features(patient_df)

    row = prediction_df.iloc[0]

    result = {
        "complication": "Диабетическая нефропатия",
        "probability": float(row["nephropathy_probability"]),
        "percent": float(row["nephropathy_percent"]),
        "risk_level": row["nephropathy_risk_level"],
        "binary_result": row["nephropathy_binary_result"],
    }

    return result


def predict_single_patient_all(**kwargs) -> list[dict]:
    """
    Делает прогноз всех реализованных осложнений для одного пациента.
    """
    patient_df = create_single_patient_features(**kwargs)
    prediction_df = predict_all_complications_from_features(patient_df)

    row = prediction_df.iloc[0]

    results = []

    for complication_key in MODEL_CONFIGS:
        results.append(row_to_single_result(row, complication_key))

    return results


def main():
    """
    Тестовый запуск файла predict.py.
    """
    print("Доступные модели:")

    all_metadata = load_all_metadata()

    for key, metadata in all_metadata.items():
        print(f"\n{key}")
        print("Осложнение:", metadata["display_name"])
        print("Модель:", metadata["model_name"])
        print("Порог:", metadata["threshold"])
        print("Статус:", metadata["status"])

    example_results = predict_single_patient_all(
        age=16,
        sex=1,
        diabetes_duration=8,
        hba1c=9.2,
        hypoglycemia_per_year=1,
        ldl=3.1,
        hdl=1.2,
        triglycerides=1.8,
        total_cholesterol=5.4,
        acr=4.2,
        albuminuria=35,
        egfr=58,
        systolic_bp=130,
        diastolic_bp=85,
        therapy=1,
        ketoacidosis=0,
        angiopathy=0,
        cataract=0,
        polyneuropathy=0,
        steatosis=1,
        lipodystrophy=1,
    )

    print("\nТестовые прогнозы для одного пациента:")

    for result in example_results:
        print("\nОсложнение:", result["complication"])
        print("Вероятность:", result["probability"])
        print("Процент:", result["percent"])
        print("Уровень риска:", result["risk_level"])
        print("Решение:", result["binary_result"])
        print("Статус:", result["status"])


if __name__ == "__main__":
    main()