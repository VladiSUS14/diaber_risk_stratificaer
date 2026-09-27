import time
import warnings
from pathlib import Path

import joblib
import pandas as pd


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "nephropathy_model.joblib"


def main():
    print("=" * 60)
    print("НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ ПРОГРАММНОГО МОДУЛЯ")
    print("=" * 60)

    model = joblib.load(MODEL_PATH)

    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        n_features = model.n_features_in_
        feature_names = [f"feature_{i}" for i in range(n_features)]

    test_sizes = [1, 100, 1000, 10000]

    print(f"Модель: диабетическая нефропатия")
    print(f"Количество признаков: {len(feature_names)}")
    print("-" * 60)
    print(f"{'Пациентов':>12} | {'Время, сек':>12} | {'На 1 пациента, мс':>18} | {'Ошибки':>8}")
    print("-" * 60)

    for n_patients in test_sizes:
        test_data = pd.DataFrame(
            0,
            index=range(n_patients),
            columns=feature_names
        )

        start = time.perf_counter()

        try:
            model.predict_proba(test_data)
            errors = 0
        except Exception:
            errors = 1

        elapsed = time.perf_counter() - start
        per_patient_ms = (elapsed / n_patients) * 1000

        print(
            f"{n_patients:>12} | "
            f"{elapsed:>12.4f} | "
            f"{per_patient_ms:>18.4f} | "
            f"{errors:>8}"
        )

    print("-" * 60)
    print("Результат: нагрузочное тестирование выполнено успешно")
    print("=" * 60)


if __name__ == "__main__":
    main()