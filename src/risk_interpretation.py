import pandas as pd


def is_present(value) -> bool:
    """
    Проверяет, что значение не пустое.
    """
    return value is not None and not pd.isna(value)


def add_warning(warnings: list[str], condition: bool, text: str) -> None:
    """
    Добавляет предупреждение в список, если условие выполнено.
    """
    if condition:
        warnings.append(text)


def interpret_single_patient_factors(
    hba1c=None,
    diabetes_duration=None,
    systolic_bp=None,
    diastolic_bp=None,
    systolic_bp_percentile=None,
    diastolic_bp_percentile=None,
    ldl=None,
    hdl=None,
    triglycerides=None,
    total_cholesterol=None,
    acr=None,
    albuminuria=None,
    egfr=None,
    hypoglycemia_per_year=None,
    ketoacidosis=0,
    angiopathy=0,
    cataract=0,
    polyneuropathy=0,
    steatosis=0,
    lipodystrophy=0,
) -> list[str]:
    """
    Формирует краткую расшифровку факторов риска для одного пациента.

    Эта функция не заменяет медицинскую диагностику.
    Она нужна для пояснения результата модели врачу.
    """
    warnings = []

    # Гликемический контроль
    if is_present(hba1c):
        add_warning(
            warnings,
            hba1c > 7.0,
            f"HbA1c выше целевого уровня для детей с сахарным диабетом 1 типа ({hba1c:.1f}%)."
        )

    # Длительность заболевания
    if is_present(diabetes_duration):
        add_warning(
            warnings,
            diabetes_duration >= 5,
            f"Длительный стаж сахарного диабета ({diabetes_duration:.1f} лет)."
        )

    # Артериальное давление
    # Для детей и подростков предпочтительно оценивать АД по процентилям
    # с учетом возраста, пола и роста.
    has_systolic_percentile = is_present(systolic_bp_percentile)
    has_diastolic_percentile = is_present(diastolic_bp_percentile)

    if has_systolic_percentile or has_diastolic_percentile:
        max_bp_percentile = max(
            [
                value
                for value in [systolic_bp_percentile, diastolic_bp_percentile]
                if is_present(value)
            ]
        )

        if max_bp_percentile >= 95:
            warnings.append(
                f"Артериальное давление соответствует уровню гипертензии "
                f"по процентилям: максимальный процентиль АД — {max_bp_percentile:.0f}."
            )
        elif max_bp_percentile >= 90:
            warnings.append(
                f"Артериальное давление соответствует высокому нормальному уровню "
                f"по процентилям: максимальный процентиль АД — {max_bp_percentile:.0f}."
            )
    else:
        # Если процентили не введены, используем упрощенную подсказку.
        # Для детей окончательная оценка АД должна выполняться по процентилям.
        if is_present(systolic_bp) and systolic_bp >= 120:
            warnings.append(
                f"САД составляет {systolic_bp:.0f} мм рт. ст. "
                "Для детей рекомендуется оценивать артериальное давление по процентилям "
                "с учетом возраста, пола и роста."
            )

        if is_present(diastolic_bp) and diastolic_bp >= 80:
            warnings.append(
                f"ДАД составляет {diastolic_bp:.0f} мм рт. ст. "
                "Для детей рекомендуется оценивать артериальное давление по процентилям "
                "с учетом возраста, пола и роста."
            )

    # Липидный профиль
    if is_present(ldl):
        add_warning(
            warnings,
            ldl > 2.6,
            f"Повышенный уровень ЛПНП ({ldl:.2f})."
        )

    if is_present(hdl):
        add_warning(
            warnings,
            hdl < 1.0,
            f"Сниженный уровень ЛПВП ({hdl:.2f})."
        )

    if is_present(triglycerides):
        add_warning(
            warnings,
            triglycerides > 1.7,
            f"Повышенный уровень триглицеридов ({triglycerides:.2f})."
        )

    if is_present(total_cholesterol):
        add_warning(
            warnings,
            total_cholesterol > 5.2,
            f"Повышенный общий холестерин ({total_cholesterol:.2f})."
        )

    # Почечные показатели — отображаются как клинические факторы,
    # но не являются входными признаками ML-модели, если по ним формировалась целевая переменная.
    if is_present(acr):
        add_warning(
            warnings,
            acr >= 3,
            f"Повышенное альбумин-креатининовое соотношение ACR ({acr:.2f})."
        )

    if is_present(albuminuria):
        add_warning(
            warnings,
            albuminuria >= 30,
            f"Повышенная альбуминурия ({albuminuria:.2f})."
        )

    if is_present(egfr):
        add_warning(
            warnings,
            egfr < 60,
            f"Сниженная скорость клубочковой фильтрации ({egfr:.1f})."
        )

    # Гипогликемии
    if is_present(hypoglycemia_per_year):
        add_warning(
            warnings,
            hypoglycemia_per_year > 0,
            f"Зафиксированы эпизоды гипогликемии ({hypoglycemia_per_year:.0f} в год)."
        )

    # Наличие осложнений и сопутствующих состояний
    add_warning(
        warnings,
        ketoacidosis == 1,
        "В анамнезе отмечен кетоацидоз."
    )

    add_warning(
        warnings,
        angiopathy == 1,
        "Отмечены признаки ангиопатии."
    )

    add_warning(
        warnings,
        cataract == 1,
        "Отмечена катаракта."
    )

    add_warning(
        warnings,
        polyneuropathy == 1,
        "Отмечена полинейропатия."
    )

    add_warning(
        warnings,
        steatosis == 1,
        "Отмечен стеатоз."
    )

    add_warning(
        warnings,
        lipodystrophy == 1,
        "Отмечены липодистрофии."
    )

    if not warnings:
        warnings.append("Выраженные дополнительные факторы риска по введённым данным не выявлены.")

    return warnings


def interpret_patient_row(row: pd.Series) -> list[str]:
    """
    Формирует расшифровку факторов риска для строки таблицы.

    Используется для режима загрузки Excel/CSV.
    """
    return interpret_single_patient_factors(
        hba1c=row.get("HbA1c", row.get("hba1c", None)),
        diabetes_duration=row.get("Стаж СД", row.get("diabetes_duration", None)),
        systolic_bp=row.get("САД", row.get("systolic_bp", None)),
        diastolic_bp=row.get("ДАД", row.get("diastolic_bp", None)),
        systolic_bp_percentile=row.get("Процентиль САД", row.get("systolic_bp_percentile", None)),
        diastolic_bp_percentile=row.get("Процентиль ДАД", row.get("diastolic_bp_percentile", None)),
        ldl=row.get("ЛПНП", row.get("ldl", None)),
        hdl=row.get("ЛПВП", row.get("hdl", None)),
        triglycerides=row.get("ТГ", row.get("triglycerides", None)),
        total_cholesterol=row.get("Общий холестерин", row.get("total_cholesterol", None)),
        acr=row.get("ACR", row.get("acr", None)),
        albuminuria=row.get("Альбуминурия", row.get("albuminuria", None)),
        egfr=row.get("СКФ", row.get("egfr", None)),
        hypoglycemia_per_year=row.get("Гипогликемии/год", row.get("hypoglycemia_per_year", None)),
        ketoacidosis=row.get("Кетоацидоз", row.get("ketoacidosis", 0)),
        angiopathy=row.get("Ангиопатия", row.get("angiopathy", 0)),
        cataract=row.get("Катаракта", row.get("cataract", 0)),
        polyneuropathy=row.get("Полинейропатия", row.get("polyneuropathy", 0)),
        steatosis=row.get("Стеатоз", row.get("steatosis", 0)),
        lipodystrophy=row.get("Липодистрофии", row.get("lipodystrophy", 0)),
    )


def warnings_to_text(warnings: list[str]) -> str:
    """
    Преобразует список предупреждений в один текст.
    Удобно для таблицы результатов.
    """
    return "; ".join(warnings)


def main():
    """
    Тестовый запуск интерпретации факторов.
    """
    warnings = interpret_single_patient_factors(
        hba1c=9.2,
        diabetes_duration=8,
        systolic_bp=130,
        diastolic_bp=85,
        ldl=3.1,
        hdl=0.9,
        triglycerides=2.0,
        total_cholesterol=5.8,
        acr=4.2,
        albuminuria=35,
        egfr=58,
        hypoglycemia_per_year=1,
        ketoacidosis=0,
        angiopathy=0,
        cataract=0,
        polyneuropathy=0,
        steatosis=1,
        lipodystrophy=1,
    )

    print("Факторы риска:")
    for warning in warnings:
        print("-", warning)


if __name__ == "__main__":
    main()