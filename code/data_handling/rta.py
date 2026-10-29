import sys
sys.path.append("..")


import pandas as pd
import re
from datetime import datetime
from itertools import combinations


SPECIAL_YEAR = 9999   # обозначает "ещё не вступило" (например, 01-Jan-9999)


def parse_year(value):
    """
    Преобразует строку с датой в год.
    Возвращает None, если дата пустая или не распознана.
    """
    if pd.isna(value) or not str(value).strip():
        return None

    value = str(value).strip()

    # Расширенный список форматов, включая даты с "/" и "-" и двухзначным годом
    formats = [
        "%d-%b-%Y",   # 01-Jan-1958
        "%d-%b-%y",   # 01-Jan-58
        "%Y",         # 1958
        "%d/%m/%Y",   # 01/01/1958
        "%d/%m/%y",   # 01/01/58
        "%d-%m-%Y",   # 01-01-1958
        "%d-%m-%y",   # 01-01-58
        "%Y-%m-%d",   # 1958-01-01
        "%y-%m-%d",   # 58-01-01
        "%m/%d/%Y",   # 01/01/1958 (американский формат)
        "%m/%d/%y",   # 01/01/58
        "%m-%d-%Y",
        "%m-%d-%y"
    ]

    current_year = datetime.now().year

    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            year = dt.year

            # Если использовался двухзначный год и полученный год в будущем,
            # скорее всего, это 1900-е, а не 2000-е.
            if "%y" in fmt and year > current_year:
                year -= 100

            return year if year < 9000 else SPECIAL_YEAR
        except ValueError:
            pass

    # Запасной вариант: поиск четырёхзначного года (19xx или 20xx)
    m = re.search(r"\b(19|20)\d{2}\b", value)
    if m:
        year = int(m.group())
        return year if year < 9000 else SPECIAL_YEAR

    return None

def parse_specific_dates(text):
    """
    Разбирает поле 'Specific Entry/Exit dates'.
    Возвращает словарь {страна: (год_вступления, год_выхода)}.
    """
    result = {}

    if pd.isna(text) or not str(text).strip():
        return result

    for token in str(text).split(";"):
        token = token.strip()
        if not token:
            continue

        # страна может содержать скобки (например, Kosovo (under UNSC res. 1244))
        m = re.match(
            r"^(?P<country>.+)\((?P<entry>[^()]*?)\s*-\s*(?P<exit>[^()]*?)\)$",
            token,
        )
        if not m:
            continue

        country = m.group("country").strip().rstrip(",").strip()
        entry_year = parse_year(m.group("entry"))
        exit_year = parse_year(m.group("exit"))
        result[country] = (entry_year, exit_year)

    return result


def get_general_entry_year(row):
    """
    Общий год вступления из колонок 'Date of Entry into Force (G)' и '(S)'.
    """
    for col in ("Date of Entry into Force (G)", "Date of Entry into Force (S)"):
        y = parse_year(row.get(col, ""))
        if y and y != SPECIAL_YEAR:
            return y
    return None


def get_parties(row, specific_map):
    """
    Список всех сторон RTA.
    Приоритет: Original signatories, затем Current signatories,
    затем добавляем тех, кто упомянут в specific_map.
    """
    original = [
        p.strip()
        for p in (str(row.get("Original signatories") or "")).split(";")
        if p.strip()
    ]
    current = [
        p.strip()
        for p in (str(row.get("Current signatories") or "")).split(";")
        if p.strip()
    ]

    parties = original if original else current
    seen = set(parties)

    for p in specific_map:
        if p not in seen:
            parties.append(p)
            seen.add(p)

    return parties


def extract_rows(row):
    """
    Для одной строки DataFrame возвращает список словарей с полями
    'Country', 'Entry Year', 'Exit Year'.
    """
    specific_map = parse_specific_dates(row.get("Specific Entry/Exit dates"))
    gen_entry = get_general_entry_year(row)
    parties = get_parties(row, specific_map)

    rows = []
    for party in parties:
        entry, exit_ = specific_map.get(party, (None, None))

        entry_out = "" if entry == SPECIAL_YEAR else (entry if entry is not None else gen_entry)
        exit_out = "" if exit_ == SPECIAL_YEAR else exit_

        rows.append({
            "Country": party,
            "Entry Year": entry_out if entry_out else None,
            "Exit Year": exit_out if exit_out else None,
        })
    return rows


def process_rta_data(file_path, sep=",", only_in_force=False):
    """
    Читает CSV-файл с данными RTA, преобразует в длинный формат.

    Параметры:
        file_path     : путь к CSV-файлу
        sep           : разделитель (по умолчанию ',', для TSV укажите '\t')
        only_in_force : если True, оставить только действующие соглашения
                        (статус содержит 'In Force')

    Возвращает pandas DataFrame с колонками:
        RTA Name, Country, Entry Year, Exit Year, Status
    """
    df = pd.read_csv(file_path, sep=sep, encoding="utf-8-sig", dtype=str)

    # Применяем функцию к каждой строке и создаём список записей
    all_rows = []
    for _, row in df.iterrows():
        rta_name = str(row.get("RTA Name") or "").strip()
        if not rta_name:
            continue

        status = str(row.get("Status") or "").strip()
        
        rta_type = str(row.get("Type") or "").strip()
        
        extracted = extract_rows(row)

        for rec in extracted:
            rec["RTA Name"] = rta_name
            rec["Status"] = status
            rec["Type"] = rta_type
            all_rows.append(rec)

    result = pd.DataFrame(all_rows)

    # Фильтрация по действующим соглашениям
    if only_in_force:
        mask = result["Status"].str.lower().str.contains("in force", na=False)
        result = result[mask]

    return result

def create_pair_year_df(df, fill_exit_year=2026):
    """
    Преобразует DataFrame с членством стран в RTA в формат:

        country_a, country_b, year, value

    где value — число соглашений, действующих между страной A и страной B
    в данном году.

    Параметры
    ---------
    df : pandas.DataFrame
        Колонки: Country, Entry Year, Exit Year, RTA Name, Status.
        Могут быть и другие колонки, они игнорируются.
    fill_exit_year : int
        Год, которым заменяется пропущенный год выхода.
        По умолчанию 2026.

    Возвращает
    ----------
    pandas.DataFrame
        Колонки: country_a, country_b, year, value.
        Пары стран неупорядоченные, country_a < country_b.
    """
    df = df[["Country", "Entry Year", "Exit Year", "RTA Name"]].copy()

    # Приводим годы к целым числам, пропуски в Entry Year отбрасываем
    df["Entry Year"] = pd.to_numeric(df["Entry Year"], errors="coerce")
    df["Exit Year"] = pd.to_numeric(df["Exit Year"], errors="coerce")

    df = df.dropna(subset=["Country", "RTA Name", "Entry Year"])

    df["Entry Year"] = df["Entry Year"].astype(int)
    df["Exit Year"] = df["Exit Year"].fillna(fill_exit_year).astype(int)

    # На всякий случай удаляем дубликаты: одна страна в одном RTA один раз
    df = df.drop_duplicates(subset=["RTA Name", "Country"])

    records = []

    # Для каждого RTA строим пары стран и пересечения их периодов членства
    for rta, group in df.groupby("RTA Name"):
        group = group.reset_index(drop=True)

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                country_a = group.loc[i, "Country"]
                country_b = group.loc[j, "Country"]

                start_a = group.loc[i, "Entry Year"]
                end_a = group.loc[i, "Exit Year"]

                start_b = group.loc[j, "Entry Year"]
                end_b = group.loc[j, "Exit Year"]

                # Период, когда обе страны одновременно были в данном RTA
                start = max(start_a, start_b)
                end = min(end_a, end_b)

                if start <= end:
                    # Сортируем названия стран, чтобы пары были уникальными
                    a, b = sorted([country_a, country_b])
                    for year in range(start, end + 1):
                        records.append((a, b, year))

    if not records:
        return pd.DataFrame(columns=["country_a", "country_b", "year", "value"])

    pair_df = pd.DataFrame(records, columns=["country_a", "country_b", "year"])

    # Считаем количество соглашений для каждой пары стран в каждом году
    result = (
        pair_df.groupby(["country_a", "country_b", "year"])
        .size()
        .reset_index(name="value")
    )

    return result


# Пример использования
# df — ваш исходный DataFrame
#pair_year_df = create_pair_year_df(df, fill_exit_year=2026)
#print(pair_year_df.head())