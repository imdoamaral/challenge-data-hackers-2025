"""
Chat NL->SQL, sobre qualquer CSV, com interface Streamlit.

Uso:
    streamlit run challenge_llm.py
"""

import os, duckdb, pandas as pd, ast
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
import streamlit as st
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# 1. CRIA TABELA EM DUCKDB (com coluna salario_rank pré-calculada)
# ──────────────────────────────────────────────────────────────────────────────
def build_duckdb(csv_path: str, tabel: str, db_file: str = ':memory:') -> SQLDatabase:
    create_table_sql = f"""
    CREATE OR REPLACE TABLE {table} AS
    SELECT *,
        CASE
            WHEN faixa_salarial = 'Acima de R$ 40.001/mês'              THEN 45000.0
            WHEN faixa_salarial = 'de R$ 30.001/mês a R$ 40.000/mês'    THEN 35000.0
            WHEN faixa_salarial = 'de R$ 25.001/mês a R$ 30.000/mês'    THEN 27500.0
            WHEN faixa_salarial = 'de R$ 20.001/mês a R$ 25.000/mês'    THEN 22500.0
            WHEN faixa_salarial = 'de R$ 16.001/mês a R$ 20.000/mês'    THEN 18000.0
            WHEN faixa_salarial = 'de R$ 12.001/mês a R$ 16.000/mês'    THEN 14000.0
            WHEN faixa_salarial = 'de R$ 8.001/mês a R$ 12.000/mês'     THEN 10000.0
            WHEN faixa_salarial = 'de R$ 6.001/mês a R$ 8.000/mês'      THEN  7000.0
            WHEN faixa_salarial = 'de R$ 4.001/mês a R$ 6.000/mês'      THEN  5000.0
            WHEN faixa_salarial = 'de R$ 3.001/mês a R$ 4.000/mês'      THEN  3500.0
            WHEN faixa_salarial = 'de R$ 2.001/mês a R$ 3.000/mês'      THEN  2500.0
            WHEN faixa_salarial = 'de R$ 1.001/mês a R$ 2.000/mês'      THEN  1500.0
            WHEN faixa_salarial = 'Menos de R$ 1.000/mês'               THEN   500.0
            ELSE 0.0
        END AS salario_numerico
    FROM read_csv_auto('{csv_path}');
    """

    db_uri = f'duckdb:///{db_file}' if db_file != ':memory:' else 'duckdb:///:memory:'
    db = SQLDatabase.from_uri(db_uri)
    db.run(create_table_sql)