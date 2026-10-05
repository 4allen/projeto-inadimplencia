"""Compara algoritmos, avalia no teste e salva modelos. Uso: python treinar.py"""
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_validate

from config import MODELOS, PROBLEMAS, RELATORIOS, SEMENTE
from ml_utils import (
    avaliar_classificacao,
    avaliar_regressao,
    candidatos,
    criar_pipeline,
    custo_credito,
    dividir,
)

LIMIARES_CREDITO = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def treinar(nome, problema):
    print(f"\n=== {problema['titulo']} ===")
    X_treino, X_teste, y_treino, y_teste = dividir(problema)
    classificacao = problema["tipo"] == "classificacao"
    metrica = "roc_auc" if classificacao else "neg_root_mean_squared_error"

    comparacao = {}
    for algoritmo, modelo_base in candidatos(problema["tipo"]).items():
        pipeline = criar_pipeline(problema, modelo_base)
        cv = cross_validate(pipeline, X_treino, y_treino, cv=5, scoring=metrica, n_jobs=-1)
        notas = cv["test_score"] if classificacao else -cv["test_score"]
        comparacao[algoritmo] = {"media": float(notas.mean()), "desvio": float(notas.std())}
        print(f"{algoritmo:<22} {notas.mean():.4f} ± {notas.std():.4f}")

    escolhido = (max if classificacao else min)(comparacao, key=lambda a: comparacao[a]["media"])
    pipeline = criar_pipeline(problema, candidatos(problema["tipo"])[escolhido])

    limiar = .5
    analise_limiar = None
    if nome == "credito":
        # Escolha do limiar exclusivamente com previsões out-of-fold do TREINO.
        prob_cv = cross_val_predict(
            pipeline, X_treino, y_treino, cv=5, method="predict_proba", n_jobs=-1
        )[:, 1]
        linhas = [custo_credito(y_treino, prob_cv, t) for t in LIMIARES_CREDITO]
        analise_limiar = linhas
        limiar = min(linhas, key=lambda r: r["custo_total"])["limiar"]
        print(f"Limiar escolhido pelo custo no treino: {limiar:.2f}")

    pipeline.fit(X_treino, y_treino)
    teste = (
        avaliar_classificacao(pipeline, X_teste, y_teste, limiar=limiar)
        if classificacao else avaliar_regressao(pipeline, X_teste, y_teste)
    )

    if nome == "credito":
        prob_teste = pipeline.predict_proba(X_teste)[:, 1]
        custo_teste = custo_credito(y_teste, prob_teste, limiar)
        teste["custo_total"] = custo_teste["custo_total"]
        teste["fp"] = custo_teste["fp"]
        teste["fn"] = custo_teste["fn"]
        teste["limiar"] = limiar

    MODELOS.mkdir(exist_ok=True)
    RELATORIOS.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELOS / f"{nome}.joblib", compress=3)

    resultado = {
        "modelo": escolhido,
        "metrica_cv": "ROC AUC" if classificacao else "RMSE",
        "validacao_cruzada": comparacao,
        "teste": teste,
        "limiar": limiar if classificacao else None,
        "analise_limiar_treino": analise_limiar,
        "sklearn": sklearn.__version__,
        "random_state": SEMENTE,
    }
    (MODELOS / f"{nome}_metadata.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return resultado


def main():
    MODELOS.mkdir(exist_ok=True)
    metricas = {nome: treinar(nome, problema) for nome, problema in PROBLEMAS.items()}
    (MODELOS / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nModelos e métricas salvos em models/.")


if __name__ == "__main__":
    main()
