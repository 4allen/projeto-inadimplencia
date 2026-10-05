"""Análise completa do exercício de crédito (Partes 1–4 + bônus)."""
from __future__ import annotations

import json
from copy import deepcopy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate

from config import DADOS, GRAFICOS, MODELOS, PROBLEMAS, RELATORIOS, SEMENTE
from ml_utils import criar_pipeline, custo_credito, dividir


def taxa_por_score(df):
    bins = [300, 500, 600, 700, 800, 1000]
    labels = ["300–499", "500–599", "600–699", "700–799", "800–1000"]
    grupos = pd.cut(df["score_credito"], bins=bins, labels=labels, include_lowest=True, right=False)
    return df.groupby(grupos, observed=False)["inadimplente"].mean()


def salvar_graficos(df, modelo, X_teste, y_teste):
    GRAFICOS.mkdir(parents=True, exist_ok=True)

    # 1. Proporção de inadimplentes
    prop = df["inadimplente"].value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Pagou (0)", "Inadimplente (1)"], prop.values)
    ax.set_ylabel("Proporção")
    ax.set_title("Proporção de contratos por classe")
    ax.set_ylim(0, 1)
    for i, v in enumerate(prop.values):
        ax.text(i, v + .02, f"{v:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(GRAFICOS / "01_proporcao_inadimplencia.png", dpi=160)
    plt.close(fig)

    # 2. Inadimplência por faixa de score
    score = taxa_por_score(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(score.index.astype(str), score.values)
    ax.set_ylabel("Taxa de inadimplência")
    ax.set_xlabel("Faixa de score")
    ax.set_title("Inadimplência por faixa de score de crédito")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(GRAFICOS / "02_inadimplencia_score.png", dpi=160)
    plt.close(fig)

    # 3. Finalidade
    finalidade = df.groupby("finalidade")["inadimplente"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(finalidade.index, finalidade.values)
    ax.set_ylabel("Taxa de inadimplência")
    ax.set_title("Inadimplência por finalidade do empréstimo")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(GRAFICOS / "03_inadimplencia_finalidade.png", dpi=160)
    plt.close(fig)

    # 4. Posse de imóvel
    imovel = df.groupby("possui_imovel")["inadimplente"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(imovel.index, imovel.values)
    ax.set_ylabel("Taxa de inadimplência")
    ax.set_title("Inadimplência e posse de imóvel")
    fig.tight_layout()
    fig.savefig(GRAFICOS / "04_inadimplencia_imovel.png", dpi=160)
    plt.close(fig)

    # 5. Ausentes
    aus = df.isna().sum()
    aus = aus[aus > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(aus.index, aus.values)
    ax.set_ylabel("Quantidade de ausentes")
    ax.set_title("Valores ausentes no conjunto de crédito")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(GRAFICOS / "05_valores_ausentes.png", dpi=160)
    plt.close(fig)

    # 6. ROC
    prob = modelo.predict_proba(X_teste)[:, 1]
    fpr, tpr, _ = roc_curve(y_teste, prob)
    auc = roc_auc_score(y_teste, prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Modelo (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title("Curva ROC — crédito")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAFICOS / "06_curva_roc.png", dpi=160)
    plt.close(fig)


def bonus_comprometimento(df, problema):
    df2 = df.copy()
    # Parcela estimada didática: valor * 1,33 / prazo, mesma aproximação usada na geração dos dados.
    df2["comprometimento_renda"] = (
        df2["valor_emprestimo"] * 1.33 / df2["prazo_meses"] / df2["renda_mensal"]
    )
    base_features = list(problema["features"])
    features_bonus = deepcopy(problema["features"])
    features_bonus["comprometimento_renda"] = {"tipo": "numero", "padrao": .3, "min": 0, "max": 3}

    X_base = df2[base_features]
    X_bonus = df2[list(features_bonus)]
    y = df2[problema["alvo"]]

    # Mesma divisão para os dois experimentos.
    from sklearn.model_selection import train_test_split
    idx_train, _ = train_test_split(
        np.arange(len(df2)), test_size=.2, random_state=SEMENTE, stratify=y
    )
    modelo = LogisticRegression(max_iter=1500, random_state=SEMENTE)
    p_base = criar_pipeline(problema, modelo)
    p_bonus = criar_pipeline(problema, LogisticRegression(max_iter=1500, random_state=SEMENTE), features_override=features_bonus)
    s_base = cross_validate(p_base, X_base.iloc[idx_train], y.iloc[idx_train], cv=5, scoring="roc_auc", n_jobs=-1)["test_score"]
    s_bonus = cross_validate(p_bonus, X_bonus.iloc[idx_train], y.iloc[idx_train], cv=5, scoring="roc_auc", n_jobs=-1)["test_score"]
    return {
        "roc_auc_base_media": float(s_base.mean()),
        "roc_auc_base_desvio": float(s_base.std()),
        "roc_auc_com_comprometimento_media": float(s_bonus.mean()),
        "roc_auc_com_comprometimento_desvio": float(s_bonus.std()),
        "delta": float(s_bonus.mean() - s_base.mean()),
    }


def bonus_grid_e_balanceado(X_treino, y_treino, problema):
    # GridSearchCV compacto para manter a execução viável em notebooks comuns.
    rf = RandomForestClassifier(random_state=SEMENTE, n_jobs=-1)
    pipe = criar_pipeline(problema, rf)
    grid = GridSearchCV(
        pipe,
        {
            "modelo__n_estimators": [120, 200],
            "modelo__max_depth": [6, 10],
            "modelo__min_samples_leaf": [3, 5],
        },
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(X_treino, y_treino)

    # class_weight balanced: comparação por previsões OOF do treino, sem usar o teste para escolher.
    normal = criar_pipeline(problema, LogisticRegression(max_iter=1500, random_state=SEMENTE))
    balanced = criar_pipeline(
        problema,
        LogisticRegression(max_iter=1500, random_state=SEMENTE, class_weight="balanced"),
    )
    prob_n = cross_val_predict(normal, X_treino, y_treino, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    prob_b = cross_val_predict(balanced, X_treino, y_treino, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    pred_n = (prob_n >= .5).astype(int)
    pred_b = (prob_b >= .5).astype(int)
    return {
        "gridsearch": {
            "melhor_roc_auc_cv": float(grid.best_score_),
            "melhores_parametros": grid.best_params_,
        },
        "class_weight": {
            "normal": {
                "precisao": float(precision_score(y_treino, pred_n, zero_division=0)),
                "recall": float(recall_score(y_treino, pred_n, zero_division=0)),
            },
            "balanced": {
                "precisao": float(precision_score(y_treino, pred_b, zero_division=0)),
                "recall": float(recall_score(y_treino, pred_b, zero_division=0)),
            },
        },
    }


def main():
    problema = PROBLEMAS["credito"]
    df = pd.read_csv(DADOS / problema["arquivo"])
    metadata = json.loads((MODELOS / "credito_metadata.json").read_text(encoding="utf-8"))
    modelo = joblib.load(MODELOS / "credito.joblib")
    X_treino, X_teste, y_treino, y_teste = dividir(problema)

    salvar_graficos(df, modelo, X_teste, y_teste)

    ausentes = df.isna().sum()
    ausentes = {k: int(v) for k, v in ausentes.items() if v > 0}
    score = {str(k): float(v) for k, v in taxa_por_score(df).items()}
    finalidade = {k: float(v) for k, v in df.groupby("finalidade")["inadimplente"].mean().items()}
    imovel = {k: float(v) for k, v in df.groupby("possui_imovel")["inadimplente"].mean().items()}

    bonus = bonus_comprometimento(df, problema)
    bonus.update(bonus_grid_e_balanceado(X_treino, y_treino, problema))

    relatorio = {
        "linhas": len(df),
        "proporcao_inadimplentes": float(df["inadimplente"].mean()),
        "ausentes": ausentes,
        "inadimplencia_por_score": score,
        "inadimplencia_por_finalidade": finalidade,
        "inadimplencia_por_imovel": imovel,
        "coluna_descartada": "id_contrato",
        "justificativa_descarte": "Identificador único, sem significado causal/preditivo; favorece memorização e não generaliza.",
        "modelo_final": metadata,
        "bonus": bonus,
        "etica": (
            "Modelos de crédito podem reproduzir discriminações históricas presentes nos dados. Mesmo que aumentassem a métrica, "
            "não usaríamos atributos sensíveis como raça/cor, religião, orientação sexual, opinião política, saúde, origem étnica ou "
            "outros dados pessoais sensíveis, nem proxies evidentes desses atributos. A LGPD exige finalidade, necessidade, transparência, "
            "segurança e tratamento adequado de dados pessoais; decisões automatizadas que afetem o titular devem ser governadas com "
            "explicabilidade, possibilidade de revisão e controles contra vieses e uso excessivo de dados."
        ),
    }
    RELATORIOS.mkdir(exist_ok=True)
    (RELATORIOS / "analise_credito.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    cv_rows = [
        {"algoritmo": k, **v} for k, v in metadata["validacao_cruzada"].items()
    ]
    pd.DataFrame(cv_rows).to_csv(RELATORIOS / "comparacao_modelos_credito.csv", index=False)
    pd.DataFrame(metadata["analise_limiar_treino"]).to_csv(
        RELATORIOS / "analise_limiares_credito.csv", index=False
    )

    print("Análise concluída. Consulte reports/ e docs/graficos/.")


if __name__ == "__main__":
    main()
