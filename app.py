"""API Flask e interface web. Uso: python app.py -> http://localhost:5000"""
from __future__ import annotations

import json
import subprocess
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from config import DADOS, MODELOS, PASTA, PROBLEMAS

app = Flask(__name__)


def garantir_artefatos():
    if not (DADOS / "credito.csv").exists():
        subprocess.run([sys.executable, str(PASTA / "gerar_dados.py")], cwd=PASTA, check=True)
    if not all((MODELOS / f"{nome}.joblib").exists() for nome in PROBLEMAS):
        subprocess.run([sys.executable, str(PASTA / "treinar.py")], cwd=PASTA, check=True)


garantir_artefatos()
MODELOS_CARREGADOS = {nome: joblib.load(MODELOS / f"{nome}.joblib") for nome in PROBLEMAS}
METRICAS = json.loads((MODELOS / "metricas.json").read_text(encoding="utf-8"))


def validar(nome, dados):
    problema = PROBLEMAS[nome]
    linha = {}
    erros = []
    for campo, cfg in problema["features"].items():
        valor = dados.get(campo, "")
        if valor in ("", None):
            linha[campo] = None
            continue
        if cfg["tipo"] == "numero":
            try:
                numero = float(valor)
            except (TypeError, ValueError):
                erros.append(f"{campo}: valor numérico inválido")
                continue
            if numero < cfg["min"] or numero > cfg["max"]:
                erros.append(f"{campo}: deve estar entre {cfg['min']} e {cfg['max']}")
            linha[campo] = numero
        else:
            if valor not in cfg["opcoes"]:
                erros.append(f"{campo}: escolha uma opção válida")
            linha[campo] = valor
    return linha, erros


@app.get("/")
def pagina():
    return render_template("index.html")


@app.get("/api/problemas")
def problemas():
    return jsonify({
        nome: {
            **problema,
            "metricas": METRICAS.get(nome, {}),
        }
        for nome, problema in PROBLEMAS.items()
    })


@app.post("/api/prever/<nome>")
def prever(nome):
    if nome not in PROBLEMAS:
        return jsonify({"erro": "Problema não encontrado"}), 404
    dados = request.get_json(silent=True) or {}
    linha, erros = validar(nome, dados)
    if erros:
        return jsonify({"erros": erros}), 400

    problema = PROBLEMAS[nome]
    X = pd.DataFrame([linha], columns=list(problema["features"]))
    modelo = MODELOS_CARREGADOS[nome]

    if problema["tipo"] == "classificacao":
        prob = float(modelo.predict_proba(X)[0, 1])
        limiar = float(METRICAS[nome].get("limiar") or .5)
        classe = int(prob >= limiar)
        if nome == "credito":
            rotulo = "Risco alto — recomendação: recusar/revisar" if classe else "Risco baixo — recomendação: aprovar"
        else:
            rotulo = "Sim" if classe else "Não"
        return jsonify({
            "valor": prob,
            "percentual": prob * 100,
            "classe": classe,
            "rotulo": rotulo,
            "limiar": limiar,
            "resultado": problema["resultado"],
        })

    valor = float(modelo.predict(X)[0])
    return jsonify({"valor": valor, "resultado": problema["resultado"]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
