"""Gera dados sintéticos com padrões de negócio. Uso: python gerar_dados.py"""
import numpy as np
import pandas as pd

from config import DADOS, SEMENTE

rng = np.random.default_rng(SEMENTE)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def escolher(opcoes, pesos, n):
    return rng.choice(opcoes, n, p=pesos)


def apagar(df, coluna, fracao):
    """Simula valores ausentes, comuns em dados reais."""
    df.loc[rng.random(len(df)) < fracao, coluna] = np.nan


def gerar_churn(n=5000):
    d = pd.DataFrame({"id_cliente": [f"C{i:05d}" for i in range(1, n + 1)]})
    d["idade"] = np.clip(rng.normal(42, 13, n), 18, 85).round()
    d["meses_contrato"] = np.clip(rng.gamma(1.4, 17, n), 0, 72).round()
    d["tipo_contrato"] = escolher(["mensal", "anual", "bianual"], [.55, .25, .20], n)
    d["pagamento"] = escolher(["boleto", "cartao", "debito_automatico", "pix"], [.25, .30, .20, .25], n)
    d["internet"] = escolher(["fibra", "dsl", "sem_internet"], [.50, .35, .15], n)
    d["streaming"] = escolher(["sim", "nao"], [.40, .60], n)
    fibra = d.internet == "fibra"
    base = np.select([fibra, d.internet == "dsl"], [120, 80], 45)
    d["mensalidade"] = np.clip(base + 35 * (d.streaming == "sim") + rng.normal(0, 15, n), 20, 350).round(2)
    d["chamados_suporte"] = rng.poisson(1 + .8 * fibra)
    d["atrasos_pagamento"] = rng.poisson(.5 + .6 * (d.pagamento == "boleto"))
    z = (
        -1.9 + 1.3 * (d.tipo_contrato == "mensal") - .9 * (d.tipo_contrato == "bianual")
        - .045 * d.meses_contrato + .38 * d.chamados_suporte + .012 * (d.mensalidade - 100)
        + .35 * (d.pagamento == "boleto") - .45 * (d.pagamento == "debito_automatico")
        + .45 * d.atrasos_pagamento + .3 * fibra - .012 * (d.idade - 42) + rng.normal(0, .6, n)
    )
    d["cancelou"] = (rng.random(n) < sigmoide(z)).astype(int)
    apagar(d, "idade", .03)
    apagar(d, "chamados_suporte", .02)
    return d


def gerar_imoveis(n=4000):
    d = pd.DataFrame({"id_imovel": [f"I{i:05d}" for i in range(1, n + 1)]})
    d["tipo"] = escolher(["apartamento", "casa"], [.65, .35], n)
    d["bairro"] = escolher(["Centro", "Jardins", "Vila Nova", "Beira-Mar", "Industrial"], [.25, .18, .27, .12, .18], n)
    casa = d.tipo == "casa"
    d["area_m2"] = np.clip(np.where(casa, 140, 70) * rng.lognormal(0, .35, n), 25, 700).round()
    d["quartos"] = np.clip(np.round(d.area_m2 / 38 + rng.normal(0, .7, n)), 1, 7)
    d["banheiros"] = np.clip(np.round(d.quartos * .7 + rng.normal(0, .5, n)), 1, 6)
    d["vagas"] = np.clip(np.round(d.area_m2 / 70 + rng.normal(0, .6, n)), 0, 5)
    d["idade_imovel"] = rng.integers(0, 51, n)
    d["distancia_metro_km"] = np.clip(rng.exponential(2.2, n), .1, 20).round(1)
    nobre = d.bairro.isin(["Jardins", "Beira-Mar"])
    d["piscina"] = np.where(rng.random(n) < .08 + .25 * casa + .2 * nobre, "sim", "nao")
    m2 = d.bairro.map({"Centro": 7000, "Jardins": 11000, "Vila Nova": 5000, "Beira-Mar": 13000, "Industrial": 3500})
    preco = (
        d.area_m2 * m2 * (1 - .007 * d.idade_imovel) * (1 + .07 * d.vagas)
        * np.exp(-.05 * d.distancia_metro_km) * np.where(d.piscina == "sim", 1.12, 1)
        * np.where(casa, .92, 1) * rng.lognormal(0, .12, n)
    )
    d["preco"] = (preco / 1000).round() * 1000
    apagar(d, "vagas", .02)
    apagar(d, "distancia_metro_km", .03)
    return d


def gerar_credito(n=6000):
    """Dados do exercício: inadimplência em empréstimos."""
    d = pd.DataFrame({"id_contrato": [f"E{i:05d}" for i in range(1, n + 1)]})
    d["idade"] = np.clip(rng.normal(40, 12, n), 18, 80).round()
    d["renda_mensal"] = np.clip(4200 * rng.lognormal(0, .6, n), 1300, 60000).round(-1)
    d["tempo_emprego_anos"] = np.clip(rng.gamma(1.6, 3.5, n), 0, 40).round(1)
    d["score_credito"] = np.clip(rng.normal(640, 110, n), 300, 1000).round()
    d["dividas_ativas"] = rng.poisson(1.1, n)
    d["possui_imovel"] = escolher(["sim", "nao"], [.4, .6], n)
    d["finalidade"] = escolher(["pessoal", "veiculo", "reforma", "educacao", "negocio"], [.35, .25, .15, .10, .15], n)
    d["prazo_meses"] = escolher([12, 24, 36, 48, 60], [.15, .30, .25, .15, .15], n)
    d["valor_emprestimo"] = np.clip(d.renda_mensal * rng.uniform(.5, 6, n), 1000, 200000).round(-2)

    comprometimento = d.valor_emprestimo * 1.33 / d.prazo_meses / d.renda_mensal
    z = (
        -2.2 + 2.8 * np.clip(comprometimento, 0, 1.5) - .009 * (d.score_credito - 640)
        + .4 * d.dividas_ativas - .06 * d.tempo_emprego_anos - .35 * (d.possui_imovel == "sim")
        + .45 * (d.finalidade == "negocio") - .015 * (d.idade - 40) + rng.normal(0, .5, n)
    )
    d["inadimplente"] = (rng.random(n) < sigmoide(z)).astype(int)
    apagar(d, "renda_mensal", .04)
    apagar(d, "tempo_emprego_anos", .03)
    return d


def main():
    DADOS.mkdir(exist_ok=True)
    for nome, gerar in [("churn", gerar_churn), ("imoveis", gerar_imoveis), ("credito", gerar_credito)]:
        df = gerar()
        df.to_csv(DADOS / f"{nome}.csv", index=False)
        print(f"data/{nome}.csv: {len(df)} linhas")
    credito = pd.read_csv(DADOS / "credito.csv")
    print(f"Inadimplentes: {credito['inadimplente'].mean():.1%}")


if __name__ == "__main__":
    main()
