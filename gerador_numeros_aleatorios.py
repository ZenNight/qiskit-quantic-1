"""Mini projeto 1 - Gerador de números aleatórios com circuitos quânticos.

Gera inteiros em [0, 1000] a partir de cadeias de bits medidas em circuitos
quânticos de 10 qubits (2**10 = 1024 >= 1001), em dois cenários:

1. Uniforme:    todos os valores têm a mesma probabilidade.
2. Tendencioso: a distribuição favorece valores mais altos.

Como 1001 não é potência de 2, os 23 valores 1001..1023 são descartados e o
circuito é executado de novo (amostragem por rejeição). Isso mantém a
distribuição uniforme no cenário 1 e só trunca a cauda no cenário 2.

Convenção do Qiskit: o qubit i é o bit de peso 2**i (qubit 0 = bit menos
significativo). Por isso int(bitstring, 2) dá o valor diretamente.

Uso:
    python gerador_numeros_aleatorios.py [--n 100000] [--seed 42] [--saida resultados]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from scipy import stats

VALOR_MAX = 1000
N_QUBITS = int(np.ceil(np.log2(VALOR_MAX + 1)))  # 10

# Tendência do cenário 2 (definida pelo grupo): P(bit = 1) cresce linearmente
# do bit menos significativo (50%) ao mais significativo (90%).
P_BIT_MENOS_SIGNIFICATIVO = 0.5
P_BIT_MAIS_SIGNIFICATIVO = 0.9


# --------------------------------------------------------------------------
# Circuitos
# --------------------------------------------------------------------------
def circuito_uniforme(n_qubits: int = N_QUBITS) -> QuantumCircuit:
    """H em todos os qubits: cada bit vale 0 ou 1 com probabilidade 1/2."""
    qc = QuantumCircuit(n_qubits, n_qubits, name="uniforme")
    qc.h(range(n_qubits))
    qc.barrier()
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def probabilidades_por_bit(
    n_qubits: int = N_QUBITS,
    p_lsb: float = P_BIT_MENOS_SIGNIFICATIVO,
    p_msb: float = P_BIT_MAIS_SIGNIFICATIVO,
) -> np.ndarray:
    """P(bit i = 1) para cada qubit i (i = 0 é o bit menos significativo)."""
    return np.linspace(p_lsb, p_msb, n_qubits)


def circuito_tendencioso(
    n_qubits: int = N_QUBITS,
    p_lsb: float = P_BIT_MENOS_SIGNIFICATIVO,
    p_msb: float = P_BIT_MAIS_SIGNIFICATIVO,
) -> QuantumCircuit:
    """H -> Rz(theta) -> H em cada qubit, com theta diferente por qubit.

    Rz só muda a fase relativa, então sozinha (ou logo após o primeiro H) não
    altera a probabilidade de medição. O segundo H converte a fase em
    amplitude: P(1) = sin^2(theta/2), logo theta = 2*arcsin(sqrt(p)).
    Com p > 0.5 o bit tende a 1, e dar p maior aos bits mais significativos
    empurra o número para valores altos.
    """
    qc = QuantumCircuit(n_qubits, n_qubits, name="tendencioso")
    for i, p in enumerate(probabilidades_por_bit(n_qubits, p_lsb, p_msb)):
        theta = 2 * np.arcsin(np.sqrt(p))
        qc.h(i)
        qc.rz(theta, i)
        qc.h(i)
    qc.barrier()
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# --------------------------------------------------------------------------
# Geração
# --------------------------------------------------------------------------
def distribuicao_exata(qc: QuantumCircuit) -> np.ndarray:
    """P(valor) para valor em 0..VALOR_MAX, condicionada à aceitação (<= 1000)."""
    sem_medida = qc.remove_final_measurements(inplace=False)
    probs = Statevector.from_instruction(sem_medida).probabilities()[: VALOR_MAX + 1]
    return probs / probs.sum()


def gerar_numeros(qc: QuantumCircuit, quantidade: int, seed: int | None = None) -> np.ndarray:
    """Executa o circuito (1 shot = 1 cadeia de bits) até ter `quantidade` inteiros em [0, 1000]."""
    sampler = StatevectorSampler(seed=seed)
    aceitos: list[int] = []
    while len(aceitos) < quantidade:
        faltam = quantidade - len(aceitos)
        shots = int(faltam * 1.3) + 64  # margem para as rejeições
        bitstrings = sampler.run([qc], shots=shots).result()[0].data.c.get_bitstrings()
        aceitos.extend(v for v in (int(b, 2) for b in bitstrings) if v <= VALOR_MAX)
    return np.array(aceitos[:quantidade])


def gerar_numero(qc: QuantumCircuit, seed: int | None = None) -> int:
    """Um único número aleatório (um shot por tentativa)."""
    return int(gerar_numeros(qc, 1, seed)[0])


# --------------------------------------------------------------------------
# Análise
# --------------------------------------------------------------------------
def probabilidade_por_bit(p_valores: np.ndarray, n_qubits: int = N_QUBITS) -> np.ndarray:
    """P(bit i = 1) implícita em uma distribuição sobre 0..VALOR_MAX."""
    valores = np.arange(len(p_valores))
    return np.array([p_valores[(valores >> i) & 1 == 1].sum() for i in range(n_qubits)])


def agrupar(contagem_ou_prob: np.ndarray, n_grupos: int) -> np.ndarray:
    """Soma valores consecutivos em `n_grupos` faixas de tamanho ~igual."""
    limites = np.linspace(0, len(contagem_ou_prob), n_grupos + 1).astype(int)
    return np.add.reduceat(contagem_ou_prob, limites[:-1])


def analisar(nome: str, numeros: np.ndarray, p_exata: np.ndarray, n_grupos: int) -> dict:
    """Estatísticas descritivas e teste qui-quadrado contra a distribuição teórica."""
    n = len(numeros)
    valores = np.arange(VALOR_MAX + 1)
    obs = agrupar(np.bincount(numeros, minlength=VALOR_MAX + 1), n_grupos)
    esp = agrupar(p_exata * n, n_grupos)
    chi2, p_valor = stats.chisquare(obs, esp)

    media_teorica = float((valores * p_exata).sum())
    print(f"\n=== {nome} ({n:,} números) ===".replace(",", "."))
    print(f"mín / máx            : {numeros.min()} / {numeros.max()}")
    print(f"média (obs | teórica): {numeros.mean():.1f} | {media_teorica:.1f}")
    print(f"mediana              : {np.median(numeros):.0f}")
    print(f"desvio padrão        : {numeros.std():.1f}")
    print(f"P(número > 500)      : {(numeros > 500).mean():.3f}")
    print(
        f"qui-quadrado vs teoria ({n_grupos} faixas, gl={n_grupos - 1}): "
        f"chi2={chi2:.1f}, p={p_valor:.3f} -> "
        f"{'compatível' if p_valor > 0.05 else 'INCOMPATÍVEL'} com a distribuição teórica (alfa=0.05)"
    )
    return {"obs": obs, "esp": esp, "p_valor": p_valor}


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------
def plotar_distribuicoes(resultados: dict, n_grupos: int, caminho: Path) -> None:
    fig, eixos = plt.subplots(1, 3, figsize=(17, 4.8))
    limites = np.linspace(0, VALOR_MAX + 1, n_grupos + 1)
    centros = (limites[:-1] + limites[1:]) / 2
    largura = limites[1] - limites[0]
    cores = {"Uniforme": "#2a78c2", "Tendencioso": "#d9622b"}

    for eixo, (nome, r) in zip(eixos[:2], resultados.items()):
        n = r["numeros"].size
        eixo.bar(centros, r["obs_grupos"] / n, width=largura * 0.92, color=cores[nome],
                 alpha=0.85, label="medido")
        eixo.plot(centros, r["esp_grupos"] / n, "k--o", ms=4, lw=1.2, label="teórico")
        eixo.set_title(f"Cenário: {nome}")
        eixo.set_xlabel(f"valor (faixas de ~{VALOR_MAX // n_grupos})")
        eixo.set_ylabel("frequência relativa")
        eixo.legend()

    # Probabilidade de cada qubit ser 1 (mostra o efeito direto do Rz).
    eixo = eixos[2]
    i = np.arange(N_QUBITS)
    for k, (nome, r) in enumerate(resultados.items()):
        medido = np.array([((r["numeros"] >> b) & 1).mean() for b in i])
        eixo.bar(i + (k - 0.5) * 0.38, medido, width=0.36, color=cores[nome], alpha=0.85, label=nome)
        eixo.plot(i + (k - 0.5) * 0.38, r["p_bit_teorico"], "k_", ms=12, mew=2)
    eixo.axhline(0.5, color="gray", lw=0.8, ls=":")
    eixo.set_xticks(i)
    eixo.set_xlabel("qubit i (peso 2^i)")
    eixo.set_ylabel("P(bit = 1)")
    eixo.set_ylim(0, 1)
    eixo.set_title("Probabilidade de cada bit ser 1 (traço preto = teórico)")
    eixo.legend()

    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


def salvar_circuito(qc: QuantumCircuit, caminho: Path) -> None:
    try:
        fig = qc.draw("mpl", fold=-1)
        fig.savefig(caminho, dpi=140, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # o desenho é opcional; não deve derrubar o experimento
        print(f"(não foi possível desenhar {qc.name} em PNG: {exc}) -- versão texto:")
        print(qc.draw("text", fold=-1))


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=100_000, help="quantidade de números por cenário")
    parser.add_argument("--seed", type=int, default=42, help="semente do simulador (reprodutibilidade)")
    parser.add_argument("--saida", type=Path, default=Path("resultados"), help="pasta dos gráficos")
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    cenarios = {
        "Uniforme": (circuito_uniforme(), 1001),  # 1 faixa por valor (esperado ~100 por valor)
        "Tendencioso": (circuito_tendencioso(), 20),  # valores raros exigem faixas maiores
    }

    print(f"Qubits: {N_QUBITS} | intervalo: 0..{VALOR_MAX} | rejeição de valores > {VALOR_MAX}")
    print("P(bit=1) por qubit no cenário 2:", np.round(probabilidades_por_bit(), 3))

    resultados = {}
    n_grupos_grafico = 20
    for nome, (qc, n_grupos_teste) in cenarios.items():
        p_exata = distribuicao_exata(qc)
        numeros = gerar_numeros(qc, args.n, seed=args.seed)
        analisar(nome, numeros, p_exata, n_grupos_teste)
        print("primeiros 10 números :", numeros[:10].tolist())
        resultados[nome] = {
            "numeros": numeros,
            "obs_grupos": agrupar(np.bincount(numeros, minlength=VALOR_MAX + 1), n_grupos_grafico),
            "esp_grupos": agrupar(p_exata * len(numeros), n_grupos_grafico),
            "p_bit_teorico": probabilidade_por_bit(p_exata),
        }
        salvar_circuito(qc, args.saida / f"circuito_{qc.name}.png")

    plotar_distribuicoes(resultados, n_grupos_grafico, args.saida / "distribuicoes.png")
    print(f"\nGráficos salvos em: {args.saida.resolve()}")


if __name__ == "__main__":
    main()
