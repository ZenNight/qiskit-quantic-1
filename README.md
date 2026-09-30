# qiskit-quantic-1

## Mini projeto 1 — Gerador de números aleatórios

Gera inteiros em **[0, 1000]** medindo circuitos quânticos de 10 qubits (2¹⁰ = 1024 ≥ 1001),
em dois cenários. Código em [gerador_numeros_aleatorios.py](gerador_numeros_aleatorios.py).

```bash
pip install -r requirements.txt
python gerador_numeros_aleatorios.py            # 100 mil números por cenário
python gerador_numeros_aleatorios.py --n 20000 --seed 7
```

Saída: estatísticas no terminal e, em `resultados/`, `distribuicoes.png`,
`circuito_uniforme.png` e `circuito_tendencioso.png`.

### Cenário 1 — Uniforme
`H` em todos os qubits, seguido de medição. Cada bit é 0 ou 1 com probabilidade ½, então
cada uma das 1024 cadeias tem probabilidade 1/1024.

### Cenário 2 — Tendencioso (definido pelo grupo)
Em cada qubit `i`: **H → Rz(θᵢ) → H**, com θᵢ = 2·arcsin(√pᵢ), o que dá P(bit i = 1) = pᵢ.
Escolhemos pᵢ crescendo linearmente de **50 % (qubit 0, bit menos significativo)** a
**90 % (qubit 9, bit mais significativo)**. Resultado: média ≈ 817 e ≈ 89 % dos números
acima de 500. Para mudar a tendência, altere `P_BIT_MENOS_SIGNIFICATIVO` e
`P_BIT_MAIS_SIGNIFICATIVO`.

> **Sobre a dica de usar S, T ou Rz.** Essas portas só alteram a *fase*. Em `H → Rz`, a
> probabilidade de medir 1 continua 0,5 (verificado numericamente); por isso o segundo `H`
> é necessário, pois ele transforma fase em probabilidade: P(1) = sin²(θ/2).
> Atenção ao ângulo: `T` = Rz(π/4) dá P(1) ≈ 0,146 (favorece o **0**); `S` = Rz(π/2) dá 0,5;
> `S·T` = Rz(3π/4) dá ≈ 0,854. Para favorecer o 1 é preciso θ entre π/2 e 3π/2.

### Por que 10 qubits e rejeição
1001 valores não cabem exatamente em 2ⁿ. Com 10 qubits, as cadeias que resultam em
1001–1023 são **descartadas e o circuito é executado de novo**. Isso preserva a uniformidade
do cenário 1 (truncar ou usar `% 1001` a quebraria) e, no cenário 2, apenas corta a cauda.

### Verificação
Para cada cenário, os números medidos são comparados (qui-quadrado) com a distribuição
**exata** calculada pelo vetor de estado do circuito:

| Cenário | Teste | 10 sementes diferentes |
|---|---|---|
| Uniforme | 1001 faixas (uma por valor), gl = 1000 | p-valores 0,03–0,91 |
| Tendencioso | 20 faixas de ~50 valores, gl = 19 | p-valores 0,08–0,96 |

Com alfa = 0,05 espera-se ~1 rejeição a cada 20 testes por acaso; observamos 1 em 20
(uniforme, semente 10, p = 0,03).

### Limitação importante
A execução usa o `StatevectorSampler` do Qiskit, um **simulador**: as amostras vêm do gerador
pseudoaleatório do NumPy (por isso a `--seed` reproduz os resultados). A aleatoriedade
*genuína* citada no enunciado só existe ao rodar os mesmos circuitos em hardware quântico
real, trocando o sampler por um primitivo de backend real (ex.: IBM Quantum).
