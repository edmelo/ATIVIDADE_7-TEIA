import random
import statistics
import time
import os
from typing import List, Tuple, Dict, Any

import matplotlib
matplotlib.use('Agg')  # garante salvamento de figuras sem backend interativo
import matplotlib.pyplot as plt

from usa13 import USA13, route_distance, is_valid_route


# ========================
# Configuração do AG (padrão exigido)
# ========================
POP_SIZE = 50
GENERATIONS = 400
TOURNEY_SIZE = 3
CROSSOVER_RATE = 0.90  # OX
MUTATION_RATE = 0.05   # Swap
ELITISM_K = 5
RUNS = 30

N_CITIES = len(USA13)
ALL_CITIES = list(range(N_CITIES))


# ========================
# Operadores do AG
# ========================

def random_individual() -> List[int]:
    ind = ALL_CITIES.copy()
    random.shuffle(ind)
    return ind


def evaluate(ind: List[int]) -> float:
    """Fitness = distância total (minimizar)."""
    return route_distance(ind, USA13)


def tournament_selection(pop: List[List[int]], fitness: List[float], k: int = TOURNEY_SIZE) -> List[int]:
    idxs = random.sample(range(len(pop)), k)
    best_idx = min(idxs, key=lambda i: fitness[i])
    return pop[best_idx][:]


def ox_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Order Crossover (OX) clássico para permutações."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))

    def ox(parent_a, parent_b):
        child = [None] * n
        # copia segmento [a:b]
        child[a:b + 1] = parent_a[a:b + 1]
        # preenche restantes na ordem do parent_b
        fill_vals = [g for g in parent_b if g not in child]
        fill_it = iter(fill_vals)
        for i in list(range(b + 1, n)) + list(range(0, a)):
            child[i] = next(fill_it)
        return child

    return ox(p1, p2), ox(p2, p1)


def swap_mutation(ind: List[int], rate: float = MUTATION_RATE) -> None:
    if random.random() < rate:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]


# ========================
# Loop do AG (uma execução)
# ========================

def run_ga_once(
    seed: int | None = None,
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    tourney_size: int = TOURNEY_SIZE,
    crossover_rate: float = CROSSOVER_RATE,
    mutation_rate: float = MUTATION_RATE,
    elitism_k: int = ELITISM_K,
) -> Dict[str, Any]:
    """
    Executa uma vez o AG com parâmetros informados.

    Retorna dict com:
      - best_ind: melhor indivíduo final
      - best_fit: fitness do melhor indivíduo final (distância)
      - best_per_gen: lista do melhor fitness por geração (inclui geração 0)
      - unique_per_gen: número de indivíduos únicos por geração (inclui geração 0)
      - runtime_sec: tempo de execução em segundos
      - conv_gen_1pct: primeira geração que atinge <= 1% acima do melhor final (ou None)
      - conv_gen_best: primeira geração que atinge o melhor final exato (ou None)
    """
    if seed is not None:
        random.seed(seed)

    start_t = time.perf_counter()

    # População inicial
    population = [random_individual() for _ in range(pop_size)]
    # Garantia de validade
    assert all(is_valid_route(ind, N_CITIES) for ind in population)

    fitness = [evaluate(ind) for ind in population]

    best_per_gen: List[float] = [min(fitness)]
    # diversidade da população (indivíduos únicos)
    def pop_unique_count(pop: List[List[int]]) -> int:
        return len({tuple(ind) for ind in pop})

    unique_per_gen: List[int] = [pop_unique_count(population)]

    for _ in range(generations):
        # Elitismo
        elites_idx = sorted(range(len(population)), key=lambda i: fitness[i])[:elitism_k]
        elites = [population[i][:] for i in elites_idx]

        # Nova população
        new_pop: List[List[int]] = []
        # Geração por cruzamento + mutação
        while len(new_pop) + elitism_k < pop_size:
            p1 = tournament_selection(population, fitness, tourney_size)
            p2 = tournament_selection(population, fitness, tourney_size)
            if random.random() < crossover_rate:
                c1, c2 = ox_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            swap_mutation(c1, rate=mutation_rate)
            swap_mutation(c2, rate=mutation_rate)
            new_pop.append(c1)
            if len(new_pop) + elitism_k < pop_size:
                new_pop.append(c2)

        # Completa com elites
        new_pop.extend(elites)

        # Atualiza população e fitness
        population = new_pop
        fitness = [evaluate(ind) for ind in population]

        # Estatística de convergência
        best_per_gen.append(min(fitness))
        unique_per_gen.append(pop_unique_count(population))

    # Resultado final
    best_idx = min(range(len(population)), key=lambda i: fitness[i])
    best_ind = population[best_idx]
    best_fit = fitness[best_idx]

    # Métricas de velocidade de convergência
    eps = 0.01
    target = best_fit * (1 + eps)
    conv_gen_1pct = None
    conv_gen_best = None
    for gi, val in enumerate(best_per_gen):
        if conv_gen_1pct is None and val <= target:
            conv_gen_1pct = gi
        if conv_gen_best is None and val <= best_fit + 1e-9:
            conv_gen_best = gi
        if conv_gen_1pct is not None and conv_gen_best is not None:
            break

    runtime_sec = time.perf_counter() - start_t

    return {
        'best_ind': best_ind,
        'best_fit': best_fit,
        'best_per_gen': best_per_gen,
        'unique_per_gen': unique_per_gen,
        'runtime_sec': runtime_sec,
        'conv_gen_1pct': conv_gen_1pct,
        'conv_gen_best': conv_gen_best,
        'params': {
            'pop_size': pop_size,
            'generations': generations,
            'tourney_size': tourney_size,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'elitism_k': elitism_k,
            'seed': seed,
        }
    }


# ========================
# Execução múltipla + relatórios/gráficos (experimentos)
# ========================

from collections import defaultdict
import csv


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def align_curves(curves: List[List[float]]) -> List[List[float]]:
    max_len = max(len(c) for c in curves)
    aligned = []
    for c in curves:
        if len(c) < max_len:
            c = c + [c[-1]] * (max_len - len(c))
        aligned.append(c)
    return aligned


def agg_stats(vals: List[float]) -> dict:
    return {
        'mean': statistics.mean(vals) if vals else float('nan'),
        'stdev': statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        'min': min(vals) if vals else float('nan'),
        'max': max(vals) if vals else float('nan'),
        'median': statistics.median(vals) if vals else float('nan'),
    }


def plot_convergence(config_curves: dict, title: str, out_png: str, ylabel: str = 'Distância (menor é melhor)'):
    plt.figure(figsize=(9, 5))
    for label, curves in config_curves.items():
        aligned = align_curves(curves)
        mean_curve = [statistics.mean(x) for x in zip(*aligned)]
        plt.plot(mean_curve, label=str(label))
    plt.xlabel('Geração')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Config')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_diversity(config_curves: dict, title: str, out_png: str):
    plot_convergence(config_curves, title, out_png, ylabel='Indivíduos únicos na população')


def plot_boxplot(data_by_label: dict, title: str, out_png: str, ylabel: str = 'Distância final (menor é melhor)'):
    labels = list(map(str, data_by_label.keys()))
    data = [data_by_label[k] for k in data_by_label.keys()]
    plt.figure(figsize=(9, 5))
    plt.boxplot(data, vert=True, labels=labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_runs_csv(path: str, rows: List[dict]):
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    cols = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_param_sweep(
    out_dir: str,
    runs: int,
    base_seed: int | None,
    param_name: str,
    values: List,
    fixed: dict,
    diversity: bool = False,
    elitism_is_percent: bool = False,
):
    ensure_dir(out_dir)

    results_by_label = {}
    curves_by_label = {}
    diversity_by_label = {} if diversity else None
    runtime_by_label = {}
    conv1pct_by_label = {}

    for v in values:
        # traduz elitismo % para K se necessário
        params = fixed.copy()
        label = v
        if param_name == 'elitism_k' and elitism_is_percent:
            pop_size = params.get('pop_size', POP_SIZE)
            elit_k = round(pop_size * (v / 100.0))
            params['elitism_k'] = elit_k
            label = f"{v}% (K={elit_k})"
        else:
            params[param_name] = v

        print(f"Config {param_name}={v} → params: {params}")

        cfg_results = []
        cfg_curves = []
        cfg_div_curves = [] if diversity else None
        cfg_runtimes = []
        cfg_conv1pct = []

        for r in range(runs):
            seed = None if base_seed is None else (base_seed + r)
            info = run_ga_once(seed=seed, **params)
            cfg_results.append(info['best_fit'])
            cfg_curves.append(info['best_per_gen'])
            if diversity:
                cfg_div_curves.append(info['unique_per_gen'])
            cfg_runtimes.append(info['runtime_sec'])
            cfg_conv1pct.append(info['conv_gen_1pct'] if info['conv_gen_1pct'] is not None else float('nan'))

        results_by_label[str(label)] = cfg_results
        curves_by_label[str(label)] = cfg_curves
        if diversity:
            diversity_by_label[str(label)] = cfg_div_curves
        runtime_by_label[str(label)] = cfg_runtimes
        conv1pct_by_label[str(label)] = cfg_conv1pct

        # salvar CSV por config
        rows = []
        for i, (res, rt, cg) in enumerate(zip(cfg_results, cfg_runtimes, cfg_conv1pct)):
            row = {
                'run': i + 1,
                'best_fit': res,
                'runtime_sec': rt,
                'conv_gen_1pct': cg,
            }
            rows.append(row)
        save_runs_csv(os.path.join(out_dir, f"runs_{param_name}_{str(v).replace('%','pct')}.csv"), rows)

    # plots gerais do experimento
    plot_convergence(curves_by_label, title=f"Convergência — sweep {param_name}", out_png=os.path.join(out_dir, f"convergencia_{param_name}.png"))
    plot_boxplot(results_by_label, title=f"Boxplot — sweep {param_name}", out_png=os.path.join(out_dir, f"boxplot_{param_name}.png"))

    # runtime boxplot opcional
    plot_boxplot(runtime_by_label, title=f"Tempo de execução (s) — sweep {param_name}", out_png=os.path.join(out_dir, f"boxplot_runtime_{param_name}.png"), ylabel='Tempo (s)')

    if diversity and diversity_by_label is not None:
        plot_diversity(diversity_by_label, title=f"Diversidade — sweep {param_name}", out_png=os.path.join(out_dir, f"diversidade_{param_name}.png"))

    # salvar estatísticas agregadas
    summary_rows = []
    for label, vals in results_by_label.items():
        s = agg_stats(vals)
        s_rt = agg_stats(runtime_by_label[label])
        s_conv = agg_stats([x for x in conv1pct_by_label[label] if x == x])  # remove NaN
        summary_rows.append({
            'config': label,
            'best_fit_mean': s['mean'], 'best_fit_std': s['stdev'], 'best_fit_min': s['min'], 'best_fit_max': s['max'], 'best_fit_median': s['median'],
            'runtime_mean': s_rt['mean'], 'runtime_std': s_rt['stdev'],
            'conv1pct_mean_gen': s_conv['mean'], 'conv1pct_std_gen': s_conv['stdev'],
        })
    save_runs_csv(os.path.join(out_dir, f"summary_{param_name}.csv"), summary_rows)

    return {
        'results_by_label': results_by_label,
        'curves_by_label': curves_by_label,
        'diversity_by_label': diversity_by_label,
        'runtime_by_label': runtime_by_label,
        'conv1pct_by_label': conv1pct_by_label,
    }


def run_all_experiments(base_dir: str = os.path.join('ATIVIDADE_6', 'resultados'), runs: int = RUNS, base_seed: int | None = None):
    ensure_dir(base_dir)

    # Parâmetros base fixos
    base_params = dict(
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        tourney_size=TOURNEY_SIZE,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        elitism_k=ELITISM_K,
    )

    # Experimento 1: Tamanho da População
    exp1_dir = os.path.join(base_dir, 'exp1_pop')
    exp1 = run_param_sweep(
        out_dir=exp1_dir,
        runs=runs,
        base_seed=base_seed,
        param_name='pop_size',
        values=[20, 50, 100],
        fixed=base_params,
        diversity=False,
    )

    # Experimento 2: Taxa de Mutação
    exp2_dir = os.path.join(base_dir, 'exp2_mut')
    exp2 = run_param_sweep(
        out_dir=exp2_dir,
        runs=runs,
        base_seed=base_seed,
        param_name='mutation_rate',
        values=[0.01, 0.05, 0.10, 0.20],
        fixed=base_params,
        diversity=False,
    )

    # Experimento 3: Tamanho do Torneio (com diversidade)
    exp3_dir = os.path.join(base_dir, 'exp3_torneio')
    exp3 = run_param_sweep(
        out_dir=exp3_dir,
        runs=runs,
        base_seed=base_seed,
        param_name='tourney_size',
        values=[2, 3, 5, 7],
        fixed=base_params,
        diversity=True,
    )

    # Experimento 4: Elitismo (% da população)
    exp4_dir = os.path.join(base_dir, 'exp4_elitismo')
    exp4 = run_param_sweep(
        out_dir=exp4_dir,
        runs=runs,
        base_seed=base_seed,
        param_name='elitism_k',
        values=[0, 1, 5, 10],  # porcentagem
        fixed=base_params,
        diversity=False,
        elitism_is_percent=True,
    )

    # Gera relatório consolidado simples
    rel_path = os.path.join(base_dir, 'RELATORIO_EXPERIMENTOS.md')
    with open(rel_path, 'w', encoding='utf-8') as f:
        f.write('# Relatório de Experimentos — AG para TSP (USA13)\n\n')
        f.write('Este relatório foi gerado automaticamente. Cada experimento executa 30 vezes por configuração e salva gráficos e CSVs em subpastas.\n\n')
        f.write('## Experimento 1 — Tamanho da População (20, 50, 100)\n')
        f.write(f'Imagens: {os.path.join("exp1_pop","convergencia_pop_size.png")} e {os.path.join("exp1_pop","boxplot_pop_size.png")}\n\n')
        f.write('## Experimento 2 — Taxa de Mutação (1%, 5%, 10%, 20%)\n')
        f.write(f'Imagens: {os.path.join("exp2_mut","convergencia_mutation_rate.png")} e {os.path.join("exp2_mut","boxplot_mutation_rate.png")}\n\n')
        f.write('## Experimento 3 — Tamanho do Torneio (2, 3, 5, 7)\n')
        f.write(f'Imagens: {os.path.join("exp3_torneio","convergencia_tourney_size.png")}, {os.path.join("exp3_torneio","diversidade_tourney_size.png")} e {os.path.join("exp3_torneio","boxplot_tourney_size.png")}\n\n')
        f.write('## Experimento 4 — Elitismo (0%, 1%, 5%, 10% da população)\n')
        f.write(f'Imagens: {os.path.join("exp4_elitismo","convergencia_elitism_k.png")} e {os.path.join("exp4_elitismo","boxplot_elitism_k.png")}\n\n')
        f.write('Consulte os arquivos CSV de cada pasta para estatísticas numéricas detalhadas.\n')

    return {'exp1': exp1, 'exp2': exp2, 'exp3': exp3, 'exp4': exp4, 'report': rel_path}


if __name__ == '__main__':
    # Ajuste o base_seed para reprodutibilidade completa; ou deixe None para aleatório
    summary = run_all_experiments(base_dir=os.path.join('ATIVIDADE_6','resultados'), runs=RUNS, base_seed=None)
    print('Relatório consolidado em:', summary['report'])
