# TSP https://developers.google.com/optimization/routing/tsp#python

# 0. New York - 1. Los Angeles - 2. Chicago - 3. Minneapolis - 4. Denver - 5. Dallas - 6. Seattle - 7. Boston - 8. San Francisco - 9. St. Louis - 10. Houston - 11. Phoenix - 12. Salt Lake City
# New York is the start and end location. Distance is measured in miles.

USA13 = [
        [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
        [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
        [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
        [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
        [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
        [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
        [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
        [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
        [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
        [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
        [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
        [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
        [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
    ]


def route_distance(route, matrix=USA13):
    """
    Calcula a distância total de uma rota no TSP (inclui retorno ao ponto inicial).

    Parâmetros:
    - route: lista ou tupla de índices de cidade (ex: [0,1,2,...])
    - matrix: matriz de distâncias (padrão: USA13)

    Retorna:
    - distância total (float ou int)

    Lança ValueError se a rota contiver índices inválidos ou tipos incorretos.
    """
    if not isinstance(route, (list, tuple)):
        raise ValueError("route deve ser list ou tuple de índices inteiros")
    if len(route) == 0:
        return 0
    n = len(matrix)
    total = 0
    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]
        if not isinstance(a, int) or not isinstance(b, int):
            raise ValueError("índices de cidade devem ser inteiros")
        if a < 0 or a >= n or b < 0 or b >= n:
            raise ValueError(f"índice de cidade fora do intervalo: {a} or {b}")
        total += matrix[a][b]
    # fechar o ciclo: último -> primeiro
    last = route[-1]
    first = route[0]
    if not isinstance(last, int) or not isinstance(first, int):
        raise ValueError("índices de cidade devem ser inteiros")
    if last < 0 or last >= n or first < 0 or first >= n:
        raise ValueError(f"índice de cidade fora do intervalo: {last} or {first}")
    total += matrix[last][first]
    return total


def is_valid_route(route, n_cities=None):
    """
    Verifica se `route` é uma permutação válida que visita cada cidade exatamente uma vez.

    Parâmetros:
    - route: lista ou tupla de índices
    - n_cities: número esperado de cidades (padrão: tamanho de USA13)

    Retorna True se for válida, False caso contrário.
    """
    if not isinstance(route, (list, tuple)):
        return False
    if n_cities is None:
        n_cities = len(USA13)
    if len(route) != n_cities:
        return False
    # todos os elementos devem ser inteiros no intervalo 0..n_cities-1
    try:
        elems = set(int(x) for x in route)
    except Exception:
        return False
    if elems != set(range(n_cities)):
        return False
    return True


if __name__ == '__main__':
    # Demo rápido
    rota_exemplo = list(range(len(USA13)))  # 0,1,2,...,12
    print('Rota de exemplo válida?', is_valid_route(rota_exemplo))
    print('Distância da rota de exemplo:', route_distance(rota_exemplo))
