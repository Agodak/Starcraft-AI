import math
import random


def distL2(x1, y1, x2, y2):
    xdiff = x2 - x1
    ydiff = y2 - y1
    return int(math.sqrt(xdiff*xdiff + ydiff*ydiff) + .5)


def distL1(x1, y1, x2, y2):
    return int(abs(x2 - x1) + abs(y2 - y1) + .5)


def mk_matrix(coord, dist):
    n = len(coord)
    D = {}
    for i in range(n-1):
        for j in range(i+1, n):
            (x1, y1) = coord[i]
            (x2, y2) = coord[j]
            D[i, j] = dist(x1, y1, x2, y2)
            D[j, i] = D[i, j]
    return n, D


def read_tsplib(filename):
    f = open(filename, 'r')
    line = f.readline()
    while line.find("EDGE_WEIGHT_TYPE") == -1:
        line = f.readline()
    if line.find("EUC_2D") != -1:
        dist = distL2
    elif line.find("MAN_2D") != -1:
        dist = distL1
    else:
        print("cannot deal with non-euclidean or non-manhattan distances")
        raise Exception
    while line.find("NODE_COORD_SECTION") == -1:
        line = f.readline()
    xy_positions = []
    while 1:
        line = f.readline()
        if line.find("EOF") != -1:
            break
        (i, x, y) = line.split()
        x = float(x)
        y = float(y)
        xy_positions.append((x, y))
    n, D = mk_matrix(xy_positions, dist)
    return n, xy_positions, D


def mk_closest(D, n):
    C = []
    for i in range(n):
        dlist = [(D[i,j], j) for j in range(n) if j != i]
        dlist.sort()
        C.append(dlist)
    return C


def length(tour, D):
    z = D[tour[-1], tour[0]]
    for i in range(1, len(tour)):
        z += D[tour[i], tour[i-1]]
    return z


def randtour(n):
    sol = list(range(n))
    random.shuffle(sol)
    return sol


def nearest(last, unvisited, D):
    near = unvisited[0]
    min_dist = D[last, near]
    for i in unvisited[1:]:
        if D[last, i] < min_dist:
            near = i
            min_dist = D[last, near]
    return near


def nearest_neighbor(n, i, D):
    unvisited = list(range(n))
    unvisited.remove(i)
    last = i
    tour = [i]
    while unvisited != []:
        next = nearest(last, unvisited, D)
        tour.append(next)
        unvisited.remove(next)
        last = next
    return tour


def exchange_cost(tour, i, j, D):
    n = len(tour)
    a, b = tour[i], tour[(i+1)%n]
    c, d = tour[j], tour[(j+1)%n]
    return (D[a, c] + D[b, d]) - (D[a, b] + D[c, d])


def exchange(tour, tinv, i, j):
    n = len(tour)
    if i > j:
        i, j = j, i
    assert i >= 0 and i < j-1 and j < n
    path = tour[i+1:j+1]
    path.reverse()
    tour[i+1:j+1] = path
    for k in range(i+1, j+1):
        tinv[tour[k]] = k


def improve(tour, z, D, C):
    n = len(tour)
    tinv = [0 for i in tour]
    for k in range(n):
        tinv[tour[k]] = k
    for i in range(n):
        a, b = tour[i], tour[(i+1)%n]
        dist_ab = D[a, b]
        improved = False
        for dist_ac, c in C[a]:
            if dist_ac >= dist_ab:
                break
            j = tinv[c]
            d = tour[(j+1)%n]
            dist_cd = D[c, d]
            dist_bd = D[b, d]
            delta = (dist_ac + dist_bd) - (dist_ab + dist_cd)
            if delta < 0:
                exchange(tour, tinv, i, j)
                z += delta
                improved = True
                break
        if improved:
            continue
        for dist_bd, d in C[b]:
            if dist_bd >= dist_ab:
                break
            j = tinv[d] - 1
            if j == -1:
                j = n-1
            c = tour[j]
            dist_cd = D[c, d]
            dist_ac = D[a, c]
            delta = (dist_ac + dist_bd) - (dist_ab + dist_cd)
            if delta < 0:
                exchange(tour, tinv, i, j)
                z += delta
                break
    return z


def localsearch(tour, z, D, C=None):
    n = len(tour)
    if C == None:
        C = mk_closest(D, n)
    while 1:
        newz = improve(tour, z, D, C)
        if newz < z:
            z = newz
        else:
            break
    return z


def multistart_localsearch(k, n, D, report=None):
    C = mk_closest(D, n)
    bestt = None
    bestz = None
    for i in range(0, k):
        tour = randtour(n)
        z = length(tour, D)
        z = localsearch(tour, z, D, C)
        if bestz == None or z < bestz:
            bestz = z
            bestt = list(tour)
            if report:
                report(z, tour)
    return bestt, bestz


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        coord = [(4, 0), (5, 6), (8, 3), (4, 4), (4, 1), (4, 10), (4, 7), (6, 8), (8, 1)]
        n, D = mk_matrix(coord, distL2)
        instance = "toy problem"
    else:
        instance = sys.argv[1]
        n, coord, D = read_tsplib(instance)
    from time import clock
    init = clock()

    def report_sol(obj, s=""):
        print("cpu:%g\tobj:%g\ttour:%s" % (clock(), obj, s))
    print("*** travelling salesman problem ***")
    print("random construction + local search:")
    tour = randtour(n)
    z = length(tour, D)
    print("random:", tour, z, ' --> ',)
    z = localsearch(tour, z, D)
    print(tour, z)
    print("greedy construction with nearest neighbor + local search:")
    for i in range(n):
        tour = nearest_neighbor(n, i, D)
        z = length(tour, D)
        print("nneigh:", tour, z, ' --> ',)
        z = localsearch(tour, z, D)
        print(tour, z)
    print("random start local search:")
    niter = 100
    tour, z = multistart_localsearch(niter, n, D, report_sol)
    assert z == length(tour, D)
    print("best found solution (%d iterations): z = %g" % (niter, z))
    print(tour)