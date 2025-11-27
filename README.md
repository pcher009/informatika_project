import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def b_pol(xs, ys, deg):
    cfs = np.polyfit(xs, ys, deg)
    x_s = sp.symbols('x')
    expr = 0
    p = deg
    for c in cfs:
        expr += c * x_s**p
        p -= 1
    expr_p = sp.simplify(expr)
    return cfs, x_s, expr_p


def fmt_num(c):
    if abs(c - round(c)) < 1e-9:
        return str(int(round(c)))
    s = f"{c:.3f}".rstrip("0").rstrip(".")
    return s


def fmt_pol(cfs):
    sup = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻"
    }

    def p_sup(p):
        if p == 1:
            return ""
        return "".join(sup[ch] for ch in str(p))

    deg = len(cfs) - 1
    terms = []

    for i, c in enumerate(cfs):
        p = deg - i
        if abs(c) < 1e-12:
            continue
        s = "-" if c < 0 else "+"
        ac = abs(c)

        if p == 0:
            cs = fmt_num(ac)
            vx = ""
        else:
            if abs(ac - 1) < 1e-9:
                cs = ""
            else:
                cs = fmt_num(ac)
            vx = "x" + p_sup(p)

        terms.append((s, cs + vx))

    if not terms:
        return "0"

    fs, ft = terms[0]
    res = ("-" + ft) if fs == "-" else ft

    for s, t in terms[1:]:
        res += f" {s} {t}"

    return res


def main():
    print("Введите x через пробел:")
    xs = list(map(float, input().strip().split()))

    print("Введите y через пробел:")
    ys = list(map(float, input().strip().split()))

    print("Введите степень многочлена:")
    deg = int(input().strip()))


    if len(xs) != len(ys):
        print("Ошибка: количество x и y должно совпадать!")
        return


    cfs, x_s, expr = b_pol(xs, ys, deg)

    print("\nМногочлен (как на клавиатуре):")
    print("f(x) =", expr)

    print("\nМногочлен (как в учебнике):")
    print("f(x) =", fmt_pol(cfs))


    x_min, x_max = min(xs), max(xs)
    dx = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
    st, en = x_min - dx, x_max + dx

    xp = np.linspace(st, en, 400)
    yp = np.polyval(cfs, xp)


    ps = sorted(zip(xs, ys))
    xs_s = [p[0] for p in ps]
    ys_s = [p[1] for p in ps]


    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(xp, yp)
    plt.scatter(xs, ys, color="red")
    plt.grid(True)
    plt.title("Гладкий график")
    plt.xlabel("x")
    plt.ylabel("f(x)")


    plt.subplot(1, 2, 2)
    plt.plot(xs_s, ys_s, marker="o")
    plt.grid(True)
    plt.title("Ломаная по точкам")
    plt.xlabel("x")
    plt.ylabel("y")


    plt.tight_layout()
    plt.show()


main()
