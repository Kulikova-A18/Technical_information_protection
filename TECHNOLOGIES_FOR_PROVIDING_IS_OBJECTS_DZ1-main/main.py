from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Функция для расчета затухания в коробах
def attenuation(length, size, material_factor):
    # Расчет коэффициента затухания (delta) на основе размера
    delta = 0.01 + (0.99 * size / 100)  # Примерная формула для затухания
    # Возвращает общее затухание, умножая delta на длину и коэффициент материала
    return delta * length * material_factor

# Параметры частот для анализа
fs = [250, 500, 1000, 2000, 4000]  # Основные частоты
fs_high = [355, 710, 1400, 2800, 5600]  # Высокочастотные параметры
fs_low = [180, 355, 710, 1400, 2800]  # Низкочастотные параметры

# Функция для расчета W(R) в зависимости от R
def W(R):
    if R < 0.15:
        # Формула для W(R), если R меньше 0.15
        return 1.54 * R * (0.25) * (1 - exp(-11 * R))
    # Формула для W(R), если R больше или равен 0.15
    tmp = -11 * R / (1 + 0.7 * R)
    return 1 - exp(tmp)

# Функция для расчета суммы произведений ps и ks
def fR(ps, ks):
    return np.sum(ps * ks)  # Возвращает сумму произведений элементов массивов ps и ks

# Функция для расчета ΔA(f) в зависимости от частоты f
def ΔA(f):
    if f > 1000:
        # Формула для ΔA(f), если f больше 1000 Гц
        return 1.37 + 1000 / (f * 0.69)
    # Формула для ΔA(f), если f меньше или равен 1000 Гц
    return 200 / (f * 0.43) - 0.37

# Функция для расчета p_f(Qs)
def p_f(Qs):
    Qs = list(Qs)  # Преобразуем Qs в список
    res = []  # Список для хранения результатов
    for qi in Qs:
        # Расчет временной переменной tmp на основе формулы p_f
        tmp = (0.78 + 5.46 * exp(-4.3 * (10 ** (-3)) * (27.3 - abs(qi) ** 2))) / \
              (1 + 10 ** (0.1 * abs(qi)))
        if qi < 0:
            res.append(tmp)  # Если qi < 0, добавляем tmp в результаты
        else:
            res.append(1 - tmp)  # В противном случае добавляем 1 - tmp
    return res

# Функция для расчета k_f(fs)
def k_f(fs):
    return np.array(
        [
            2.57 * (10 ** (-8)) * fi ** (2.4)
            for fi in filter(lambda x: x <= 400, fs)  # Для частот <= 400 Гц
        ] + [
            1 - 1.074 * exp(-10 ** (-4) * fi ** 1.18)
            for fi in filter(lambda x: x > 400 and x < 10000, fs)  # Для частот между 400 и 10000 Гц
        ]
    )

def main():
    # Вывод формул для понимания расчетов
    print("\nФормулы:")
    print(r"""
        Формула для затухания:
            Δ = (0.01 + (0.99 * size / 100)) * length * material_factor

        Формула для W(R):
            W(R) =
            {
                1.54 * R * (0.25) * (1 - e^(-11R)), если R < 0.15
                1 - e^(-11R / (1 + 0.7R)), иначе
            }

        Формула для fR(ps, ks):
            R = Σ(ps * ks)

        Формула для ΔA(f):
            ΔA(f) =
            {
                1.37 + (1000 / (f * 0.69)), если f > 1000
                (200 / (f * 0.43)) - 0.37, иначе
            }

        Формула для p_f(Qs):
            p_f(Qs) = 
            {
                (0.78 + (5.46 * e^(-4.3 * (10^(-3)) * (27.3 - |qi|^2))) / 
                (1 + 10^(0.1 * |qi|)), если qi < 0
                иначе: 
                1 - p_f(Qs)
            }

        Формула для k_f(fs):
            k_f(fs) =
            {
                2.57 * (10^(-8)) * fi^(2.4), если fi <= 400
                1 - (1.074 * e^(-10^(-4) * fi^(1.18)), если fi > 400 и fi < 10000
            }
        """)

    # Уровни речи и шума в дБ
    speach_lvls = np.array([41, 38, 37, 34, 33])
    noise_lvls = np.array([63, 58, 55, 52, 50])
    dLt = 35

    # Расчет уровней Qs и Es на основе уровней речи и шума
    qs = speach_lvls - noise_lvls
    es = speach_lvls - noise_lvls - dLt

    # Создание таблицы для qi и Ei с использованием pandas DataFrame
    df_qi_ei = pd.DataFrame({
        'qi': qs,
        'Ei': es
    })

    print("Таблица qi и Ei:")
    print(df_qi_ei)

    # Расчет ΔA для каждой частоты в списке fs
    dAs = [ΔA(fsi) for fsi in fs]
    # Вывод формулы ΔA(f)
    for fsi, dAi in zip(fs, dAs):
        if fsi > 1000:
            formula = f"ΔA({fsi}) = 1.37 + (1000 / ({fsi} * 0.69)) = {dAi:.2f}"
        else:
            formula = f"ΔA({fsi}) = (200 / ({fsi} * 0.43)) - 0.37 = {dAi:.2f}"
        print(formula)

    # Расчет разности между k_f для высоких и низких частот
    ks = k_f(fs_high) - k_f(fs_low)

    # Расчет Qs и ps на основе ранее вычисленных значений
    Qs = qs - dAs
    ps = p_f(Qs)

    # Вывод формулы fR(ps, ks)
    r = fR(ps, ks)
    print(f"\nfR(ps, ks) = Σ(ps * ks) = {r:.2f}")

    # Вывод формулы W(R)
    w = W(r)
    print(f"W(R) = {w:.4f}")

    # Пример расчета затухания для короба с заданными параметрами
    length = 10  # длина короба в метрах
    size = 50   # поперечное сечение в см^2
    material_factor = 1.0   # коэффициент материала

    attn = attenuation(length, size, material_factor)

    # Вывод формулы затухания с подставленными значениями
    delta = 0.01 + (0.99 * size / 100)
    print(f"Затухание: Δ = ({delta:.2f}) * {length} * {material_factor} = {attn:.2f} дБ")

    # Создание таблицы результатов с использованием pandas DataFrame
    df_results = pd.DataFrame({
        'Частота (Гц)': fs,
        'ΔAi': np.round(dAs, 2),
        'Qi': np.round(Qs, 2),
        'Pi': np.round(ps, 3),
        'Ki': np.round(ks, 3)
    })

    print("\nТаблица результатов:")
    print(df_results)

    # Визуализация результатов с использованием matplotlib
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fs, dAs, marker='o', label='ΔAi')
    plt.plot(fs, Qs, marker='o', label='Qi')
    plt.title('ΔAi и Qi по частоте')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Значение')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fs, ps, marker='o', label='Pi')
    plt.plot(fs, ks, marker='o', label='Ki')
    plt.title('Pi и Ki по частоте')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Значение')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

