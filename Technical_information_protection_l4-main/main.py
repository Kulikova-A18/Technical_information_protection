import math

h = 6.62607015e-34              # Дж * с (постоянная Планка)
c = 299792458                   # м/с (скорость света в вакууме)
lambda_nm = 1490                # нм (длина волны)
lambda_ = lambda_nm * 1e-9      # переводим в метры
tau = 1e-6                      # с (тактовый период, переводим из мкс в с)
I = 256                         # бит (количество информации на символ)

def calculate_Pmin(n):
    Pmin = ((h * c) / (lambda_ * tau)) * ((2 * 2 * I / n) - 1) ** 2
    formula = f"Pmin = (({h} * {c}) / ({lambda_} * {tau})) * ((2 * 2 * {I} / {n}) - 1)^2"
    print(formula)
    return Pmin

def calculate_D_and_DdB(power_source_dBm, sensitivity_receiver_dBm):
    D = 10 ** ((power_source_dBm - sensitivity_receiver_dBm) / 10)
    formula_D = f"D = 10^(({power_source_dBm} - {sensitivity_receiver_dBm}) / 10)"
    print(formula_D)

    DdB = 10 * math.log10(D)
    formula_DdB = f"DdB = 10 * log10(D)"
    print(formula_DdB)

    return D, DdB

n = float(input("Введите количество единичных сигналов (n): "))
print(f"Количество единичных сигналов (n): {n}")

# Pmin
Pmin = calculate_Pmin(n)
print(f"Pmin: {Pmin:.6e} Вт")

power_source_dBm = -3           # дБм (максимальная мощность источника оптического излучения)
sensitivity_receiver_dBm = -19  # дБм (минимальная пороговая чувствительность приемника)

# D и DdB
D, DdB = calculate_D_and_DdB(power_source_dBm, sensitivity_receiver_dBm)
print(f"D: {D:.6f}")
print(f"DdB: {DdB:.6f} дБ")
