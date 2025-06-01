import math
class MeasurementSetup:
    def __init__(self):
        self.distance_to_generator = 1  # расстояние до генератора в метрах
        self.distance_to_microphone = 1  # расстояние до микрофона в метрах
        self.device_on = False
        self.detected_signals = []
        self.receiver_frequency = None
        self.bandwidth = (290, 3200)  # диапазон частот в Гц
        self.acoustic_signal_frequency = 1000  # частота акустического сигнала в Гц
        self.sound_pressure_level = None  # уровень звукового давления в дБ
        self.has_acoustic_transformations = False
        self.error_converter = 0.05  # погрешность входного преобразователя в дБ
        self.error_receiver = 0.05     # погрешность измерительного приемника в дБ
        self.impedance = 377            # волновое сопротивление, Ом
        self.E_field_strength = None    # уровень электрического поля, мкВ/м
        self.H_field_strength = None     # уровень магнитного поля, мкА/м
        self.distance_to_recon_location = 1  # расстояние до места разведки
        self.Pn = 0.95  # предельно допустимое значение вероятности правильного обнаружения сигнала
        self.signal_noise_threshold = None  # предельно допустимое значение сигнал/шум δ
        self.Ei = 58e-6  # уровень электрического поля Eи в В/м
        self.Ep = 23e-6  # уровень помех Eп в В/м
        self.pHi = 73e-6  # уровень магнитного поля pHи в В/м
        self.pHp = 33e-6  # уровень помех pHп в В/м
        self.r = 10       # расстояние в метрах
    def set_distance(self):
        print(f"Установлено расстояние до генератора: {self.distance_to_generator} м")
        print(f"Установлено расстояние до микрофона: {self.distance_to_microphone} м")
    def measure_distance_to_recon_location(self):
        print(f"Расстояние до места разведки: {self.distance_to_recon_location} м")
    def turn_on_device(self):
        self.device_on = True
        print("ВТСС включен в штатный режим работы.")
    def detect_signals(self):
        self.detected_signals = [100, 200, 300]
        print(f"Обнаруженные сигналы: {self.detected_signals} МГц")
    def tune_receiver(self):
        if not self.detected_signals:
            print("Сигналы не обнаружены. Настройка приемника невозможна.")
            return
        self.receiver_frequency = max(self.detected_signals)
        self.bandwidth = 20
        print(f"Приемник настроен на частоту: {self.receiver_frequency} МГц")
        print(f"Полоса пропускания установлена: {self.bandwidth} МГц")
    def setup_acoustic_system(self, sound_pressure_with_amplification=True):
        self.acoustic_signal_frequency = 1000  # частота в Гц
        if sound_pressure_with_amplification:
            self.sound_pressure_level = 80  # уровень звукового давления в дБ
        else:
            self.sound_pressure_level = 72  # уровень звукового давления в дБ
        print(f"Акустическая система настроена на частоту: {self.acoustic_signal_frequency} Гц")
        print(f"Уровень звукового давления установлен на: {self.sound_pressure_level} дБ")

    def check_acoustic_transformations(self):
        self.has_acoustic_transformations = True
        if self.has_acoustic_transformations:
            print("Обнаружены акустоэлектрические преобразования.")
            self.measure_fields()
            self.calculate_informative_levels()
        else:
            print("Акустоэлектрические преобразования не обнаружены.")
    def measure_fields(self):
        self.E_field_strength = 5.0   # напряженность электрического поля (кВ/м)
        self.H_field_strength = 0.1     # напряженность магнитного поля (мкА/м)
        print(f"Уровень напряженности электрического поля Eи: {self.E_field_strength} кВ/м")
        print(f"Уровень напряженности магнитного поля ρHи: {self.H_field_strength} мкА/м")
    def calculate_informative_levels(self):
        # Преобразование напряженности электрического поля из кВ/м в мкВ/м
        E_field_mkV_per_m = self.E_field_strength * 1000  # E (мкВ/м) = E (кВ/м) * 1000
        print(f"Напряженность электрического поля: {self.E_field_strength} кВ/м = {E_field_mkV_per_m} мкВ/м")
        # Преобразование напряженности магнитного поля из кА/м в мкА/м
        H_field_mkA_per_m = self.H_field_strength * 1  # H (мкА/м) = H (кА/м) * 1
        print(f"Напряженность магнитного поля: {self.H_field_strength} кА/м = {H_field_mkA_per_m} мкА/м")
        xi_a = (10**(self.error_converter / 10) + (10**(self.error_receiver / 10) - 1))**0.5# Вычисление коэффициента xi_a
        print(f"Коэффициент ξa: ξa = √(10^({self.error_converter}/10) + (10^({self.error_receiver}/10) - 1)) = {xi_a}")
        E_c = (xi_a**2 - xi_a**2) * E_field_mkV_per_m # Вычисление уровня информативного сигнала Ec
        print(f"Уровень информативного сигнала Ec: Ec = ({xi_a}² - {xi_a}²) * {E_field_mkV_per_m} = {E_c} мкВ/м")
        # Вычисление уровня информативного сигнала Hc
        H_c = (xi_a**2 - xi_a**2) * H_field_mkA_per_m
        print(f"Уровень информативного сигнала Hc: Hc = ({xi_a}² - {xi_a}²) * {H_field_mkA_per_m} = {H_c} мкА/м")
    def measure_interference_levels(self):
        interference_E_field_strength = 0.5  #  уровень помех (кВ/м)
        interference_H_field_strength = 0.05   #  уровень помех (А/м)
        print(f"Уровень напряженности помех Eпj: {interference_E_field_strength} кВ/м")
        print(f"Уровень напряженности помех ρHпj: {interference_H_field_strength} А/м")
    def calculate_attenuation_coefficient(self):
        if not self.receiver_frequency:
            print("Частота сигнала не задана.")
            return
        f = self.receiver_frequency
        r = self.distance_to_recon_location
        print(f"Частота сигнала (f): {f} МГц")
        print(f"Расстояние до места восстановления (r): {r} м")
        if f <= 47.75:
            V_r = 3 * r / f
            print(f"nКоэффициент затухания V_r для f <= 47.75 МГц:")
            print(f"V_r = 3 * r / f")
            print(f"Подстановка значений: V_r = 3 * {r} / {f} = {V_r}")
        elif 47.75 < f <= 1800:
            V_r = r ** 2 / f
            print(f"nКоэффициент затухания V_r для 47.75 < f <= 1800 МГц:")
            print(f"V_r = r² / f")
            print(f"Подстановка значений: V_r = {r}² / {f} = {V_r}")
        elif f > 1800:
            V_r = r
            print(f"nКоэффициент затухания V_r для f > 1800 МГц:")
            print(f"V_r = r")
            print(f"Подстановка значений: V_r = {r} = {V_r}")
        print(f"nИтоговый коэффициент затухания V_r: {V_r}")
    def calculate_signal_noise_ratio(self):
        E_c = (10 ** (self.sound_pressure_level / 10)) / self.impedance
        H_c = (10 ** (self.sound_pressure_level / 10)) / self.impedance
        print(f"Значение звукового давления (Sound Pressure Level): {self.sound_pressure_level} дБ")
        print(f"Импеданс: {self.impedance} Ом")
        print(f"nРасчет E_c:")
        print(f"E_c = 10^(SPL / 10) / Z")
        print(f"Подстановка значений: E_c = 10^({self.sound_pressure_level} / 10) / {self.impedance} = {E_c}")
        print(f"nРасчет H_c:")
        print(f"H_c = 10^(SPL / 10) / Z")
        print(f"Подстановка значений: H_c = 10^({self.sound_pressure_level} / 10) / {self.impedance} = {H_c}")
        qE = self.Ei / (math.sqrt(2) * (10 ** (self.error_converter / 10)))
        qH = self.pHi / (math.sqrt(2) * (10 ** (self.error_receiver / 10)))
        print(f"nЭнергия сигнала Ei: {self.Ei}")
        print(f"Ошибка преобразователя: {self.error_converter} дБ")
        print(f"Ошибка приемника: {self.error_receiver} дБ")
        print(f"nРасчет qE:")
        print(f"qE = Ei / (sqrt(2) * 10^(error_converter / 10))")
        print(f"Подстановка значений: qE = {self.Ei} / (sqrt(2) * 10^({self.error_converter} / 10)) = {qE}")
        print(f"nРасчет qH:")
        print(f"qH = pHi / (sqrt(2) * 10^(error_receiver / 10))")
        print(f"Подстановка значений: qH = {self.pHi} / (sqrt(2) * 10^({self.error_receiver} / 10)) = {qH}")
        # Расчет предельно допустимого значения сигнал/шум δ
        delta = (1 + (3.2 * math.exp(-qE)) + (3.16 * self.Pn * math.exp(-qH)))
        print(f"nРасчет предельно допустимого значения сигнал/шум δ:")
        print(f"δ = 1 + (3.2 * exp(-qE)) + (3.16 * Pn * exp(-qH))")
        print(f"Подстановка значений: δ = 1 + (3.2 * exp(-{qE})) + (3.16 * {self.Pn} * exp(-{qH})) = {delta}")
        # Проверка условий
        if qE <= delta and qH <= delta:
            print("Перехват ПЭМИ ВТСС невозможен на заданных частотах.")
            return True
        else:
            print("Перехват ПЭМИ ВТСС возможен. Необходимо определение реального коэффициента затухания.")
            return False
    def determine_real_attenuation_coefficient(self):
        d = 1.0  # расстояние от вспомогательного излучателя до приемника в метрах
        E_g = float(input("Введите уровень напряженности электрического поля от вспомогательного излучателя (в кВ/м): "))
        r = float(input("Введите расстояние до места возможного нахождения средств технической разведки (в метрах): "))
        E_r = float(input("Введите уровень напряженности электрического поля в месте нахождения средств разведки (в кВ/м): "))
        E_p = float(input("Введите уровень помех при выключенном генераторе (в кВ/м): "))
        #  реальное затухание как V_r*
        Vr_star = E_g / E_r
        print("nРасчет реального затухания V_r*:")
        print(f"V_r* = E_g / E_r")
        print(f"Подстановка значений: V_r* = {E_g} / {E_r}")
        print(f"V_r* = {Vr_star:.4f}n")
        qV = self.qE * Vr_star
        print("Расчет отношения сигнал/шум на входе измерительного приемника q(q):")
        print(f"q(q) = qE * V_r*")
        print(f"Подстановка значений: q(q) = {self.qE} * {Vr_star:.4f}")
        print(f"q(q) = {qV:.4f}n")
        E_p = self.Ep
        pH_p = self.pHp
        #  реальное затухание как V_r* для Eи и pHи
        Vr_star_E = self.Ei / E_p
        Vr_star_H = self.pHi / pH_p
        print("Расчет реального коэффициента затухания для Eи и pHи:")
        # Расчет V_r* для Eи
        print("nДля Eи:")
        print(f"V_r* (E) = Ei / E_p")
        print(f"Подстановка значений: V_r* (E) = {self.Ei} / {E_p}")
        print(f"V_r* (E) = {Vr_star_E:.4f}")
        # Расчет V_r* для pHи
        print("nДля pHи:")
        print(f"V_r* (H) = pHi / pH_p")
        print(f"Подстановка значений: V_r* (H) = {self.pHi} / {pH_p}")
        print(f"V_r* (H) = {Vr_star_H:.4f}n")
        return Vr_star_E, Vr_star_H, qV
        E_p = self.Ep  # Убедитесь, что Ep определено
        pH_p = self.pHp  # Убедитесь, что pHp определено
        Vr_star_E = self.Ei / E_p
        Vr_star_H = self.pHi / pH_p
        print(f"nРасчет реального коэффициента затухания:")
        print(f"V_r* (E) = Ei / E_p")
        print(f"Подстановка значений: V_r* (E) = {self.Ei} / {E_p} = {Vr_star_E}")
        print(f"V_r* (H) = pHi / pH_p")
        print(f"Подстановка значений: V_r* (H) = {self.pHi} / {pH_p} = {Vr_star_H}")
        return Vr_star_E, Vr_star_H, qV

if __name__ == "__main__":
    setup = MeasurementSetup()
    setup.set_distance()
    setup.measure_distance_to_recon_location()
    setup.turn_on_device()
    setup.detect_signals()
    setup.tune_receiver()
    setup.setup_acoustic_system()
    setup.check_acoustic_transformations()
    setup.measure_interference_levels()
    if not setup.calculate_signal_noise_ratio():
        setup.determine_real_attenuation_coefficient()
