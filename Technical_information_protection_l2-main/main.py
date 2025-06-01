import numpy as np 
import pandas as pd 

def sound_pressure_level(L_w, r): 
  return L_w - 20 * np.log10(r) - 11 
def corrected_sound_pressure_level(L_p, alpha): 
  return L_p - 10 * np.log10(1 - alpha) 
def combined_sound_pressure_level(L_levels): 
  return 10 * np.log10(np.sum(10 ** (np.array(L_levels) / 10))) 
def main(): 
  a, b, c = 7, 6, 3     alpha_avg = 0.2 
  L_w = 70 
  distances = [4, 2, 4] 
  L_p = [sound_pressure_level(L_w, r) for r in distances] 
  L_p_corr = [corrected_sound_pressure_level(lp, alpha_avg) for lp in L_p] 
  L_p_total = combined_sound_pressure_level(L_p_corr) 
  
  data = { 
    'Стенд': ['1', '2', '3'], 
    'Расстояние (м)': distances, 
    'Уровень звукового давления (дБ)': L_p, 
    'Скорректированный уровень (дБ)': L_p_corr 
  }      
  
  df = pd.DataFrame(data)    

  print(f"1. Уровень звуковой мощности стендов: {L_w} дБ") 
  for i, r in enumerate(distances):         
    print(f"Формула для стенда {i + 1}: L_p[{i + 1}] = {L_w} - 20 * log10({r}) - 11 = {L_p[i]:.2f} дБ") 
  
  print("2. Расстояния и уровни звукового давления:")     print(df.to_string(index=False)) 
  
  print("Формула: L_p_corr = L_p - 10 * log10(1 - alpha)") 
  for i, lp in enumerate(L_p): 
    print(f"Формула для стенда {i + 1}: L_p_corr[{i + 1}] = {lp:.2f} - 10 * log10(1 - {alpha_avg}) = {L_p_corr[i]:.2f} дБ") 
  
  print("4. Общий уровень звукового давления в точке PТ:")     
  
  print("Формула: L_p_total = 10 * log10(Σ(10^(L_p_corr / 10)))") 
  combined_expr = " + ".join([f"10^({lp / 10:.2f})" for lp in L_p_corr])     
  print(f"L_p_total = 10 * log10(Σ({combined_expr})) = {L_p_total:.2f} дБ") 

if __name__ == "__main__": 
  main() 
