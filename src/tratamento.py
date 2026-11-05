
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
import seaborn as srn

caminho = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "dados_sensores_vulcantech.csv")

df = pd.read_csv(caminho)

print(df.head())
df.info()

print('\nValores ausentes em cada coluna:')
print(df.isnull().sum())

print('\nEstatísticas descritivas:')
print(df.describe())

# VALORES CATEGÓRICOS
print('\nAnálise dos valores categóricos:')
for col in df.select_dtypes(include='object').columns:
    print('Coluna:', col)
    print(df[col].value_counts())


# FILTROS PARA IDENTIFICAR ERROS DE LEITURA (ruídos físicos)
filtro_erros_temp = df['temperatura_fuso_c'] >= 400          # 999.9°C
filtro_erros_vib  = df['vibracao_triaxial_rms_mms'] < 0      # -99.0 mm/s
filtro_erros_vaz  = df['vazao_refrigerante_lmin'] < 0
filtro_erros_pres = df['pressao_hidraulica_bar'] < 0         # -1.0 bar

# CONVERSÃO DO TIMESTAMP E ORDENAÇÃO (necessária para interpolação temporal)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# SUBSTITUIÇÃO DOS RUÍDOS POR NaN
print('\n\nTRATAMENTO DE RUÍDOS\n')

# 1 - TEMPERATURA
print(f'Temperatura — ruídos: {filtro_erros_temp.sum()}')
df.loc[filtro_erros_temp, 'temperatura_fuso_c'] = np.nan

# 2 - VIBRAÇÃO
print(f'Vibração — ruídos: {filtro_erros_vib.sum()}')
df.loc[filtro_erros_vib, 'vibracao_triaxial_rms_mms'] = np.nan

# 3 - VAZÃO
print(f'Vazão — ruídos: {filtro_erros_vaz.sum()}')
df.loc[filtro_erros_vaz, 'vazao_refrigerante_lmin'] = np.nan

# 4 - PRESSÃO
print(f'Pressão — ruídos: {filtro_erros_pres.sum()}')
df.loc[filtro_erros_pres, 'pressao_hidraulica_bar'] = np.nan

# INTERPOLAÇÃO TEMPORAL CONDICIONAL POR EQUIPAMENTO (id_celula)
# O projeto exige interpolação temporal por célula, não média global.
df = df.set_index('timestamp')
for col in ['temperatura_fuso_c', 'vibracao_triaxial_rms_mms',
            'vazao_refrigerante_lmin', 'pressao_hidraulica_bar']:
    df[col] = df.groupby('id_celula', group_keys=False)[col].apply(
        lambda g: g.interpolate(method='time'))
    # Fallback: mediana por liga para NaNs remanescentes (início da série)
    if df[col].isnull().any():
        df[col] = df[col].fillna(df.groupby('liga_metalica')[col].transform('median'))
        df[col] = df[col].fillna(df[col].median())
df = df.reset_index()

# CONFERÊNCIA DOS DADOS APÓS O TRATAMENTO
print('\nVALORES AUSENTES APÓS O TRATAMENTO:')
print(df.isnull().sum())

print('\nESTATÍSTICAS APÓS O TRATAMENTO:')
print(df.describe())

# 3. ENGENHARIA DE FEATURES DINÂMICAS POR LOTE

# Ordenar por data/hora para permitir cálculos de janelas temporais
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').set_index('timestamp')

#Médias e desvios de Rollings baseado nas colunas
#Criação da coluna para os Rollings
colunas_sensores = [
    'temperatura_fuso_c',
    'vibracao_triaxial_rms_mms',
    'vazao_refrigerante_lmin',
    'pressao_hidraulica_bar'
]

#Janelas de tempo para os rolling means e std
for coluna in colunas_sensores:
    # Janela de 30 minutos
    df[f'{coluna}_media_30min'] = (df[coluna].rolling('30min', min_periods=1).mean())
    df[f'{coluna}_desvio_30min'] = (df[coluna].rolling('30min', min_periods=1).std())
    # Janela de 2 horas
    df[f'{coluna}_media_2h'] = (df[coluna].rolling('2h', min_periods=1).mean())
    df[f'{coluna}_desvio_2h'] = (df[coluna].rolling('2h', min_periods=1).std())

# ÍNDICE DE ESFORÇO MECÂNICO (IEM)
# IEM = Torque x Vibração RMS / (Vazão + 1)

df['IEM'] = (df['torque_fuso_nm'] * df['vibracao_triaxial_rms_mms']) / (df['vazao_refrigerante_lmin'] + 1)
print('\n\n O valor de IEM é :\n', df['IEM'].describe())

# NORMALIZAÇÃO PELA MÉDIA HISTÓRICA DA LIGA METÁLICA

# que identifica a liga metálica de cada registro.
for coluna in colunas_sensores:

    media_historica = df.groupby('liga_metalica')[coluna].transform('mean')
    df[f'{coluna}_normalizada'] = (df[coluna] / media_historica)

# CONFERÊNCIA DAS NOVAS FEATURES
print('\nNovas features criadas:')
print(df.head())

print('\nColunas disponíveis:')
print(df.columns.tolist())
