import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
# Calculamos el IMC: peso / (altura en metros) al cuadrado
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3
# Normalizar
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    # Convertimos a formato long usando melt
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    # Agrupamos
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7
    # Creamos el gráfico
    fig = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar').fig

    # 8 (La variable fig ya se asignó en el paso 7)

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    # Matriz de correlación
    corr = df_heat.corr()

    # 13
    # Máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    # Configuramos la figura
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15
    # Graficamos el mapa de calor
    sns.heatmap(corr, 
                mask=mask, 
                annot=True, 
                fmt='.1f', 
                center=0, 
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .5}, 
                ax=ax)

    # 16
    fig.savefig('heatmap.png')
    return fig