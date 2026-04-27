import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("boilerplate-medical-data-visualizer/medical_examination.csv")

# 2
df['overweight'] = df.apply(lambda x: 1 if (x['weight'] / (x['height'] / 100)**2) > 25 else 0, axis=1)
print(df)
# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    g = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', kind='bar')
    fig = g.fig

    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    # 11: Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12: Calculate the correlation matrix
    corr = df_heat.corr()

    # 13: Generate a mask for the upper triangle
    mask = np.triu(corr)

    # 14: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15: Plot the correlation matrix
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.1f', 
        center=0, square=True, linewidths=.5, 
        cbar_kws={"shrink": .5}, ax=ax
    )

    # 16
    fig.savefig('heatmap.png')
    return fig