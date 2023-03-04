import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew, kurtosis


def caract_df(df: pd.DataFrame):
    """
    Características básicas de um dataframe.

    :param df: Dataframe a ser análisado
    :return: None
    """
    print('Nº rows: {}\nNº columns: {}'.format(df.shape[0], df.shape[1]))
    print('Nº duplicated rows: {}'.format(df.duplicated().sum()))
    print('\nNº of missings*:')
    for col in df.columns:
        # n_na = pd.isna(df[col]).sum()
        n_na = df[col].isnull().sum()
        if n_na > 1:
            print('\t{}: {} - {}%'.format(col,
                                          n_na,
                                          round(100 * n_na / df.shape[0], 2)))
    print('(*) Before processing')
    return


def barplot(serie: pd.Series, c='Green'):
    """
    Função automática de criação de sns.barplot
    :param serie: Série que origininará o gráfico
    :param c: Cor do gráfico
    :return: None
    """

    # Contabilizando Na's
    serie = serie.replace(np.nan, 'Missing', regex=True).copy()

    # Gerando dados para plot
    df = serie.value_counts().reset_index()
    df = df.astype({'index': str, serie.name: int})

    # Separando dados missings
    miss = df[df['index'] == 'Missing']

    # criando categoria "outros"
    df = df[df['index'] != 'Missing'].reset_index(drop=True)
    size_outros = 0
    if df.shape[0] > 10:
        size_outros = df.shape[0] - 6
        df_top = df.head(5).copy()
        df = pd.concat([df_top, pd.DataFrame({'index': ['Outros'],
                                              serie.name: df.drop([0, 1, 2, 3, 4])[serie.name].sum()})])

    df = pd.concat([df, miss]).reset_index(drop=True)

    # Display dados
    df_display = df.copy()
    df_display.columns = [serie.name, 'Freq']
    if size_outros > 0:
        print("Mostrando as maiores categorias de {}".format(df_display.shape[0] + size_outros))

    # Setup Plot
    palette = {catg: c if (catg != 'Outros') & (catg != 'Missing') else 'gray' if catg != 'Missing' else 'black'
               for catg in df['index']}
    altura = 4 if df.shape[0] < 16 else int(len(df['index']) / 4)
    plt.figure(figsize=(14, altura))

    # Plot
    sns.barplot(x=serie.name, y='index', data=df, palette=palette)
    for y, x in enumerate(df[serie.name]):
        dif = df[serie.name].max() * 0.005
        percent = 100 * x / df[serie.name].sum()

        # Posição das legendas
        if x > 7 * dif:
            plt.annotate(x, xy=(x - dif, y), ha='right', va='center', color='white')
            plt.annotate('{:.2f}%'.format(percent), xy=(x + dif, y), ha='left', va='center', color='black')
        else:
            plt.annotate('{} - {:.2f}%'.format(x, percent), xy=(x + dif, y), ha='left', va='center', color='black')

    # layout
    plt.xlim(0, df[serie.name].max() * 1.1)

    plt.title('Frequência da coluna {}'.format(serie.name))
    plt.xlabel('Frequência')
    plt.ylabel(serie.name.title().replace('_', ' '))
    plt.show()

    return


def iqr(serie: pd.Series, multiplicador=3.0):
    """
    Análise de outliers da série numérica

    :param serie: Série de valores numéricos
    :param multiplicador: Intervalo do que é considerado aceitavel, sugestão 3.0
    :return: série sem outliers
    """
    # Valores outliers
    # q1, q3 = np.quantile(serie, [0.25, 0.75])
    # IQR = (q3 - q1) * multiplicador
    # limit_lower = q1 - IQR if q1 > IQR else 0
    # limit_upper = q3 + IQR
    factor = multiplicador
    limit_upper = serie.mean() + serie.std() * factor
    limit_lower = serie.mean() - serie.std() * factor

    # Outliers
    outliers = [x for x in serie if (x > limit_upper) | (x < limit_lower)]
    in_limits = [x if (x <= limit_upper) & (x >= limit_lower) else np.nan for x in serie]

    print('Número de outliers (excluídos): {} ({}% do total)'.format(len(outliers),
                                                                     round(len(outliers) * 100 / len(serie),
                                                                           2)))
    print('Número de registros considerados: {}'.format(len(serie) - len(outliers)))

    return pd.Series(in_limits, name=serie.name)


def numeric_plot(serie: pd.Series, c='Green', outliers=True, mult=3.0):
    """
    Análise de dados numéricos.

    :param serie: Série a ser analisada
    :param c: cor do gráfico
    :param outliers: Indicador dos outliers
    :param mult: Intervalo do que é considerado aceitavel, sugestão 1.5 ou 2.5.
    :return: None
    """

    # Outliers
    if outliers:
        serie = iqr(serie.copy(), multiplicador=mult)

    serie = serie.loc[pd.notna(serie)].copy()
    df = serie.describe().reset_index()
    df = pd.concat([df, pd.DataFrame({'index': ['skewness', 'Kurtosis'],
                                      serie.name: [skew(serie), kurtosis(serie)]})])

    df.columns = ['', 'Valor']
    df.set_index('', inplace=True)

    # Plot
    fig, ax = plt.subplots(2, figsize=(15, 6), sharex=True)

    violin = sns.violinplot(x=serie, color=c, inner=None, ax=ax[0])
    plt.setp(violin.collections, alpha=.3)
    sns.boxplot(x=serie, color=c, ax=ax[0])

    sns.histplot(x=serie, color=c, ax=ax[1])

    # Plot layout
    ax[0].set_xlabel('Valor')
    ax[1].set_xlabel('Valor')

    ax[0].set_ylabel('{}'.format(serie.name.title().replace('_', ' ')))
    ax[1].set_ylabel('Frequência')

    ax[0].set_title('Distribuição da variável {}'.format(serie.name.title().replace('_', ' ')))
    ax[1].set_title('Distribuição da variável {}'.format(serie.name.title().replace('_', ' ')))

    plt.tight_layout()
    plt.show()
    return
