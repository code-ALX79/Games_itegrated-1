

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
print("Análisis anual de videojuegos: ventas, reseñas y plataformas para detectar patrones de éxito.")

print()

print("Importar las librerias")


print()

print("Exploración inicial del dataset: revisión de estructura, tipos de datos y valores ausentes.")


print()


df_gms = pd.read_csv(
    'https://practicum-content.s3.us-west-1.amazonaws.com/datasets/games.csv')
print(df_gms.head())


print()


print(df_gms.info())

print()

print()

print(df_gms.duplicated().sum())

print()

n_rows = df_gms.shape[0]
100 * df_gms.isna().sum() / n_rows

print()

df_gms['Year_of_Release'] = df_gms['Year_of_Release']
df_gms['Year_of_Release'] = df_gms['Year_of_Release']

print()

print("Verificación de valores duplicados para asegurar consistencia en el análisis.")


(df_gms['Year_of_Release'].dtype)

(df_gms['Year_of_Release'])

print("Análisis de valores ausentes por columna para evaluar su impacto en el estudio.")

print()


df_gms.duplicated().sum()

print(df_gms.columns)

print()

df_gms.rename(columns={'Name': 'name',
                       'Platform': 'platform',
                       'Year_of_Release': 'year_of_release',
                       'Genre': 'genre',
                       'NA_sales': 'na_sales',
                       'EU_sales': 'eu_sales',
                       'JP_sales': 'jp_sales',
                       'Other_sales': 'other_sales',
                       'Critic_Score': 'critic_score',
                       'User_Score': 'user_score',
                       'Rating': 'rating'}, inplace=True)

print()

print(df_gms.columns)

print()

print()

print("Análisis de valores ausentes por columna para evaluar su impacto en el estudio.")


n_rows = df_gms.shape[0]
100 * df_gms.isna().sum() / n_rows


df_gms['user_score'].isna().sum()

print()

print("Evaluación de valores ausentes en critic_score y user_score; se mantienen cuando no es posible imputar.")

print()

print()

print("Reemplazo de valores 'tbd' en user_score por NaN para evitar distorsiones en análisis futuros.")

print()


tbd_count = df_gms['user_score'].str.contains('tbd').sum()
print(tbd_count)


print()

df_gms['user_score'] = df_gms['user_score'].replace('tbd', np.nan)
tbd_count

print()

n_rows = df_gms.shape[0]
100 * df_gms.isna().sum() / n_rows

print("Revisión del estado general del dataset tras el preprocesamiento.")

print(df_gms.describe())

print()


print("Cálculo de ventas globales sumando las ventas de todas las regiones.")

print()

all_sales = df_gms['jp_sales'] + df_gms['eu_sales'] + \
    df_gms['na_sales'] + df_gms['other_sales']
df_gms['all_sales'] = [0 if x < 0 else x for x in all_sales]
print(df_gms.query('all_sales > 0').head(2))

print("Se crea una nueva columna **'all_sales'**, que suma las ventas de cada reguion en especifico para cada juego.Ahora podremos tomarla en cuenta para seguir con nuestro analisis.")

print()

games_per_year = df_gms['year_of_release'].value_counts()
games_per_year.reset_index().head()

print()

print("Evaluación de la evolución del mercado de videojuegos a lo largo del tiempo.")

print()


games_per_year = df_gms.groupby(
    'year_of_release').size().reset_index(name='num_games')
games_per_year.head()

print()


print(games_per_year.tail())


print("Análisis de la cantidad de videojuegos lanzados por año.")

print()

sales_by_platform = df_gms.groupby(
    'platform')['year_of_release'].sum().sort_values(ascending=False)
sales_by_platform = df_gms.groupby(
    'platform')['all_sales'].sum().sort_values(ascending=False)
sales_by_platform = sales_by_platform.reset_index()
print(sales_by_platform.head())


print()

print("Análisis de ventas globales por plataforma para identificar las más rentables.")

print()

plt.figure(figsize=(10, 6))
plt.bar(sales_by_platform['platform'],
        sales_by_platform['all_sales'], color='yellow')
plt.xlabel('Plataforma')
plt.ylabel('Ventas globales (en millones)')
plt.title('Ventas globales por plataforma')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print()

filtered_df = df_gms[(df_gms['year_of_release'] >= 2012)
                     & (df_gms['year_of_release'] <= 2016)]
sales_by_year_platform = filtered_df.groupby(['year_of_release', 'platform'])[
    'all_sales'].sum().unstack(fill_value=0)

print()

print("Análisis de ventas globales por plataforma para identificar las más rentables.")

print()

print("Visualización de ventas por plataforma mediante gráficos comparativos.")

print()

sales_by_year_platform.plot(kind='area', stacked=True, figsize=(12, 6))
plt.xlabel('Año de lanzamiento')
plt.ylabel('Ventas globales (en millones)')
plt.title('Distribución de ventas por año y plataforma')
plt.legend(title='Plataforma', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

print()

print("Ahora vamos a buscar las plataformas que solían ser populares pero que ahora no tienen ventas, para verificar cuánto tardan generalmente las nuevas plataformas en aparecer.")

print()

sales_by_platform_min = df_gms.groupby(
    'platform')['all_sales'].sum().sort_values()
print(sales_by_platform_min.reset_index().head())

print()


platform_sales_by_year = df_gms.groupby(['platform', 'year_of_release'])[
    'all_sales'].sum().reset_index()
print(platform_sales_by_year)

print()

print(sales_by_platform_min.reset_index().tail())


print()

print("Estas son las 5 plataformas que han generado mayor cantiad de ventas en los ultimoos años.")

print()

print("Identificación de plataformas con menor y mayor volumen de ventas históricas.")

print()


df_gms_e = df_gms[df_gms['year_of_release'] >= 2012]
sales_by_platform_mx = df_gms.groupby(['all_sales', 'year_of_release'])[
    'platform'].sum().sort_values()
print(sales_by_platform_mx.reset_index().head())


print("Análisis del comportamiento de ventas por plataforma y año de lanzamiento.")

print()

sales_by_platform_mx = df_gms_e.groupby(['platform', 'year_of_release'])[
    'all_sales'].sum().reset_index()
plt.figure(figsize=(12, 6))
for platform in sales_by_platform_mx['platform'].unique():
    platform_data = sales_by_platform_mx[sales_by_platform_mx['platform'] == platform]
    plt.bar(platform_data['year_of_release'],
            platform_data['all_sales'], label=platform)

plt.xlabel('Año de Lanzamiento')
plt.ylabel('Ventas Globales (en millones)')
plt.title('Ventas Globales por Plataforma y Año de Lanzamiento')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print()

platform_medians = df_gms.groupby(
    'platform')['all_sales'].median().sort_values()
sorted_platforms = platform_medians.index

print("Determinación del ciclo de vida promedio de una plataforma en el mercado.")


print()

print()


plt.figure(figsize=(12, 8))
plt.boxplot([df_gms[df_gms['platform'] == platform]['all_sales']
             for platform in df_gms['platform'].unique()],
            labels=df_gms['platform'].unique())

plt.xlabel('Plataforma')
plt.ylabel('Ventas globales (en millones)')
plt.title('Diagrama de Caja de Ventas Globales por Plataforma')
plt.xticks(rotation=90)
plt.grid(True)
plt.ylim(0, 5)
plt.show()


print(df_gms.columns)

print("Análisis de la distribución de ventas globales por plataforma mediante diagramas de caja.")


print()


print("Evaluación de diferencias significativas en ventas entre plataformas.")

print()


ps_df = df_gms[df_gms['platform'] == 'PS'][[
    'user_score', 'critic_score', 'all_sales']]
print(ps_df.reset_index().head())


print()

ps_df = df_gms[df_gms['platform'] == 'PS'][[
    'user_score', 'critic_score', 'all_sales']].dropna()
print(ps_df.reset_index().head())

print("Análisis del impacto de las reseñas de usuarios y críticos en las ventas de una plataforma popular.")

print()

ps_df['user_score'] = pd.to_numeric(ps_df['user_score'], errors='coerce')
ps_df['critic_score'] = pd.to_numeric(ps_df['critic_score'], errors='coerce')

print()


print("Cálculo de correlaciones entre puntuaciones y ventas globales.")

plt.figure(figsize=(8, 6))
plt.scatter(ps_df['user_score'], ps_df['all_sales'],
            color='blue', label='Usuario', alpha=0.5)
plt.scatter(ps_df['critic_score'], ps_df['all_sales'],
            color='red', label='Crítico', alpha=0.5)
plt.title('Gráfico de dispersión: Reseñas de Usuarios y Críticos vs Ventas Globales (Plataforma PS)')
plt.xlabel('Puntuación')
plt.ylabel('Ventas Globales (en millones)')
plt.legend()
plt.grid(True)
user_correlation = ps_df['user_score'].corr(ps_df['all_sales'])
critic_correlation = ps_df['critic_score'].corr(ps_df['all_sales'])
print("Correlación entre reseñas de usuarios y ventas globales en PS:", user_correlation)
print("Correlación entre reseñas de críticos y ventas globales en PS:",
      critic_correlation)

plt.show()

print()

print("Comparación del peso de las reseñas de críticos frente a las de usuarios.")

print()


plataformas = ['PS4', 'X360', 'XOne']
juegos_populares = pd.DataFrame()

for plataforma in plataformas:
    juegos_plataforma = df_gms[df_gms['platform']
                               == plataforma].nlargest(5, 'all_sales')

    juegos_populares = pd.concat([juegos_populares, juegos_plataforma])

juegos_populares.reset_index(drop=True, inplace=True)


print(juegos_populares.reset_index().head(9))

print()

print("Comparación del peso de las reseñas de críticos frente a las de usuarios.")

print()

print("Identificación de los juegos más vendidos en las plataformas líderes del mercado.")

print()


juegos_interes = ['Grand Theft Auto V', 'Call of Duty: Black Ops 3', 'FIFA 18']

df_interes = df_gms[df_gms['name'].isin(juegos_interes)]

ventas_juegos_interes = df_interes[['name', 'platform', 'all_sales']]
print(ventas_juegos_interes.reset_index())

print("Analisis por genero")

print()

print("verificar los generos de mayor interes y determinar de esta forma, si podemos generalizar acerca de los géneros con ventas altas y bajas.")

print()

genre_counts = df_gms['genre'].value_counts()
(genre_counts.reset_index())

print()

print("Aqui contamos la cantidad de juegos en cada género utilizando value_counts() en la columna 'genre'. Tambien puede ser prudente generar un grafico de barras para visualizarla  de mejor manera.")

# %%
plt.figure(figsize=(10, 6))
genre_counts.plot(kind='bar', color='orange')
plt.title('Distribución de juegos por género')
plt.xlabel('Género')
plt.ylabel('Cantidad de juegos')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print()

print("Y tambien realizaremos una agrupacion con la columna 'all_sales', para verificar cuales generos son los mas rentables que tenemos")
# %%
ventas_por_genero = df_gms.groupby(
    'genre')['all_sales'].sum().sort_values(ascending=False)
print(ventas_por_genero.reset_index().head())


print("Con este enfoque, podemos analizar las ventas totales de cada género  en función de estas ventas totales en general.")

ventas_por_genero = ventas_por_genero.reset_index()
print(ventas_por_genero)


print("Primero convertimos la Serie ventas_por_genero en un DataFrame utilizando el método reset_index(), y luego procedemos a graficar las ventas por género.")
plt.figure(figsize=(10, 6))
plt.bar(ventas_por_genero['genre'],
        ventas_por_genero['all_sales'], color='silver')
plt.xlabel('Género')
plt.ylabel('Ventas totales')
plt.title('Ventas totales por género')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print()

print("Ahora determinaremos las 5 plataformas principales para cada reguion especifica.")

print()

region_plataforma_sls = df_gms.groupby('platform')[
    ['na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'all_sales']].sum()
pricpl_region_pltsfrm = region_plataforma_sls.apply(
    lambda x: x.nlargest(5), axis=0)
pricpl_region_pltsfrm.head()

print()

print()

df_gms_clean = df_gms.dropna(
    subset=['na_sales', 'eu_sales', 'jp_sales', 'other_sales'])
df_gms_clean = pricpl_region_pltsfrm.dropna()
df_gms_clean.head()


for region in ['na_sales', 'eu_sales', 'jp_sales', 'other_sales']:
    region_plataforma_sls = pricpl_region_pltsfrm.sort_values(
        by=region, ascending=False)
    top_plataformas = region_plataforma_sls.head(5)
    # Recorrer columnas de top_plataformas y asignar valores por columna
    for col in top_plataformas.columns:
        # Asignación por columna
        pricpl_region_pltsfrm[region] = top_plataformas[col]
    top_plataformas.dropna()
print(top_plataformas.head())

print("Ahora compararemos estas cuotas de mercado entre las diferentes regiones para cada una de las cinco plataformas principales.")

print()

print("Para esto generaremos diferentes graficos de barras para poder describir, las variaciones en las cuotas de mercado de una región a otra.")
ventas_por_region_y_plataforma = df_gms.groupby(
    'platform')[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum()
plataformas_principales_por_region = {}
for region in ['na_sales', 'eu_sales', 'jp_sales', 'other_sales']:
    region_plataforma_sls = ventas_por_region_y_plataforma.sort_values(
        by=region, ascending=False)
    top_plataformas = region_plataforma_sls.head(5)
    plataformas_principales_por_region[region] = top_plataformas
top_plataformas.dropna()

print(top_plataformas)


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
for i, (region, top_plataformas) in enumerate(plataformas_principales_por_region.items()):
    ax = axes[i // 2, i % 2]
    top_plataformas.plot(kind='bar', ax=ax)
    ax.set_title(f"Top 5 Plataformas en {region}")
    ax.set_xlabel("Plataforma")
    ax.set_ylabel("Ventas (en millones)")
    ax.grid(axis='y')
plt.tight_layout()
plt.show()


print("A continuacion vamor a determinar los cinco géneros principales. y a verificar sus diferencias principales.")
sales_gnre = df_gms.groupby('genre')['all_sales'].sum()
sales_gnre = sales_gnre.sort_values(ascending=False)
sales_gnre.reset_index().head(5)
print()

print()

df_gms_2014 = df_gms[df_gms['year_of_release'] >= 2014]
sales_gnre.plot(kind='bar', figsize=(10, 6), color='magenta')
plt.title('Cinco Géneros Principales por Ventas Totales')
plt.xlabel('Género')
plt.ylabel('Ventas Totales (en millones)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
print()

print()

print("Tambien es importante determinar si las clasificaciones de ESRB afectan a las ventas en regiones individuales. Asi que vamos  a revisar esta parte tambien.")
df_gms_2014 = df_gms[df_gms['year_of_release'] >= 2014]
regiones = ['na_sales', 'eu_sales', 'jp_sales', 'other_sales']
ventas_por_esrb_y_region = df_gms_2014.groupby(
    ['rating'])[regiones].sum().reset_index()
ventas_por_esrb_y_region

# %% [markdown]
print("Vamos a generar un grafico de barras para visualizar esta tabla")
plt.figure(figsize=(10, 6))
for region in regiones:
    plt.bar(ventas_por_esrb_y_region['rating'],
            ventas_por_esrb_y_region[region], label=region)
plt.title('Ventas por Clasificación ESRB y Región')
plt.xlabel('Clasificación ESRB')
plt.ylabel('Ventas (en millones)')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()
print()

print()

print("Este gráfico permite visualizar fácilmente cómo varían las ventas de videojuegos por clasificación ESRB en diferentes regiones.")
print("Cada barra representa una clasificación ESRB y los diferentes colores de las barras corresponden a las diferentes regiones.")

print()


print("Asi determinamos que las calificasiones de ESRB afesctan las ventas de ciertas regiones individuales como 'jp_sales' o 'eu_sales,")
print("ya que estas son las reguiones mas irregulares segun su calificasion especifica. Aun asi, podemos ver una notoria alza de resultados por la calificasion 'M' de ESRB**")

print()

print("Al tener demasiada similitud en las calificasiones de los usuarios en genral para las plataformas 'XboxOne' y 'PC', vamos a generar nuestra hipoteis nula basandonos en el hecho de que las calificasiones para estas platformas son las mismas.")

print()

print("Para ello, lo primero sera filtar los datos y eliminar los valores auusentes en las columnas de interes.")
xbox_one_ratings = df_gms[df_gms['platform'] ==
                          'XOne']['user_score'].dropna().astype('float')
pc_ratings = df_gms[df_gms['platform'] ==
                    'PC']['user_score'].dropna().astype('float')
xbox_one_ratings.reset_index().head()

print()


print(pc_ratings.reset_index().head())


print()

print("Generamos asi 2 tablas. 'xbox_one_ratings' que muestra las calificasiones de los usuarios para la plataforma 'XOne' y 'pc_ratings' que a su vez, nos indica las calificasiones para plataforma 'PC', y tambien realizamos una descripcion de las series.")


print()

print("Generacion de Hipotesis 1.")


print()

print("H0 = Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.***")

print()

print("H1 = Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son diferentes.")

print()

calificaciones_xbox = xbox_one_ratings
calificaciones_pc = pc_ratings

print(t_statistic, p_value=stats.ttest_ind(
    calificaciones_xbox, calificaciones_pc))

print()


print("Estadístico t:", t_statistic)
print("Valor p:", p_value)


alpha = 0.05
if p_value < alpha:
    print("Hay evidencia suficiente para rechazar la hipótesis nula.")
    print("Las calificaciones promedio son diferentes entre las dos plataformas.")
else:
    print("No hay suficiente evidencia para rechazar la hipótesis nula.")
    print("No hay diferencias significativas en las calificaciones promedio entre las dos plataformas.")

print()

print()

print("De esta manera concluimos que nuesta hipotesis alternativa H1, la cual dicta que **Las calificaciones promedio de los usuarios para las plataformas 'XboxOne' y 'PC' son diferentes** es la correcta segun este test.")
print("Rechazando de esta manera la hipotesis nula H0, debido a que tenemos una diferencia estadistica significativa entre el promedio de las calificasiones para ambas plataformas.")


print()

print("Generacion de Hipotesis 2.")

print()

print("La siguiente hipotesis a comprobar es sobre la calificasion en los generos mas populares.")
print("En este caso, lo que vamos a probar es que a pesar de que existen muchas similitudesen las calificasiones de los generos de Accion y Deportes, en promedio son diferentes.")

print()


print("H0 = Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.")

print()

print("H1 = Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son las mismas.")

print()


genres_of_interest = ['Action', 'Sports']
filtered_df = df_gms[df_gms['genre'].isin(genres_of_interest)]
print(filtered_df.head())

print()

print("Filtramos el conjunto de datos  en primera instancia para incluir solo las calificaciones de los usuarios y los géneros de Acción y Deportes.")
print("Utilizamos el método isin() de Pandas para crear un filtro booleano que selecciona solo las filas donde el valor de la columna 'genre' coincide con alguno de los géneros de interés.")

print()

print("Luego realizamos un analisis esdtadistico de los datos para calcular y despues compara las estadisticas descriptivas.")

print()

action_ratings = filtered_df[filtered_df['genre']
                             == 'Action']['user_score'].dropna().astype(float)
sports_ratings = filtered_df[filtered_df['genre']
                             == 'Sports']['user_score'].dropna().astype(float)
action_mean = action_ratings.mean()
print(action_mean)


sports_mean = sports_ratings.mean()
sports_mean

print()


levene_statistic, levene_p_value = stats.levene(action_ratings, sports_ratings)
equal_var = True
if levene_p_value < alpha:
    print("Las varianzas no son iguales. Usaremos equal_var=False en la prueba t.")
    equal_var = False
else:
    print("No se puede rechazar la hipótesis nula de igualdad de varianzas. Usaremos equal_var=True en la prueba t.")

print()

t_statistic, p_value = stats.ttest_ind(action_ratings, sports_ratings)

print("Estadístico t:", t_statistic)
print("Valor p:", p_value)

alpha = 0.05
if p_value < alpha:
    print("Hay evidencia suficiente para rechazar la hipótesis nula.")
    print("Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.")
else:
    print("No hay suficiente evidencia para rechazar la hipótesis nula.")
    print("No hay diferencias significativas en las calificaciones promedio de los usuarios para los géneros de Acción y Deportes.")

print()

print("De esta forma nos quedamos con nuestra hipostesis nula H0, la cual dicta que **Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.")
print("Sin embargo esta diferencia promedio entre puntajes no es tan significativa para ambos ambos generos como tal, ya que tienen una difecia de 0.1 en el puntaje para estos generos.")
print("Tanto Las hipotesis nulas y alternativas fueron generadas a partir de la similitud en tre las calificasiones para ciertas plataformas y generos de juegos en especifico, lo cual es importante,")
print("tener presente para la proyeccion del procimo año, ya que de esta manera ahora sabremos mas exactamente es la plataforma y el genero que mas  ha agradado a los usuarios en general, y podriamos empezar ha hacer enfasis en ello.")
print("Para generar las hipotesis se considero el citerio de la importancia  por que estos generos y plataformas son las que han tenido mejores resultados en el ultimo año, con el obsejitivo de empezar por los generos y plataformnas especificas, que sean mas rentables en los ultimos años.")
