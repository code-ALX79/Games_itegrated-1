🎮 **Proyecto: Análisis de Videojuegos**

📊 Exploración de Factores que Influyen en el Éxito Comercial de los Videojuegos

**Autor:** Alexander Herrera

**Lenguaje:** Python

**Librerías utilizadas:** pandas, numpy, matplotlib, scipy

**Tipo de proyecto:** Análisis Exploratorio de Datos (EDA)

**Nivel:** Analista de Datos Jr. — Intermedio


⚙️ **Configuración del entorno**

Para garantizar la correcta ejecución del proyecto, se recomienda crear un entorno virtual y utilizar las dependencias listadas en requirements.txt.


```
1️⃣ Crear el entorno virtual
python -m venv venv

2️⃣ Activarlo (Windows)
. ./venv/Scripts/activate

3️⃣.2️⃣  Activarlo (Mac / Linux)
source venv/bin/activate

 4️⃣ clonar el repositortio
git clone https://github.com/code-ALX79/Credit_Scoring_Analitics.git

5️⃣ Instalar las dependencias
pip install -r requirements.txt

```

🧩 **Descripción general**

Este proyecto tiene como objetivo **analizar datos históricos de videojuegos** para identificar **patrones que influyen en su éxito comercial**, considerando variables como:

- Plataforma

- Género

- Ventas por región

- Calificaciones de críticos y usuarios

- Año de lanzamiento

El análisis se centra en comprender **qué factores están más relacionados con mayores ventas**, y cómo estas relaciones pueden variar según la región o el tipo de videojuego.

Se aplican técnicas de **limpieza, transformación y análisis exploratorio de datos (EDA)** utilizando Python y librerías estándar del ecosistema de análisis de datos.


🎯 **Objetivos del proyecto**

1️⃣ Analizar la evolución de las **ventas de videojuegos a lo largo del tiempo.**

2️⃣ Identificar las **plataformas y géneros más relevantes** en términos de ventas.

3️⃣ Evaluar la relación entre **reseñas (críticos y usuarios)** y el desempeño comercial.

4️⃣ Comparar el comportamiento de ventas entre **diferentes regiones.**

5️⃣ Aplicar un flujo de trabajo analítico **claro, reproducible y documentado**, siguiendo buenas prácticas.


⚙️ **Estructura del proyecto**


```
Platforms_video-games_analytics/
│
├── data/
│   └── games_data.csv            # Dataset utilizado para el análisis
│
├── notebooks/
│   └── proyect_games.ipynb       # Notebook con la exploración y análisis de datos (EDA)
│
├── scripts/
│   └── games_proyect.py          # Script en Python con el flujo del análisis
│
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Documentación del proyecto
```

**Etapas del análisis**

1️⃣ **Carga y exploración inicial**

- Lectura del dataset con ```pandas.read_csv()```

- Revisión general con ```.info()```, ```.head()``` y ```.describe()```

- Identificación de valores nulos y tipos de datos inconsistentes

2️⃣ **Limpieza y preparación de datos**

- Tratamiento de valores faltantes

- Conversión de tipos de datos

- Normalización de nombres y categorías

- Eliminación de registros irrelevantes para el análisis

3️⃣ **Análisis exploratorio (EDA)**

- Distribución de ventas globales y por región

- Comparación de plataformas y géneros

- Análisis temporal de lanzamientos

- Evaluación de la relación entre reseñas y ventas

4️⃣ **Visualización de datos**

- Gráficos de barras y líneas

- Histogramas de distribución

- Análisis comparativo entre variables clave
(Todo utilizando ```matplotlib```)


📌 **Principales hallazgos (ejemplo)**

- Algunas **plataformas concentran la mayor parte de las ventas**, pero su popularidad varía con el tiempo.

- Ciertos **géneros muestran un mejor desempeño comercial de forma consistente**.

- Las **reseñas de críticos tienen mayor correlación con ventas** que las de usuarios en algunos casos.

- Existen diferencias claras entre regiones en cuanto a preferencias de videojuegos.

🚀 **Cómo ejecutar el proyecto**

- Ejecutar el notebook:

```jupyter notebook notebooks/proyect_games.ipynb```

- Ejecutar el script:

```python scripts/games_proyect.py```


🤝 **Próximos pasos y colaboración**

Este proyecto está **abierto a mejoras y nuevas perspectivas**.
Se invita a **analistas y científicos de datos** a clonar el repositorio y aportar con:

- Modelos predictivos de ventas

- Análisis más avanzados por región

- Nuevas visualizaciones o dashboards

- Optimización del flujo de análisis

Si te interesa explorar los datos desde otro enfoque, ¡tu contribución será bienvenida!