# Regressão de Imóveis com Apache Spark

Projeto de **Machine Learning com PySpark** desenvolvido para trabalhar com dados de imóveis e estimar o preço de um imóvel a partir de suas características.

O notebook apresenta um fluxo completo de preparação dos dados, exploração, treinamento de modelos de regressão, avaliação e otimização utilizando recursos do **Apache Spark MLlib**.

## 📌 Sobre o projeto

O projeto utiliza um conjunto de dados de imóveis carregado a partir de um arquivo JSON. Ao longo do notebook são trabalhadas diferentes etapas de um processo de Machine Learning:

- Preparação e exploração dos dados;
- Seleção de features;
- Tratamento de tipos de dados;
- Filtragem de imóveis residenciais;
- Tratamento de dados faltantes;
- Criação de variáveis dummy com `pivot`;
- Transformação das features com `VectorAssembler`;
- Análise de correlação;
- Separação entre dados de treino e teste;
- Regressão Linear;
- Árvore de Decisão para regressão;
- Random Forest para regressão;
- Avaliação utilizando R² e RMSE;
- Cross Validation;
- Otimização de hiperparâmetros;
- Previsão do preço de um novo imóvel.

## 🧠 Tecnologias e bibliotecas

O projeto foi desenvolvido em **Python** utilizando principalmente:

- **Apache Spark / PySpark** - processamento distribuído e Machine Learning;
- **Spark SQL / DataFrame** - manipulação dos dados estruturados;
- **Spark MLlib** - criação e avaliação dos modelos de Machine Learning;
- **Pandas** - organização da matriz de correlação para análise;
- **Matplotlib** - visualização;
- **Seaborn** - geração do mapa de calor de correlação;
- **Google Colab** - ambiente utilizado para execução do notebook.

## 📂 Estrutura do projeto

Uma estrutura recomendada para publicar o projeto no GitHub é:

```text
spark-regressao/
│
├── Spark_regressão.ipynb
├── README.md
└── dados/
    └── imoveis.json
```

> O notebook original carrega o arquivo `imoveis.json` a partir do Google Drive no caminho utilizado no Google Colab.

## ⚙️ Preparação do ambiente

O notebook inicia instalando o PySpark:

```bash
pip install pyspark
```

Em seguida, é criada uma `SparkSession` em modo local:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master('local[*]') \
    .appName("Regressão com Spark") \
    .getOrCreate()
```

## 📊 Carregamento dos dados

Os dados são carregados em formato JSON utilizando o Spark:

```python
dados = spark.read.json(
    '/content/drive/MyDrive/Alura/imoveis.json'
)
```

O notebook utiliza o Google Drive para acessar o arquivo quando executado no Google Colab.

## 🔎 1. Seleção de features

O conjunto original possui estruturas aninhadas. O projeto seleciona informações relacionadas ao imóvel, incluindo:

- Identificação do cliente;
- Tipo do imóvel;
- Características do imóvel;
- Endereço;
- Preço;
- Condomínio;
- IPTU.

Algumas colunas são removidas durante a preparação, como:

```text
city
location
totalAreas
```

## 🧹 2. Tratamento dos dados

Algumas colunas são convertidas para tipos numéricos adequados:

- `usableAreas` → `IntegerType`;
- `price` → `DoubleType`;
- `condo` → `DoubleType`;
- `iptu` → `DoubleType`.

O projeto também filtra os registros para considerar apenas imóveis de uso **Residencial**.

### Dados faltantes

Os valores nulos e `NaN` são identificados utilizando funções do PySpark e, posteriormente, os valores ausentes são preenchidos com `0`.

Também são removidos registros em que a coluna `zone` está vazia.

## 🔢 3. Criação de variáveis dummy

As variáveis categóricas `unit` e `zone` são transformadas em variáveis dummy utilizando `groupBy`, `pivot` e `fill`.

As informações são posteriormente incorporadas ao dataset por meio de `join` pelo `customerID`.

## 🧮 4. Definição das features

A coluna `price` é renomeada para `label`, passando a representar o valor que o modelo deve prever.

As features utilizadas no modelo são:

```text
bathrooms
bedrooms
floors
parkingSpaces
suites
unitFloor
unitsOnTheFloor
usableAreas
condo
iptu
Apartamento
Casa
Outros
Zona Central
Zona Norte
Zona Oeste
Zona Sul
```

O `VectorAssembler` é utilizado para reunir essas variáveis em um único vetor chamado `features`.

```python
assembler = VectorAssembler(
    inputCols=X,
    outputCol='features'
)
```

## 📈 5. Exploração dos dados

O projeto calcula a correlação entre as features utilizando:

```python
from pyspark.ml.stat import Correlation
```

A matriz de correlação é convertida para um DataFrame do Pandas e apresentada em um **heatmap** utilizando Matplotlib e Seaborn.

Essa etapa permite visualizar a relação entre as variáveis utilizadas na regressão.

## 📐 6. Regressão Linear

A primeira técnica de Machine Learning aplicada é a **Regressão Linear**.

Os dados são separados em:

- **70% para treinamento**;
- **30% para teste**.

A divisão utiliza `randomSplit` com `seed=101`.

```python
treino, teste = dataset_prep.randomSplit(
    [0.7, 0.3],
    seed=101
)
```

O modelo é treinado utilizando `LinearRegression`.

### Métricas

O desempenho é analisado utilizando:

- **R² (coeficiente de determinação)**;
- **RMSE (Root Mean Squared Error)**.

As métricas são calculadas tanto para os dados de treinamento quanto para os dados de teste.

## 🌳 7. Árvore de Decisão - Regressão

Na terceira etapa do projeto é utilizado o `DecisionTreeRegressor`.

O modelo é configurado inicialmente com:

```python
DecisionTreeRegressor(
    seed=101,
    maxDepth=7
)
```

Depois do treinamento, as previsões são avaliadas utilizando R² e RMSE, permitindo comparar o desempenho da Árvore de Decisão com a Regressão Linear.

## 🌲 8. Random Forest - Regressão

O projeto também utiliza o `RandomForestRegressor`.

A configuração inicial utiliza:

```python
RandomForestRegressor(
    seed=101,
    maxDepth=7,
    numTrees=10
)
```

Assim como nos modelos anteriores, as previsões são avaliadas utilizando R² e RMSE para os conjuntos de treino e teste.

## 🔧 9. Técnicas de otimização

O notebook utiliza **Cross Validation** para buscar melhores configurações dos modelos.

Para isso são utilizados:

- `CrossValidator`;
- `ParamGridBuilder`;
- `RegressionEvaluator`.

### Árvore de Decisão com Cross Validation

Para o `DecisionTreeRegressor`, são testados diferentes valores de:

```text
maxDepth: 2, 5, 10
maxBins: 10, 32, 45
```

A validação utiliza **3 folds**.

O desempenho do modelo com Cross Validation é comparado ao modelo sem essa etapa por meio de R² e RMSE.

### Random Forest com Cross Validation

Para o `RandomForestRegressor`, são testados:

```text
numTrees: 10, 20, 30
maxDepth: 5, 10
maxBins: 10, 32, 45
```

Também são utilizados **3 folds** de Cross Validation.

O modelo otimizado é então comparado ao Random Forest sem Cross Validation.

## 🏠 10. Previsão de um novo imóvel

Ao final do notebook é criado um exemplo de novo imóvel com características como:

- 2 banheiros;
- 2 quartos;
- 2 andares;
- 1 vaga de estacionamento;
- 1 suíte;
- 200 m² de área útil;
- condomínio de 200;
- imóvel do tipo Casa;
- localizado na Zona Sul.

Esses dados são transformados em um DataFrame Spark e posteriormente vetorizados com o mesmo `VectorAssembler` utilizado no treinamento.

O melhor modelo obtido com Cross Validation é utilizado para gerar a previsão do imóvel:

```python
modelo_rfr_cv.transform(meu_lar_vetorizado).show()
```

## 🔄 Fluxo do projeto

```text
Dados de imóveis
       ↓
Carregamento com PySpark
       ↓
Seleção de features
       ↓
Tratamento dos dados
       ↓
Dados faltantes
       ↓
Variáveis dummy
       ↓
VectorAssembler
       ↓
Análise de correlação
       ↓
Divisão treino / teste
       ↓
┌───────────────────────┐
│ Modelos de Regressão  │
├───────────────────────┤
│ Regressão Linear      │
│ Árvore de Decisão     │
│ Random Forest         │
└───────────────────────┘
       ↓
R² e RMSE
       ↓
Cross Validation
       ↓
Otimização
       ↓
Previsão de novo imóvel
```

## 🎯 Objetivo de aprendizado

O projeto foi desenvolvido para praticar o uso do **Apache Spark aplicado a Machine Learning**, especialmente em problemas de regressão.

Os principais conceitos trabalhados são:

- PySpark;
- SparkSession;
- Spark DataFrame;
- Spark SQL;
- Preparação de dados;
- Variáveis dummy;
- VectorAssembler;
- Correlação;
- Regressão Linear;
- Árvores de Decisão;
- Random Forest;
- R²;
- RMSE;
- Treino e teste;
- Cross Validation;
- Grid Search de hiperparâmetros;
- Previsão de novos dados.

## 🚀 Como executar

### 1. Instale o PySpark

```bash
pip install pyspark
```

### 2. Abra o notebook

Abra o arquivo:

```text
Spark_regressão.ipynb
```

O notebook pode ser executado em um ambiente compatível com Jupyter ou Google Colab.

### 3. Disponibilize o dataset

O notebook foi estruturado para acessar o arquivo `imoveis.json` pelo Google Drive:

```text
/content/drive/MyDrive/Alura/imoveis.json
```

Caso execute o projeto fora do Google Colab, será necessário adaptar o caminho do arquivo para a localização do dataset no seu ambiente.

### 4. Execute as células em ordem

As etapas dependem de objetos criados anteriormente, portanto recomenda-se executar o notebook sequencialmente.

## 📚 Organização do notebook

O conteúdo está organizado em cinco etapas principais:

```text
Aula 1 - Preparando os Dados
Aula 2 - Regressão Linear
Aula 3 - Árvore de Decisão - Regressão
Aula 4 - Random Forest - Regressão
Aula 5 - Técnicas de Otimização
```

## 👨‍💻 Autor

Projeto desenvolvido como estudo prático de **Machine Learning, regressão e processamento de dados com Apache Spark e PySpark**.
