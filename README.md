# Spaceship Titanic — Predição de Transporte

Atividade Avaliativa Final — Disciplina: Inteligência Computacional
Curso Superior de Tecnologia em Ciência de Dados — Fatec Jundiaí
Professor:  Mateus Guilherme Fuini

---

## Integrantes

- Julio Cesar Rufino Rocha
- Christian Ryu Kondo Cannavan
- Caio Roberto Farias Saraiva

---

## Descrição do Problema

O dataset **Spaceship Titanic** é uma competição do Kaggle que simula um cenário de ficção científica: no ano de 2912, a nave espacial *Spaceship Titanic* colidiu com uma anomalia espaço-temporal e aproximadamente metade dos passageiros foi transportada para uma dimensão alternativa.

O objetivo é **prever quais passageiros foram transportados** (`Transported = True/False`) com base em informações demográficas, de cabine e de gastos a bordo da nave.

- **Tipo de problema:** Classificação binária
- **Variável alvo:** `Transported`
- **Fonte:** [Kaggle — Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

---

## Dataset

| Característica | Detalhe |
|---|---|
| Registros (treino) | 8.693 |
| Colunas | 14 |
| Valores ausentes | Sim — presentes em quase todas as colunas |
| Variáveis categóricas | `HomePlanet`, `Destination`, `CryoSleep`, `VIP`, `Cabin` |
| Variáveis numéricas | `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` |

### Colunas principais

| Coluna | Descrição |
|---|---|
| `PassengerId` | ID único do passageiro |
| `HomePlanet` | Planeta de origem |
| `CryoSleep` | Se o passageiro estava em criosono |
| `Cabin` | Cabine no formato `Deck/Número/Lado` |
| `Destination` | Planeta de destino |
| `Age` | Idade do passageiro |
| `VIP` | Se o passageiro é VIP |
| `RoomService` ... `VRDeck` | Gastos em serviços a bordo |
| `Transported` | **Target** — se foi transportado |

---

## Solução Desenvolvida

A solução foi desenvolvida em Python com Scikit-learn, seguindo um fluxo completo de Ciência de Dados:

### 1. Análise Exploratória de Dados (EDA)
- Análise de valores ausentes por coluna
- Estatísticas descritivas (média, mediana, desvio padrão, skewness)
- Histogramas das variáveis numéricas
- Boxplots para identificação de outliers nos gastos a bordo
- Scatterplot de gastos vs. variável alvo
- Heatmap de correlação
- Taxa de transporte por variável categórica

### 2. Engenharia de Atributos

Foram criados 4 novos atributos derivados dos dados originais:

| Atributo | Origem | Justificativa |
|---|---|---|
| `TotalSpend` | Soma dos 5 gastos a bordo | Captura o comportamento de consumo geral do passageiro |
| `Deck` | Extração de `Cabin` | A posição na nave pode influenciar a probabilidade de transporte |
| `Side` | Extração de `Cabin` | Lado da cabine (P/S) como possível fator preditivo |
| `FaixaEtaria` | Agrupamento de `Age` | Reduz impacto de outliers e facilita aprendizagem do modelo |

### 3. Tratamento de Outliers

Os boxplots das variáveis de gastos a bordo revelam valores extremos expressivos. A decisão foi **manter os outliers** pelas seguintes razões:

- Os valores extremos representam passageiros que gastaram muito em serviços premium — são ocorrências reais, não erros de medição
- Passageiros em CryoSleep têm gastos zerados obrigatoriamente, o que explica a forte assimetria e é um sinal preditivo importante
- O `StandardScaler` no pipeline reduz o impacto dos outliers na escala das variáveis
- O atributo `TotalSpend` agrega os gastos, suavizando o efeito de valores extremos em colunas individuais

Caso fosse necessário tratar os outliers, a abordagem recomendada seria o **clipping por IQR**, preservando as observações mas limitando sua influência.

### 4. Pré-processamento com Pipeline

Utilizamos `Pipeline` e `ColumnTransformer` do Scikit-learn para organizar as transformações de forma segura e sem data leakage:

**Variáveis numéricas:**
- `SimpleImputer(strategy='median')` — imputação pela mediana, mais robusta a outliers
- `StandardScaler()` — normalização para algoritmos baseados em distância

**Variáveis categóricas:**
- `SimpleImputer(strategy='most_frequent')` — imputação pela moda
- `OneHotEncoder(handle_unknown='ignore')` — codificação binária sem hierarquia implícita

### 5. Prevenção de Data Leakage

- Divisão treino/teste realizada **antes** de qualquer transformação
- O `fit` do pré-processador ocorre **apenas nos dados de treino**
- O Pipeline garante que o conjunto de teste receba somente o `transform`

### 6. Modelagem

Foram aplicados dois algoritmos supervisionados:

- `LogisticRegression` — modelo linear interpretável, boa baseline
- `KNeighborsClassifier` — modelo baseado em distância, otimizado via GridSearchCV

### 7. Validação

**K-Fold Cross Validation (k=5):**

| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Média | Desvio |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.7692 | 0.8030 | 0.8066 | 0.8009 | 0.7950 | **0.7949** | 0.0134 |
| KNN (default) | 0.7541 | 0.7728 | 0.7635 | 0.7692 | 0.7799 | **0.7679** | 0.0087 |

A Regressão Logística apresentou média superior (79,49%) com desvio padrão de 1,34%. O KNN padrão foi mais estável entre os folds (desvio 0,87%) porém com acurácia média menor (76,79%).

**GridSearchCV — Ajuste de Hiperparâmetros do KNN:**

Foram testadas 20 combinações de parâmetros com 5 folds cada (100 fits no total):

| n_neighbors | weights | metric | Acurácia (CV) |
|---|---|---|---|
| 11 | uniform | euclidean | **0.7849** |
| 15 | uniform | euclidean | 0.7849 |
| 11 | uniform | manhattan | 0.7817 |
| 15 | uniform | manhattan | 0.7816 |
| 7 | uniform | manhattan | 0.7775 |

**Melhores hiperparâmetros:** `n_neighbors=11`, `weights=uniform`, `metric=euclidean`

- `n_neighbors=11`: número de vizinhos que equilibrou sensibilidade e generalização
- `weights=uniform`: todos os vizinhos com o mesmo peso se saiu melhor que ponderado por distância
- `metric=euclidean`: distância euclidiana funcionou melhor que manhattan neste dataset

---

## Resultados

| Modelo | Acurácia (CV) | Acurácia (Teste) |
|---|---|---|
| Logistic Regression | 79,49% | — |
| KNN (default) | 76,79% | — |
| KNN (GridSearchCV) | 78,49% | **78,09%** |

> A acurácia no teste foi avaliada apenas para o modelo final (KNN otimizado). Os demais foram comparados via validação cruzada.

**Métricas detalhadas — KNN otimizado no conjunto de teste (1.739 amostras):**

| Classe | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Não Transportado | 0.77 | 0.80 | 0.78 | 863 |
| Transportado | 0.79 | 0.77 | 0.78 | 876 |
| **Média** | **0.78** | **0.78** | **0.78** | **1739** |

O modelo acertou 78,09% dos passageiros no conjunto de teste. As métricas de precisão e recall ficaram equilibradas entre as duas classes, o que é esperado dado o balanceamento quase perfeito do dataset (~50/50).

---

## Conclusões

- O pré-processamento via Pipeline foi fundamental para garantir um fluxo correto e sem data leakage
- A Regressão Logística obteve a melhor acurácia de validação cruzada (79,49%), mostrando que um modelo linear já captura bem os padrões do dataset
- O KNN melhorou após o ajuste de hiperparâmetros via GridSearchCV, passando de 76,79% para 78,49% no CV e 78,09% no teste final
- A engenharia de atributos, especialmente `TotalSpend` e a extração de `Deck`/`Side` da coluna `Cabin`, enriqueceu a representação dos dados
- Passageiros em CryoSleep apresentaram padrão de transporte distinto, sendo uma das variáveis mais relevantes
- Como melhoria futura, algoritmos como `RandomForestClassifier` ou `GradientBoostingClassifier` poderiam ser explorados para ganhos adicionais de desempenho

---

## Estrutura do Repositório

```
spaceship-titanic/
├── Trabalho_Spaceship_Titanic.ipynb   # Notebook principal com todo o fluxo
└── README.md                          # Este arquivo
```

---

## Como Executar

1. Abra o notebook no [Google Colab](https://colab.research.google.com/)
2. Gere sua chave de API em [kaggle.com](https://www.kaggle.com) → Settings → API → Create New Token
3. Aceite os termos da competição em [kaggle.com/competitions/spaceship-titanic](https://www.kaggle.com/competitions/spaceship-titanic)
4. Execute as células em sequência — a primeira célula solicitará o upload do `kaggle.json`

**Dependências:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `kaggle`
