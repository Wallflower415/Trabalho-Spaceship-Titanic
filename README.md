# Spaceship Titanic — Predição de Transporte

Atividade Avaliativa Final — Disciplina: Inteligência Computacional
Curso Superior de Tecnologia em Ciência de Dados — Fatec Jundiaí
Professor: Me. Mateus Guilherme Fuini

---

## Integrantes

- Julio Rocha
- Christian Cannavan
- Caio Saraiva

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
| Registros (treino) | ~8.700 |
| Colunas | 14 |
| Valores ausentes | Sim — presentes em quase todas as colunas |
| Variáveis categóricas | `HomePlanet`, `Destination`, `CryoSleep`, `VIP`, `Cabin` |
| Variáveis numéricas | `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` |

### Colunas principais

| Coluna | Descrição |
|---|---|
| `PassengerId` | ID único do passageiro |
| `HomePlanet` | Planeta de origem |
| `CryoSleep` | Se o passageiro estava em criossono |
| `Cabin` | Cabine no formato `Deck/Numero/Lado` |
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

- **K-Fold Cross Validation** com `k=5` para estimativa confiável do desempenho
- **GridSearchCV** testando combinações de `n_neighbors`, `weights` e `metric` no KNN

---

## Resultados

| Modelo | Acurácia (CV) | Acurácia (Teste) |
|---|---|---|
| Logistic Regression | 79,49% | 79,36% |
| KNN (default) | 76,79% | 76,77% |
| KNN (GridSearchCV) | 78,49% | 78,09% |

**Melhores hiperparâmetros encontrados pelo GridSearchCV:**
`n_neighbors=11`, `weights='uniform'`, `metric='euclidean'`

---

## Conclusões

- O pré-processamento via Pipeline foi fundamental para garantir um fluxo correto e sem data leakage
- A engenharia de atributos, especialmente `TotalSpend` e a extração de `Deck`/`Side` da coluna `Cabin`, enriqueceu significativamente a representação dos dados
- Passageiros em CryoSleep apresentaram padrão de transporte distinto, sendo uma das variáveis mais relevantes
- O GridSearchCV permitiu encontrar a melhor combinação de hiperparâmetros para o KNN de forma sistemática
- A Regressão Logística superou o KNN otimizado, indicando que a fronteira de decisão deste problema tem caráter predominantemente linear
- Como melhoria futura, algoritmos como `RandomForestClassifier` ou `GradientBoostingClassifier` poderiam ser explorados para ganhos adicionais de desempenho

---

## Estrutura do Repositório

```
Trabalho-Spaceship-Titanic/
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
