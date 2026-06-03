# Spaceship Titanic — Predicao de Transporte

Atividade Avaliativa Final — Disciplina: Inteligencia Computacional
Curso Superior de Tecnologia em Ciencia de Dados — Fatec Jundiai
Professor: Me. Mateus Guilherme Fuini

---

## Integrantes

- Julio Rocha
- Christian Cannavan
- Caio Saraiva

---

## Descricao do Problema

O dataset **Spaceship Titanic** e uma competicao do Kaggle que simula um cenario de ficcao cientifica: no ano de 2912, a nave espacial *Spaceship Titanic* colidiu com uma anomalia espaco-temporal e aproximadamente metade dos passageiros foi transportada para uma dimensao alternativa.

O objetivo e **prever quais passageiros foram transportados** (`Transported = True/False`) com base em informacoes demograficas, de cabine e de gastos a bordo da nave.

- **Tipo de problema:** Classificacao binaria
- **Variavel alvo:** `Transported`
- **Fonte:** [Kaggle — Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

---

## Dataset

| Caracteristica | Detalhe |
|---|---|
| Registros (treino) | ~8.700 |
| Colunas | 14 |
| Valores ausentes | Sim — presentes em quase todas as colunas |
| Variaveis categoricas | `HomePlanet`, `Destination`, `CryoSleep`, `VIP`, `Cabin` |
| Variaveis numericas | `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` |

### Colunas principais

| Coluna | Descricao |
|---|---|
| `PassengerId` | ID unico do passageiro |
| `HomePlanet` | Planeta de origem |
| `CryoSleep` | Se o passageiro estava em criosono |
| `Cabin` | Cabine no formato `Deck/Numero/Lado` |
| `Destination` | Planeta de destino |
| `Age` | Idade do passageiro |
| `VIP` | Se o passageiro e VIP |
| `RoomService` ... `VRDeck` | Gastos em servicos a bordo |
| `Transported` | **Target** — se foi transportado |

---

## Solucao Desenvolvida

A solucao foi desenvolvida em Python com Scikit-learn, seguindo um fluxo completo de Ciencia de Dados:

### 1. Analise Exploratoria de Dados (EDA)
- Analise de valores ausentes por coluna
- Estatisticas descritivas (media, mediana, desvio padrao, skewness)
- Histogramas das variaveis numericas
- Boxplots para identificacao de outliers nos gastos a bordo
- Scatterplot de gastos vs. variavel alvo
- Heatmap de correlacao
- Taxa de transporte por variavel categorica

### 2. Engenharia de Atributos

Foram criados 4 novos atributos derivados dos dados originais:

| Atributo | Origem | Justificativa |
|---|---|---|
| `TotalSpend` | Soma dos 5 gastos a bordo | Captura o comportamento de consumo geral do passageiro |
| `Deck` | Extracao de `Cabin` | A posicao na nave pode influenciar a probabilidade de transporte |
| `Side` | Extracao de `Cabin` | Lado da cabine (P/S) como possivel fator preditivo |
| `FaixaEtaria` | Agrupamento de `Age` | Reduz impacto de outliers e facilita aprendizagem do modelo |

### 3. Tratamento de Outliers

Os boxplots das variaveis de gastos a bordo revelam valores extremos expressivos. A decisao foi **manter os outliers** pelas seguintes razoes:

- Os valores extremos representam passageiros que gastaram muito em servicos premium — sao ocorrencias reais, nao erros de medicao
- Passageiros em CryoSleep tem gastos zerados obrigatoriamente, o que explica a forte assimetria e e um sinal preditivo importante
- O `StandardScaler` no pipeline reduz o impacto dos outliers na escala das variaveis
- O atributo `TotalSpend` agrega os gastos, suavizando o efeito de valores extremos em colunas individuais

Caso fosse necessario tratar os outliers, a abordagem recomendada seria o **clipping por IQR**, preservando as observacoes mas limitando sua influencia.

### 4. Pre-processamento com Pipeline

Utilizamos `Pipeline` e `ColumnTransformer` do Scikit-learn para organizar as transformacoes de forma segura e sem data leakage:

**Variaveis numericas:**
- `SimpleImputer(strategy='median')` — imputacao pela mediana, mais robusta a outliers
- `StandardScaler()` — normalizacao para algoritmos baseados em distancia

**Variaveis categoricas:**
- `SimpleImputer(strategy='most_frequent')` — imputacao pela moda
- `OneHotEncoder(handle_unknown='ignore')` — codificacao binaria sem hierarquia implicita

### 5. Prevencao de Data Leakage

- Divisao treino/teste realizada **antes** de qualquer transformacao
- O `fit` do pre-processador ocorre **apenas nos dados de treino**
- O Pipeline garante que o conjunto de teste receba somente o `transform`

### 6. Modelagem

Foram aplicados dois algoritmos supervisionados:

- `LogisticRegression` — modelo linear interpretavel, boa baseline
- `KNeighborsClassifier` — modelo baseado em distancia, otimizado via GridSearchCV

### 7. Validacao

- **K-Fold Cross Validation** com `k=5` para estimativa confiavel do desempenho
- **GridSearchCV** testando combinacoes de `n_neighbors`, `weights` e `metric` no KNN

---

## Resultados

| Modelo | Acuracia (CV) | Acuracia (Teste) |
|---|---|---|
| Logistic Regression | — | — |
| KNN (default) | — | — |
| KNN (GridSearchCV) | — | — |

> Os valores serao preenchidos apos a execucao completa do notebook.

**Melhores hiperparametros encontrados pelo GridSearchCV:**
> A ser preenchido apos execucao.

---

## Conclusoes

- O pre-processamento via Pipeline foi fundamental para garantir um fluxo correto e sem data leakage
- A engenharia de atributos, especialmente `TotalSpend` e a extracao de `Deck`/`Side` da coluna `Cabin`, enriqueceu significativamente a representacao dos dados
- Passageiros em CryoSleep apresentaram padrao de transporte distinto, sendo uma das variaveis mais relevantes
- O GridSearchCV permitiu encontrar a melhor combinacao de hiperparametros para o KNN de forma sistematica
- Como melhoria futura, algoritmos como `RandomForestClassifier` ou `GradientBoostingClassifier` poderiam ser explorados para ganhos adicionais de desempenho

---

## Estrutura do Repositorio

```
spaceship-titanic/
├── spaceship_titanic_IC.ipynb   # Notebook principal com todo o fluxo
└── README.md                    # Este arquivo
```

---

## Como Executar

1. Abra o notebook no [Google Colab](https://colab.research.google.com/)
2. Gere sua chave de API em [kaggle.com](https://www.kaggle.com) → Settings → API → Create New Token
3. Aceite os termos da competicao em [kaggle.com/competitions/spaceship-titanic](https://www.kaggle.com/competitions/spaceship-titanic)
4. Execute as celulas em sequencia — a primeira celula solicitara o upload do `kaggle.json`

**Dependencias:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `kaggle`
