# 🎬 Relatório Final – Desafio Cientista de Dados Indicium

---

# Abaixo estará a explicação de como instalar e executar o projeto

---

## Conteúdo do Repositório

* `notebooks/` – Jupyter Notebook com análises exploratórias, pré-processamento e treinamento de modelos.
* `modelo_rf_eda.pkl` – Modelo Random Forest treinado e otimizado usando GridSearch, com variáveis selecionadas via EDA.
* `requirements.txt` – Arquivo contendo todos os pacotes e suas versões necessários para executar o projeto.
* **Dataframes utilizados:**

  * `Box_Office` – Conjunto de dados externos, que trazem informações financeiras e de bilheteria dos filmes.
  * `desafio_indicium_imdb` – Conjunto principal de dados com informações gerais e notas do IMDb.
  * `filmes_com_franquia` – Conjunto de dados externos, com informações sobre se o filme faz parte de uma franquia ou é lançamento original (extraído do [TMDb](https://www.themoviedb.org)).

---

## Requisitos

* Python >= 3.11 (testado em Python 3.13)
* Instalar dependências (recomendo usar ambiente virtual):

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## Como Executar

1. Clone o repositório:

```bash
git clone https://github.com/WiiniSilva/LH_CD_WINICIUS_ANTONIO_SOUZA_SILVA.git
cd LH_CD_WINICIUS_ANTONIO_SOUZA_SILVA
```

2. Ative seu ambiente virtual e instale as dependências (veja seção de requisitos).

3. Abra o Jupyter Notebook para visualizar as análises e resultados:

---

## Como Usar o Modelo Random Forest Salvo

Você pode carregar e utilizar o modelo Random Forest salvo (`modelo_rf_eda.pkl`) para fazer predições em novos dados:

```python
import joblib

# Carregar o modelo salvo
modelo_rf_eda = joblib.load("modelo_rf_eda.pkl")

# Fazer predições em novos dados (novo dataset deve ter as mesmas colunas pré-processadas)
y_pred = modelo_rf_eda.predict(novo_dado)

print(f"Predições: {y_pred}")
```

> **Observação:** `novo_dado` precisa estar pré-processado da mesma forma que os dados usados para treinar o modelo (mesmo escalonamento, codificação de variáveis categóricas, transformação log, etc.).

---

---

## ❓ Perguntas e Respostas

---

### 2.a. Qual filme você recomendaria para uma pessoa que você não conhece?

Como não conheço a pessoa nem seus gostos, eu indicar filmes que tenham **notas altas no IMDb** e um **ROI mais equilibrado**, porque isso aumenta as chances de o filme ter qualidade e agradar. Diretores como **Christopher Nolan** costumam manter um padrão de avaliações bem alto, enquanto **Steven Spielberg** consegue unir boa recepção do público com retorno financeiro forte. Já nos gêneros, **Mystery** e **Western** apareceram com as maiores notas médias, então acabam sendo apostas seguras para recomendar de forma geral. No fim, recomendaría obras desses diretores ou desses gêneros, já que unem qualidade, boa aceitação do público e potencial de agradar mesmo sem conhecer seu gosto.

---

### 2.b. Quais são os principais fatores que estão relacionados com alta expectativa de faturamento de um filme?

O estudo mostrou que alguns fatores ajudam a prever se um filme vai ter um faturamento maior:

* **Orçamento**: filmes com mais dinheiro geralmente arrecadam mais, mas produções de baixo custo também podem surpreender.
* **Diretor**: certos diretores, como Spielberg e Clint Eastwood, têm ROI consistente e vários sucessos garantidos.
* **Franquia**: filmes que fazem parte de franquias costumam ter resumos mais detalhados, universos complexos e personagens recorrentes, o que atrai mais público.
* **Gênero**: mesmo que a diferença entre gêneros não seja estatisticamente significativa, filmes de **Família** tiveram retornos médios mais altos e estáveis.

Ou seja, além do dinheiro investido, a escolha do diretor e se o filme é parte de uma franquia ou original também influenciam bastante no desempenho financeiro, embora sempre exista algum risco.

**Se eu tivesse que fazer uma recomendação para um investidor, seria:** Priorize **diretores experientes**, prefira **franquias** ou continuações, mantenha o **orçamento equilibrado** e foque em **filmes de Família** para aumentar a chance de retorno financeiro estável.

---

### 2.c. Quais insights podem ser tirados com a coluna Overview? É possível inferir o gênero do filme a partir dessa coluna?

Insights da Coluna Overview

Os resumos dos filmes (Overview) trazem informações valiosas sobre o gênero e o conteúdo da obra:

Cada gênero apresenta vocabulário característico:

- **Ação/Aventura/Animação**: combates, missões, exploração.
- **Drama/Crime/Terror**: relações humanas, conflitos, tensão.
- **Comédia**: situações leves, humor e romance.
- **Biografia**: narrativas de vida e conquistas pessoais.

Inicialmente usamos o modelo TF-IDF + LinearSVC que não superou a baseline de acurácia, mas ao realizar o teste de permutação (p=0,005), ele nos mostrou que há sinais estatisticamente significativos, ou seja, os resumos contêm informação relevante sobre o gênero.

Palavras-chave como *fight*, *mission* e *army* aumentam a probabilidade de um filme ser de ação, enquanto *woman*, *mysterious* e *investigate* reduzem essa chance.

Diferenças sutis entre franquias e originais: resumos de franquias tendem a reforçar enredos épicos e universos complexos, enquanto originais exploram experiências mais pessoais e sociais.

Apesar de as Overviews serem muitas vezes genéricas, combiná-las com técnicas avançadas de NLP (como BERT ou embeddings pré-treinados) e dados estruturados (duração, número de votos, ROI) pode aumentar a precisão na classificação de gênero e na análise de sucesso do filme.

---

O estudo indica que, para maximizar qualidade percebida e retorno financeiro, é estratégico considerar **diretor, gênero, orçamento e franquia**. Os resumos fornecem pistas valiosas sobre o gênero, mas sua análise isolada tem limitações. Modelos de machine learning integrando texto e dados estruturados, como **Random Forest com variáveis selecionadas pela EDA**, se mostraram eficazes para predizer notas no IMDb, oferecendo uma base sólida para decisões de produção e recomendação de filmes.

---

### 3.1. Explique como você faria a previsão da nota do imdb a partir dos dados. Quais variáveis e/ou suas transformações você utilizou e por quê?

Para prever a nota do IMDb dos filmes, estamos lidando com um **problema de regressão**, já que o alvo é contínuo, ou seja, queremos estimar o valor numérico da nota de cada filme. O objetivo do modelo é prever a nota usando apenas informações que estariam disponíveis **antes do lançamento**, evitando dados do futuro que poderiam distorcer o aprendizado (problema de *data leakage*).

No pré-processamento, removemos variáveis que poderiam atrapalhar a predição. Por exemplo, **Gross**, **ROI** e **Production Budget** foram excluídos porque são informações financeiras só conhecidas depois que o filme é lançado. Incluir esses dados faria o modelo aprender padrões impossíveis de prever antes do lançamento, resultando em predições irreais. Outras variáveis removidas foram **Overview** e **overview\_clean** (textos completos do resumo), **Series\_Title** e **Released\_Year**, que não ajudavam diretamente na previsão.

Criamos a variável **overview\_len**, que mede o tamanho do resumo do filme. Essa transformação gera um número que indica indiretamente quão detalhado é o roteiro, fornecendo informação útil sem precisar trabalhar com texto bruto, o que exigiria processamento mais complexo de NLP. Para lidar com valores extremos, aplicamos uma transformação logarítmica (*log(1+x)*) e depois escalonamento.

As variáveis numéricas (**Runtime, Meta\_score, No\_of\_Votes e overview\_len**) foram escalonadas, e No\_of\_Votes e overview\_len também receberam log-transformação para reduzir o impacto de valores muito altos. As variáveis categóricas (**Certificate** e **Main\_Genre**) foram transformadas em **One-Hot Encoding**, enquanto **Director** recebeu **Target Encoding**, usando a média das notas do IMDb por diretor.

A variável **is\_franchise**, uma variável externa obtida do TMDb, indica se o filme faz parte de uma franquia ou é original. Transformamos **True/False em 1/0**, tornando-a numérica e pronta para o modelo. Ela pode ajudar a capturar efeitos indiretos, como maior popularidade ou expectativa do público, mas não é decisiva sozinha, funciona como uma variável auxiliar.

No começo, incluímos os atores principais (**Star1-4**) usando Target Encoding, acreditando que atores renomados poderiam influenciar a nota. Porém, ao treinar modelos como **Random Forest, SVM e Decision Tree**, percebemos que isso causava **overfitting**: o modelo aprendia padrões específicos dos atores do conjunto de treino, mas não funcionava bem com novos filmes. Por isso, removemos essas variáveis, focando em informações que realmente ajudam a prever de forma consistente.

Para selecionar as variáveis mais importantes, combinamos três métodos: análise exploratória (EDA), Feature Importance via árvore de decisão e Information Value (IV). As variáveis que mais se destacaram e foram consistentes entre os métodos foram **Runtime, No\_of\_Votes, Meta\_score, Main\_Genre e Director**. Variáveis adicionais como **Certificate, overview\_len e is\_franchise** foram mantidas por poderem melhorar ainda mais a performance do modelo.

| Variável          | EDA | Feature Importances | IV |
| ----------------- | --- | ------------------- | -- |
| **Runtime**       | X   | X                   | X  |
| **No\_of\_Votes** | X   | X                   | X  |
| **Meta\_score**   | X   | X                   | X  |
| **Main\_Genre**   | X   | X                   | X  |
| **Director**      | X   | X                   | X  |
| **is\_franchise** | X   | X                   |    |
| **Certificate**   | X   |                     |    |
| **overview\_len** |     | X                   | X  |

---

### 3.2. Qual tipo de problema estamos resolvendo (regressão, classificação)?

Estamos lidando agora com um problema de **regressão** porque nosso alvo (`y`) é uma variável contínua, a nota do IMDb, e não categorias discretas. Diferente de um problema de classificação, onde preveríamos rótulos como “bom/ruim” ou “alta/baixa nota”, aqui queremos **estimar valores numéricos exatos**, justificando o uso de métricas e modelos específicos para regressão.

---

### 3.3. Qual modelo melhor se aproxima dos dados e quais seus prós e contras?

Testamos quatro modelos de regressão: Random Forest Regressor, SVM (SVR), Ridge Regression e Decision Tree Regressor. O Random Forest se destacou, explicando cerca de 67–72% da variação das notas (R² ~0.729) e mantendo consistência entre os folds da validação cruzada. Este modelo é robusto a outliers, captura relações não lineares e interações complexas, além de fornecer métricas de importância das variáveis. Limitações incluem menor interpretabilidade, necessidade de ajuste de hiperparâmetros, maior tempo de treino e dificuldade em extrapolar valores fora do padrão observado.

| Aspecto            | Prós                                                      | Contras                                                     |
| ------------------ | --------------------------------------------------------- | ----------------------------------------------------------- |
| Acurácia           | Captura relações complexas e não lineares entre variáveis |                                                             |
| Robustez           | Resistente a outliers e ruídos nos dados                  |                                                             |
| Interpretabilidade | Permite avaliar importância das variáveis                 | Difícil de interpretar individualmente (caixa-preta)        |
| Overfitting        | Menor risco que uma árvore única                          |                                                             |
| Desempenho         |                                                           | Treinamento e previsão mais lentos                          |
| Hiperparâmetros    |                                                           | Necessita ajuste de parâmetros para performance ótima       |
| Extrapolação       |                                                           | Não lida bem com valores fora do padrão observado no treino |

---

### 3.4. Qual medida de performance do modelo foi escolhida e por quê?

Para medir performance, utilizamos várias métricas complementares:

- **R²**: mede a proporção da variação da nota do IMDb explicada pelo modelo, mostrando o poder explicativo global.
- **RMSE (Root Mean Squared Error)**: penaliza mais os erros grandes, dando ênfase a previsões muito distantes do valor real.
- **MAE (Mean Absolute Error)**: indica o erro médio absoluto, oferecendo uma medida mais intuitiva do desvio médio das previsões sem exagerar nos outliers.
- **MAPE (Mean Absolute Percentage Error)**: fornece o erro em termos percentuais, útil para comparar a precisão relativa das previsões em diferentes escalas.

A combinação dessas métricas garante que o modelo não apenas explique bem a variação das notas, mas também apresente previsões precisas tanto em termos absolutos quanto relativos. A Random Forest, combinada com a seleção de variáveis via EDA, oferece alto poder explicativo, baixo risco de overfitting e segurança contra data leakage, tornando-se a escolha ideal para prever a nota do IMDb.

| Método de seleção          | R²    | RMSE  | MAE   | MAPE  | Observação/Comentário                                                                            |
| -------------------------- | ----- | ----- | ----- | ----- | ------------------------------------------------------------------------------------------------ |
| **EDA**                    | 0.729 | 0.134 | 0.097 | 1.21% | Resultados obtidos com as variáveis selecionadas via EDA                                         |
| **Feature Importances**    | 0.728 | 0.134 | 0.099 | 1.23% | Desempenho próximo ao da EDA, variáveis importantes do modelo foram usadas                       |
| **IV (Information Value)** | 0.724 | 0.135 | 0.099 | 1.23% | Bom desempenho, quase igual ao da EDA, variáveis escolhidas via IV se mostraram muito relevantes |
| **Melhor resultado**       | 0.729 | 0.134 | 0.097 | 1.21% | Até agora, melhor resultado obtido com EDA  

---

### 4. Supondo um filme com as seguintes características:

{'Series_Title': 'The Shawshank Redemption',
 'Released_Year': '1994',
 'Certificate': 'A',
 'Runtime': '142 min',
 'Genre': 'Drama',
 'Overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
 'Meta_score': 80.0,
 'Director': 'Frank Darabont',
 'Star1': 'Tim Robbins',
 'Star2': 'Morgan Freeman',
 'Star3': 'Bob Gunton',
 'Star4': 'William Sadler',
 'No_of_Votes': 2343110,
 'Gross': '28,341,469'}


    Qual seria a nota do IMDB?


Para estimar a nota do IMDb desse filme, começamos aplicando **todas as transformações que definimos no nosso modelo**. Isso inclui:

* Transformar a duração (**Runtime**) em número inteiro,
* Criar a variável **overview\_len** a partir do tamanho do resumo (**Overview**) e aplicar log-transformação + escalonamento,
* Escalonar as variáveis numéricas (**Meta\_score, No\_of\_Votes**),
* Transformar variáveis categóricas (**Certificate, Genre** → Main\_Genre, Director) usando One-Hot Encoding ou Target Encoding,
* Adicionar e transformar a variável booleana **is\_franchise** em 0/1,
Em resumoi, foi necessário garantir que todas as variáveis estão no mesmo formato que o modelo espera.

Depois de preparar o filme dessa forma e passá-lo pelo nosso **modelo Random Forest**, ele retornou uma previsão de **nota IMDb igual a 8.7**.

Isso significa que, segundo nosso modelo, **“The Shawshank Redemption” teria uma avaliação muito alta**, alinhada com sua reputação clássica. A previsão leva em conta o diretor, a nota do Meta\_score, o número de votos, o tamanho do resumo e outros fatores que o modelo aprendeu a associar a boas avaliações, sem precisar olhar para dados financeiros ou qualquer informação pós-lançamento.

---