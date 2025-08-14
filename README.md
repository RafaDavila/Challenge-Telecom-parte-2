# Challenge-Telecom-parte-2

## Parte 2 do desafio de data science da Alura

# 1)Remoção de Colunas Irrelevantes

Elimine colunas que não trazem valor para a análise ou para os modelos preditivos, como identificadores únicos (por exemplo, o ID do cliente). Essas colunas não ajudam na previsão da evasão e podem até prejudicar o desempenho dos modelos.

<img width="1176" height="107" alt="image" src="https://github.com/user-attachments/assets/f16ff9d7-8342-461a-9acc-374238e02337" />

customerID → provavelmente é um identificador único (nunca se repete), então não ajuda a prever se o cliente vai ter Churn ou não.

Churn → é a variável alvo (não podemos remover, é o que queremos prever).

customer, phone, internet, account → precisam ser analisadas, porque podem conter informações úteis.

Então, removeremos o customerID: df = df.drop(columns=['customerID'])

# 2)Encoding

Transforme as variáveis categóricas em formato numérico para torná-las compatíveis com algoritmos de machine learning. Utilize um método de codificação adequado, como o one-hot encoding.

Uma ou mais colunas do DataFrame não têm valores de texto simples, mas sim dicionários (dict) dentro das células — e o pd.get_dummies() não sabe lidar com isso diretamente.

<img width="1447" height="457" alt="image" src="https://github.com/user-attachments/assets/c707b37f-2e37-47ae-a8df-d53efa0c4154" />

# 3)Verificação da Proporção de Evasão

Calcule a proporção de clientes que evadiram em relação aos que permaneceram ativos. Avalie se há desequilíbrio entre as classes, o que pode impactar modelos preditivos e a análise de resultados.

<img width="958" height="617" alt="image" src="https://github.com/user-attachments/assets/ef31ed5c-4c4f-498a-bf75-20a0a755d6b3" />

<img width="747" height="315" alt="image" src="https://github.com/user-attachments/assets/c1b32fd0-0625-491a-a121-e2d4ee129aac" />

Isso significa que a coluna Churn tem três categorias diferentes:

"No" → clientes que não cancelaram (71,19%)

"Yes" → clientes que cancelaram (25,72%)

um valor vazio/NaN → 3,08% dos registros (provavelmente clientes sem informação de churn)

📌 Interpretação:

Existe um leve desequilíbrio —  quase 3 vezes mais clientes que ficaram do que clientes que saíram.

Esses 3% sem informação podem atrapalhar o modelo, precisa decidir:

remover essas linhas, ou

tratá-las (por exemplo, preenchendo com "No" ou "Yes", se fizer sentido).

## Balanceamento de Classes (opcional )

Caso queira aprofundar a análise, aplique técnicas de balanceamento como undersampling ou oversampling. Em situações de forte desbalanceamento, ferramentas como o SMOTE podem ser úteis para gerar exemplos sintéticos da classe minoritária.


com o SMOTE aplicado, esse deve ser o resultado:


<img width="756" height="621" alt="image" src="https://github.com/user-attachments/assets/6e56780b-cfa8-43ea-a0ab-d803ecd81f04" />

Cada classe agora representa 33,33% do total, ou seja, o número de exemplos em cada classe é igual.

Isso garante que, ao treinar um modelo, ele não vai favorecer nenhuma classe por ter mais exemplos do que outra.

## 4) Normalização ou Padronização (se necessário)

Avalie a necessidade de normalizar ou padronizar os dados, conforme os modelos que serão aplicados.
Modelos baseados em distância, como KNN, SVM, Regressão Logística e Redes Neurais, requerem esse pré-processamento.
Já modelos baseados em árvore, como Decision Tree, Random Forest e XGBoost, não são sensíveis à escala dos dados.

Padronização (StandardScaler) → transforma os dados para terem média 0 e desvio padrão 1.

<img width="845" height="202" alt="image" src="https://github.com/user-attachments/assets/129a4633-00cc-431a-9b50-15c0ef89d4ae" />

Normalização (MinMaxScaler) → transforma os dados para um intervalo específico, geralmente 0 a 1.

<img width="696" height="271" alt="image" src="https://github.com/user-attachments/assets/045de145-5aef-49e7-bf44-43d5d84f5cfe" />

## 5) Análise de Correlação

Visualize a matriz de correlação para identificar relações entre variáveis numéricas. Observe especialmente quais variáveis apresentam maior correlação com a evasão, pois elas podem ser fortes candidatas para o modelo preditivo.










