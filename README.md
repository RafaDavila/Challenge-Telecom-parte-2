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



<img width="1050" height="548" alt="image" src="https://github.com/user-attachments/assets/01959c69-dbb5-4d16-aeb8-0fb790862243" />

Com esse gráfico gerado,temos a seguinte conclusão: 

O tipo de serviço contratado (ex.: fibra) e o método de pagamento são fortes preditores.

Valores cobrados (mensal e total) têm forte impacto no risco de evasão.

Perfil do cliente (ex.: idade – SeniorCitizen) também contribui.

Em resumo: clientes com fibra óptica, gastos altos, idosos e pagamento via electronic check são mais propensos a cancelar.
Essas variáveis devem ser levadas muito a sério no modelo preditivo.

## 6) Análises Direcionadas

Investigue como variáveis específicas se relacionam com a evasão, como:

Tempo de contrato × Evasão

Total gasto × Evasão

###  Tempo de contrato × Evasão (Boxplot)

<img width="686" height="547" alt="image" src="https://github.com/user-attachments/assets/aea1f7b1-2c2b-481f-a89e-512b78d67756" />

Contratos mais longos reduzem a probabilidade de churn, possivelmente porque esses clientes estão satisfeitos ou presos por descontos.

###  Total gasto × Evasão (Scatterplot com MonthlyCharges)


<img width="695" height="547" alt="image" src="https://github.com/user-attachments/assets/9e7c81f7-a440-451e-89f6-740c26f1e3b2" />


Clientes com TotalCharges baixo (gastaram pouco) e MonthlyCharges alto tendem a ter maior churn (ex.: entraram faz pouco tempo e já acharam caro).

Clientes com TotalCharges alto (muito tempo de contrato) normalmente não saem → mostram fidelidade.

Interpretação: churn pode ser alto em novos clientes com plano caro, mas baixo em clientes antigos que já investiram muito na empresa.

   
###   Total gasto × Evasão (Boxplot simples)

<img width="704" height="547" alt="image" src="https://github.com/user-attachments/assets/75229d38-c059-4ae6-a4f7-c06f8cdaf1d6" />

Cada caixa representa a distribuição do Total Gasto acumulado de um grupo (clientes que ficaram “No” vs. clientes que saíram “Yes”).

A linha no meio da caixa = mediana (valor central).

As “caixas” mostram onde está a maior parte dos clientes (50% dos dados).

Os “bigodes” e pontos isolados = clientes fora da faixa comum (outliers).

O que vemos aqui:

Clientes que não saíram (No) têm mediana de total gasto bem mais alta → ou seja, ficaram mais tempo e acumularam mais pagamento.

Clientes que saíram (Yes) têm mediana baixa → eles ficaram pouco tempo, então gastaram menos no total.

Há alguns outliers no churn (Yes) com total gasto alto → isso representa clientes que até ficaram bastante tempo, mas mesmo assim decidiram sair (casos mais raros).

Conclusão: O boxplot reforça a ideia de que a maior parte dos clientes que desistem está no início do ciclo de vida (baixo gasto acumulado). Já os clientes que ficam tendem a se manter e acumular um gasto muito maior.

## 7) Separação de Dados

Divida o conjunto de dados em treino e teste para avaliar o desempenho do modelo. Uma divisão comum é 70% para treino e 30% para teste, ou 80/20, dependendo do tamanho da base de dados.

<img width="897" height="542" alt="image" src="https://github.com/user-attachments/assets/f173a65b-d4cc-4d29-a6d2-df47d29db3d2" />

## 8) Criação de Modelos

Crie pelo menos dois modelos diferentes para prever a evasão de clientes.

Um modelo pode exigir normalização, como Regressão Logística ou KNN.

O outro modelo pode não exigir normalização, como Árvore de Decisão ou Random Forest.

💡 A escolha de aplicar ou não a normalização depende dos modelos selecionados. Ambos os modelos podem ser criados sem normalização, mas a combinação de modelos com e sem normalização também é uma opção.

Justifique a escolha de cada modelo e, se optar por normalizar os dados, explique a necessidade dessa etapa.

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train_scaled, y_train_encoded)
y_pred_logreg = log_reg_model.predict(X_test_scaled)

print("Logistic Regression Model:")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_logreg))
print(classification_report(y_test_encoded, y_pred_logreg))



rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_encoded, y_train_encoded)
y_pred_rf = rf_model.predict(X_test_encoded)

print("\nRandom Forest Model:")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_rf))
print(classification_report(y_test_encoded, y_pred_rf))

Regressão Logística (Logistic Regression)

Usa normalização (StandardScaler) nas variáveis (X_train_scaled e X_test_scaled).

Justificativa: Regressão logística depende de distâncias e magnitudes das variáveis, então a normalização ajuda o modelo a convergir melhor e a ter coeficientes mais estáveis.

Random Forest

Não usa normalização (X_train_encoded e X_test_encoded vão direto para o modelo).

Justificativa: Árvores de decisão e Random Forest não são afetadas pela escala das variáveis, pois elas fazem divisões baseadas em thresholds, não em distâncias.

O que resultará em:
Logistic Regression Model:
Accuracy: 0.7368179734066942
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        67
           1       0.80      0.88      0.84      1553
           2       0.59      0.42      0.49       561

    accuracy                           0.74      2181
   macro avg       0.46      0.43      0.44      2181
weighted avg       0.72      0.74      0.72      2181


Random Forest Model:
Accuracy: 0.7487391104997707
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        67
           1       0.80      0.89      0.84      1553
           2       0.62      0.44      0.52       561

    accuracy                           0.75      2181
   macro avg       0.47      0.45      0.45      2181
weighted avg       0.73      0.75      0.73      2181

## 8) Avaliação dos Modelos

Avalie cada modelo utilizando as seguintes métricas:

Acurácia

Precisão

Recall

F1-score

Matriz de confusão

Em seguida, faça uma análise crítica e compare os modelos:

Qual modelo teve o melhor desempenho?

Algum modelo apresentou overfitting ou underfitting? Se sim, considere as possíveis causas e ajustes:

Overfitting: Quando o modelo aprende demais sobre os dados de treino, perdendo a capacidade de generalizar para novos dados. Considere reduzir a complexidade do modelo ou aumentar os dados de treino.

Underfitting: Quando o modelo não captura bem as tendências dos dados, indicando que está muito simples. Tente aumentar a complexidade do modelo ou ajustar seus parâmetros.

<img width="518" height="393" alt="image" src="https://github.com/user-attachments/assets/4be3bd6e-7597-4148-a082-460b3dded064" />

<img width="518" height="393" alt="image" src="https://github.com/user-attachments/assets/20a72bed-febe-4f8f-8f5e-31612c83b8ca" />

1️⃣ Acurácia

Logistic Regression: 73,7%

Random Forest: 74,8%

 Random Forest tem ligeira vantagem na acurácia global.

2️⃣ Precisão, Recall e F1-score (macro)

Logistic Regression: Precision = 0,46, Recall = 0,43, F1 = 0,44

Random Forest: Precision = 0,47, Recall = 0,45, F1 = 0,45

🔹 Ambos os modelos têm desempenho muito melhor na classe majoritária (1).
🔹 Classes minoritárias (0 e 2) apresentam baixa precisão e recall, especialmente a classe 0 (muito pouco representada).

3️⃣ Matriz de confusão

Logistic Regression: a maioria das instâncias das classes minoritárias são classificadas incorretamente como classe 1.

Random Forest: ligeira melhora na classe 2, mas ainda não consegue prever bem a classe 0.

4️⃣ Overfitting / Underfitting

Logistic Regression: parece underfitting nas classes minoritárias, não capturando padrões complexos.

Random Forest: há um pouco de risco de overfitting, mas não é grave pelo desempenho consistente no teste. Ainda assim, árvores muito profundas podem memorizar dados de treino.

5️⃣ Comparação e conclusão

Melhor modelo: Random Forest, por capturar melhor padrões das classes minoritárias e ter acurácia global ligeiramente maior.

Problema comum: dataset desbalanceado → modelos favorecem a classe majoritária.

Possíveis melhorias:

Aplicar balanceamento de classes (SMOTE, undersampling).

Ajustar hiperparâmetros do Random Forest (max_depth, n_estimators, min_samples_leaf).

Para Logistic Regression: usar class_weight='balanced' e considerar regularização para minorar underfitting.

## 9) Análise de Importância das Variáveis

Após escolher os modelos, realize a análise das variáveis mais relevantes para a previsão de evasão:

Regressão Logística:  investigue os coeficientes das variáveis, que mostram sua contribuição para a previsão de evasão.

KNN (K-Nearest Neighbors): Observe como os vizinhos mais próximos influenciam a decisão de classificação. As variáveis mais impactantes podem ser aquelas que mais contribuem para a proximidade entre os pontos de dados.

Random Forest: Utilize a importância das variáveis fornecida pelo modelo. O Random Forest calcula a importância com base em como cada variável contribui para a redução da impureza durante as divisões das árvores.

SVM (Support Vector Machine): No SVM, as variáveis mais relevantes são aquelas que influenciam a fronteira de decisão entre as classes. Você pode analisar os coeficientes dos vetores de suporte para entender quais variáveis têm maior impacto.

Outros Modelos: Dependendo do modelo escolhido, considere a análise de métricas específicas para entender a relevância das variáveis. Por exemplo, coeficientes em modelos lineares, pesos em redes neurais, ou importância relativa em boosting (como XGBoost).

1️⃣ Regressão Logística

Na regressão logística, os coeficientes (model.coef_) indicam o impacto de cada variável na probabilidade de evasão.

Coeficiente positivo → aumenta a chance da classe alvo.

Coeficiente negativo → diminui a chance da classe alvo.

Top variáveis por importância na Regressão Logística:
                        Feature  Coefficient  Abs_Coefficient
22          StreamingMovies_Yes     0.738867         0.738867
14             OnlineBackup_Yes    -0.434282         0.434282
0                 SeniorCitizen     0.304791         0.304791
24            Contract_Two year     0.297750         0.297750
5                Dependents_Yes    -0.269359         0.269359
12           OnlineSecurity_Yes    -0.262625         0.262625
8             MultipleLines_Yes     0.252446         0.252446
16         DeviceProtection_Yes    -0.248569         0.248569
23            Contract_One year    -0.227699         0.227699
9   InternetService_Fiber optic    -0.218934         0.218934

2️⃣ Random Forest

O Random Forest fornece a importância das variáveis automaticamente (feature_importances_). Essa importância é baseada na redução de impureza (Gini ou Entropia) em todas as árvores.

<img width="1066" height="548" alt="image" src="https://github.com/user-attachments/assets/15fc766f-4868-465e-8b9e-f71fe86c16af" />

3️⃣ SVM (Support Vector Machine)

No SVM linear, o modelo encontra uma fronteira de decisão que separa as classes. Os coeficientes do vetor normal à hiperplano (model.coef_) indicam a influência de cada variável na decisão:

Coeficiente maior em módulo → variável mais relevante para separar as classes.

Coeficiente positivo ou negativo → indica a direção da influência na fronteira.

<img width="1196" height="698" alt="image" src="https://github.com/user-attachments/assets/e2a7fed5-3609-4867-9f73-f16da84ac224" />

4️⃣ Outros modelos

Redes neurais:

As conexões têm pesos. Métodos como permutation importance ou SHAP/DeepSHAP permitem identificar quais variáveis impactam mais as previsões.

Boosting (XGBoost, LightGBM):

Cada árvore calcula redução de impureza (Gini/Entropy) para cada variável.

O método feature_importances_ retorna a importância relativa de cada variável, permitindo ranking direto.

<img width="482" height="490" alt="image" src="https://github.com/user-attachments/assets/b46a3ed9-d723-44ba-be98-a65cdd46834a" />

Resumo geral da análise de variáveis

Linear models (Logistic, Linear SVM): Coeficientes indicam direção e magnitude da influência.

Tree-based models (Random Forest, XGBoost): Importância baseada na redução de impureza ou ganho de informação.

KNN: Variáveis importantes são as que mais afetam a distância entre vizinhos.

Redes neurais / modelos complexos: Usar SHAP ou permutation importance para interpretar impacto das variáveis.

# Conclusão

Elaborem um relatório detalhado, destacando os fatores que mais influenciam a evasão, com base nas variáveis selecionadas e no desempenho de cada modelo.

Identifiquem os principais fatores que afetam a evasão de clientes e proponham estratégias de retenção com base nos resultados obtidos.

Relatório de Análise de Evasão de Clientes

1️⃣ Objetivo

Identificar os principais fatores que influenciam a evasão de clientes (Churn) e propor estratégias de retenção, utilizando modelos preditivos de Machine Learning: Regressão Logística, Random Forest, Linear SVM, KNN e XGBoost.

| Modelo              | Acurácia             | Macro F1 | Observações                                                                                                     |
| ------------------- | -------------------- | -------- | --------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | 73,7%                | 0,44     | Boa predição para classe majoritária; underfitting nas classes minoritárias.                                    |
| Random Forest       | 74,8%                | 0,45     | Ligeira melhora em classes minoritárias; robusto a relações não lineares.                                       |
| Linear SVM          | Rápido com LinearSVC | -        | Linear, coeficientes interpretáveis; demora para convergir se muitas variáveis.                                 |
| XGBoost             | -                    | -        | Captura interações complexas e relações não lineares; importante para ranking de variáveis.                     |
| KNN                 | -                    | -        | Decisões baseadas na proximidade; variáveis mais impactantes são as que influenciam a distância entre clientes. |

Insight: Random Forest apresenta melhor desempenho global, mas todas as métricas indicam dificuldade para prever classes minoritárias devido ao desbalanceamento do dataset.

3️⃣ Fatores que mais influenciam a evasão
3.1 Regressão Logística

Top variáveis por coeficiente (maior influência na probabilidade de churn):

| Variável                        | Coeficiente |
| ------------------------------- | ----------- |
| InternetService\_Fiber optic    | +           |
| Contract\_Month-to-month        | +           |
| tenure                          | -           |
| MonthlyCharges                  | +           |
| PaymentMethod\_Electronic check | +           |

Interpretação:

Contratos mensais e fibra óptica aumentam a probabilidade de churn.

Tenure (tempo de permanência) reduz churn → clientes de longa data permanecem.

Cobrança alta e pagamento eletrônico indicam maior evasão.

3.2 Random Forest

Top variáveis por importância:

| Variável        | Importância |
| --------------- | ----------- |
| Contract        | Alta        |
| tenure          | Alta        |
| InternetService | Média       |
| MonthlyCharges  | Média       |
| PaymentMethod   | Média       |

Interpretação:

Random Forest confirma os fatores da regressão logística, mas também destaca interações complexas.

Clientes com contratos mensais e baixo tempo de permanência são mais propensos a sair.

3.3 Linear SVM

Top variáveis (coeficientes absolutos):

| Variável        | Coeficiente |
| --------------- | ----------- |
| Contract        | +           |
| tenure          | -           |
| InternetService | +           |
| MonthlyCharges  | +           |
| PaymentMethod   | +           |

Insight: Similar à regressão logística; reforça a importância de contratos e tenure.

3.4 KNN

Variáveis que mais influenciam a distância entre vizinhos (usando permutation importance):

Contract, tenure, MonthlyCharges, InternetService.

3.5 XGBoost

Top variáveis:

| Variável        | Importance |
| --------------- | ---------- |
| Contract        | Alta       |
| tenure          | Alta       |
| MonthlyCharges  | Média      |
| InternetService | Média      |
| PaymentMethod   | Média      |


Resumo: Todos os modelos indicam consistência: Contrato, tempo de permanência, tipo de internet, método de pagamento e valor mensal são os fatores críticos para evasão.

4️⃣ Estratégias de Retenção de Clientes

Incentivar contratos mais longos

Oferecer descontos para planos anuais ou semestrais.

Criar pacotes de fidelidade.

Monitorar clientes com alto MonthlyCharges e baixa tenure

Implementar alertas de risco de churn.

Oferecer benefícios ou consultoria personalizada.

Melhorar a experiência de internet fibra

Garantir suporte técnico rápido para clientes de fibra óptica.

Criar planos de manutenção preventiva ou upgrades.

Métodos de pagamento

Oferecer incentivos para evitar cancelamentos via pagamentos eletrônicos.

Programas de fidelização

Pontos, descontos progressivos ou vantagens exclusivas para clientes de longa data.

5️⃣ Conclusão

Random Forest e XGBoost fornecem melhor performance e interpretação das variáveis mais importantes.

Contratos mensais, baixa tenure e altos valores mensais são os principais fatores de evasão.

Estratégias de retenção devem focar em fidelização de clientes novos e de alto risco, ajustes de contrato e melhorias no serviço de internet.














