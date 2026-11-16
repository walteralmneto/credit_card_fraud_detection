# Detecção de Fraude em Cartão de Crédito

Projeto de estudo que treina modelos para detectar fraude em transações reais de cartão de crédito,
do carregamento dos dados à explicação do modelo com SHAP.

Todo o trabalho está em um único notebook, com as saídas salvas (tabelas e gráficos):
[`credit_card_fraud_detection.ipynb`](credit_card_fraud_detection.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/walteralmneto/credit_card_fraud_detection/blob/main/credit_card_fraud_detection.ipynb)

---

## 1. O problema e por que o desbalanceamento muda a avaliação

A base tem **284.807 transações**, e só **492 são fraude (0,17%)**.

- `Time`: segundos desde a primeira transação
- `Amount`: valor da transação
- `V1` a `V28`: variáveis transformadas por PCA para proteger a privacidade
- `Class`: 1 = fraude, 0 = normal

Com tão poucas fraudes, a **acurácia engana**. No notebook, um `DummyClassifier` que responde sempre
"não é fraude" tem **99,83% de acurácia e não encontra nenhuma fraude** (recall de 0%).

Por isso, a avaliação olha para a **classe de fraude**:

| Métrica | Pergunta que responde |
|---|---|
| **Recall** | De todas as fraudes reais, quantas o modelo encontrou? |
| **Precisão** | Do que o modelo marcou como fraude, quanto era fraude mesmo? |
| **F1** | Equilíbrio entre precisão e recall |
| **PR-AUC** | Área sob a curva de precisão e recall (resume o modelo em todos os limiares) |

## 2. Preparação dos dados

1. **Leitura direta pelo link** com `pd.read_csv` (o CSV passa de 100 MB e não vai para o repositório):
   `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`
2. **Log do valor:** `log_amount = np.log1p(Amount)`, porque `Amount` é muito assimétrico (de 0 a 25.691).
3. **Treino e teste com `stratify=y`** (80/20): as duas partes ficam com ~0,17% de fraude
   (394 fraudes no treino, 98 no teste).
4. **Padronização com `StandardScaler`**, ajustado só no treino. Depois, no passo de ajuste,
   a padronização entra num `Pipeline` junto com o modelo.

## 3. Comparação entre os modelos

Resultado no conjunto de teste (56.962 transações, **98 fraudes**), limiar padrão de 0,5:

| Modelo | Acurácia | Recall (fraude) | Precisão (fraude) | F1 (fraude) | PR-AUC | Fraudes pegas | Alarmes falsos |
|---|---|---|---|---|---|---|---|
| Regressão Logística (baseline) | 99,92% | 65,3% | 83,1% | 0,731 | 0,739 | 64/98 | 13 |
| Random Forest (`class_weight='balanced'`) | 99,93% | 83,7% | 75,9% | 0,796 | 0,856 | 82/98 | 26 |
| XGBoost (`scale_pos_weight`) | 99,94% | 83,7% | 82,8% | **0,832** | **0,866** | 82/98 | 17 |

- A acurácia é praticamente igual nos três e **não ajuda a escolher**.
- Os modelos de árvore encontram **18 fraudes a mais** que a Regressão Logística.
- O XGBoost tem o melhor equilíbrio: mesmo recall do Random Forest com menos alarmes falsos.
- Na **curva ROC** os três parecem quase iguais (AUC de 0,96 a 0,99). Na **curva de precisão e recall**
  a Regressão Logística fica claramente abaixo das árvores. É a curva mais útil num problema desbalanceado.

## 4. Ajuste: limiar de decisão, GridSearchCV e SHAP

### Limiar de 0,5 para 0,3 (Pipeline com Regressão Logística)

| Limiar | Recall | Precisão | Fraudes pegas | Alarmes falsos |
|---|---|---|---|---|
| 0,5 | 65,3% | 83,1% | 64/98 | 13 |
| 0,3 | 69,4% | 75,6% | 68/98 | 22 |

Baixar o limiar pegou 4 fraudes a mais em troca de 9 alarmes falsos a mais.

### GridSearchCV escolhendo pelo recall
A busca (`C` e `class_weight`, validação cruzada estratificada de 5 dobras) escolheu `class_weight='balanced'` e `C=1`.
O recall no teste subiu para **91,8% (90/98)**, mas a precisão caiu para **6%: 1.403 alarmes falsos**.
Lição: otimizar **só** o recall leva o modelo a acusar fraude demais. É preciso equilibrar com a precisão.

### Limiar escolhido: 0,4 no Random Forest
O limiar foi escolhido pelo melhor F1 numa **validação separada do treino**, e não no teste.
Resultado final no teste:

| Random Forest no Pipeline | Recall | Precisão | F1 | Fraudes pegas | Alarmes falsos |
|---|---|---|---|---|---|
| limiar 0,5 | 83,7% | 79,6% | 0,816 | 82/98 | 21 |
| limiar 0,3 | 85,7% | 67,7% | 0,757 | 84/98 | 40 |
| **limiar 0,4 (escolhido)** | **85,7%** | **75,7%** | **0,804** | **84/98** | **27** |

O 0,4 pega as mesmas 84 fraudes que o 0,3 com 13 alarmes falsos a menos.

### O que o SHAP mostrou (XGBoost)
- **`V14` é a variável mais importante**: valores **baixos** de `V14` empurram forte para fraude.
- Depois vêm **`V4` alto**, e **`V12` e `V10` baixos**, também sinais de fraude.
- Na fraude explicada com o gráfico *waterfall*, `V14` muito baixo sozinho somou +5,1 ao score do modelo,
  seguido de `V10`, `V16`, `V4` e `V12`.
- Como `V1`–`V28` vêm de PCA, não dá para saber o significado real de cada uma, mas dá para ver
  que o modelo decide por padrões consistentes.

## 5. O que mudei em relação à aula

1. **Prova do paradoxo da acurácia** com um `DummyClassifier` (99,83% de acurácia, 0 fraudes encontradas).
2. **Undersampling × Oversampling** com o `Pipeline` do imbalanced-learn (reamostra só o treino):

   | Regressão Logística | Recall | Precisão | PR-AUC | Alarmes falsos |
   |---|---|---|---|---|
   | Sem reamostragem | 65,3% | 83,1% | 0,739 | 13 |
   | Undersampling | 91,8% | 3,7% | 0,607 | 2.324 |
   | Oversampling (SMOTE) | 91,8% | 5,8% | 0,722 | 1.459 |

   As duas chegam ao mesmo recall, mas o **SMOTE gera bem menos alarmes falsos**. Nesta base, oversampling ganhou.
3. **Random Forest no Pipeline com ajuste de limiar**, como a Expert sugeriu, varrendo limiares de 0,1 a 0,9.
4. **Limiar escolhido na validação, não no teste**, para a métrica final não ficar otimista demais.
5. Tabelas com **"fraudes pegas" e "alarmes falsos"** em números absolutos, que são mais fáceis de explicar do que porcentagens.

## Como executar

**Google Colab:** clique no botão "Open in Colab" acima e rode todas as células (*Ambiente de execução → Executar tudo*).

**Localmente:**

```bash
git clone https://github.com/walteralmneto/deteccao-fraude-cartao.git
cd deteccao-fraude-cartao
pip install -r requirements.txt
jupyter notebook credit_card_fraud_detection.ipynb
```

O notebook leva alguns minutos para rodar do início ao fim (o download da base tem cerca de 150 MB).

## Estrutura do repositório

```text
├── credit_card_fraud_detection.ipynb   # notebook com o pipeline completo e as saídas salvas
├── requirements.txt                    # bibliotecas necessárias
├── .gitignore                          # impede que o CSV e caches entrem no repositório
├── LICENSE                             # licença MIT
└── README.md                           # este arquivo
```

## Licença

MIT. Veja o arquivo [LICENSE](LICENSE).

