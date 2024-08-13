import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Simulação de dados com rótulos verdadeiros (true_label)
# Em um cenário real, creio que seria necessário identificar
# quais rótulos são de fato verdadeiros antes de treinar o modelo
data = {
    'ip_address': ['192.168.0.1', '192.168.0.2', '192.168.0.3', '192.168.0.4', '192.168.0.5', '192.168.5.7'],
    'timestamp': [1, 2, 3000, 4, 1000, 3],  # Exemplo de tempo em segundos
    'user_agent': [1, 1, 1, 2, 2, 1],  # Apenas como exemplo, normalmente codificado
    'query_field': [1, 2, 4, 1, 3, 1],  # Campos preenchidos
    'honeypot_field': [0, 1, 0, 0, 1, 0],  # Valor preenchido no campo honeypot
    'true_label': [0, 0, 1, 0, 1, 0]  # Rótulos verdadeiros: 0 para normal, 1 para anômalo
}

df = pd.DataFrame(data)

# Pré-processamento dos dados

# Normalização, para evitar que variáveis com escalas diferentes
# dominem a análise (ex: timestamp pode dominar a análise só porque tem números maiores)
scaler = StandardScaler()

df_scaled = scaler.fit_transform(df[['timestamp', 'user_agent', 'query_field', 'honeypot_field']])

# Testar diferentes valores de contamination (proporção esperada de anomalias - 10%, 20%, 30%)
contamination_values = [0.1, 0.2, 0.3]
for contamination in contamination_values:
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(df_scaled)

    # Predição das anomalias
    df['predicted_label'] = model.predict(df_scaled)

    # O modelo retorna 1 para amostras normais e -1 para anômalas,
    # mas os valores seão convertidos para 0 e 1, para funcionar em harmonia com os valores
    # de 'true_label"
    df['predicted_label'] = df['predicted_label'].apply(lambda x: 1 if x == -1 else 0)

    # Avaliação das métricas de desempenho
    print(f"Relatório de Classificação com contamination={contamination}:")
    print(classification_report(df['true_label'], df['predicted_label']))

    # Exibir IPs dos dados considerados anômalos
    anomalies = df[df['predicted_label'] == 1]
    print(
        f"IPs considerados anômalos com proporção esperada de anomalias = {contamination} ou {contamination * 100}%):")
    print(anomalies[['ip_address', 'predicted_label']])

    # Visualização dos dados
    plt.figure(figsize=(10, 6))

    # Gráfico de dispersão para 'timestamp' e 'query_field'
    plt.scatter(df['timestamp'], df['query_field'], c=df['predicted_label'], cmap='coolwarm', label='Previstos')

    # Destacar os pontos anômalos (em vermelho)
    plt.scatter(anomalies['timestamp'], anomalies['query_field'], color='red', edgecolor='k', s=100, label='Anomalias')

    # Exibir nos gráficos os IPs dos pontos anômalos
    for i in anomalies.index:
        plt.annotate(df.loc[i, 'ip_address'],
                     (df.loc[i, 'timestamp'], df.loc[i, 'query_field']),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    plt.xlabel('Timestamp')
    plt.ylabel('Query Field')
    plt.title(f'Distribuição dos Dados com Detecção de Anomalias '
              f'\n(com proporção esperada de anomalias = {contamination} ou {contamination * 100}%)')
    plt.legend()
    plt.show()

    print()
