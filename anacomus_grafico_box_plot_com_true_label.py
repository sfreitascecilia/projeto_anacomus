import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Simulação de dados com rótulos verdadeiros (true_label)
# Em um cenário real, creio que seria necessário identificar
# quais rótulos são de fato verdadeiros antes de treinar o modelo
data = {
    'ip_address': ['192.168.0.1', '192.168.0.2', '192.168.0.3', '192.168.0.4', '192.168.0.5', '192.168.5.7'],
    'timestamp': [1, 2, 3000, 4, 1000, 3],  # Exemplo de tempo em segundos
    'user_agent': [1, 1, 1, 2, 2, 1],  # Apenas como exemplo, normalmente codificado
    'query_field': [1, 2, 14, 3, 4, 0],  # Campos preenchidos
    'honeypot_field': [0, 1, 0, 0, 1, 0],  # Valor preenchido no campo honeypot
    'true_label': [0, 0, 1, 0, 1, 0]  # Rótulos verdadeiros: 0 para normal, 1 para anômalo
}

df = pd.DataFrame(data)

# Pré-processamento dos dados
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
    # mas os valores serão convertidos para 0 e 1
    df['predicted_label'] = df['predicted_label'].apply(lambda x: 1 if x == -1 else 0)

    # Exibir IPs dos dados considerados anômalos
    anomalies = df[df['predicted_label'] == 1]
    print(
        f"IPs considerados anômalos com proporção esperada de anomalias = {contamination} ou {contamination * 100}%):")
    print(anomalies[['ip_address', 'true_label', 'predicted_label']])

    # Visualização dos dados com box plots (anômalos em vermelho)
    plt.figure(figsize=(14, 6))

    # Box plot para 'timestamp'
    plt.subplot(1, 2, 1)
    plt.boxplot(df['timestamp'], vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'),
                medianprops=dict(color='red'))

    # Retirar o comentário se quiser mostrar os valores que estão nos limites
    # plt.scatter(df['timestamp'], [1] * len(df), color='black', edgecolor='k', s=50, label='Pontos em Geral')
    plt.scatter(anomalies['timestamp'], [1] * len(anomalies), color='red', edgecolor='k', s=100, label='Anomalias')

    # Anotar IPs e true_label dos pontos anômalos
    for i in anomalies.index:
        plt.annotate(f"{df.loc[i, 'ip_address']} (True: {df.loc[i, 'true_label']})",
                     (df.loc[i, 'timestamp'], 1),
                     textcoords="offset points",
                     xytext=(5, 0),
                     ha='left',
                     fontsize=8,
                     color='black')

    plt.yticks([])
    plt.xlabel('Timestamp')
    plt.title(
        f'Box Plot para a Variável Timestamp \n(Proporção Esperada de Anomalias ='
        f' {contamination} ou {contamination * 100}%)')

    # Box plot para 'query_field'
    plt.subplot(1, 2, 2)
    plt.boxplot(df['query_field'], vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightgreen', color='green'),
                whiskerprops=dict(color='green'),
                capprops=dict(color='green'),
                medianprops=dict(color='red'))

    # Retirar o comentário se quiser mostrar os valores que estão nos limites
    # plt.scatter(df['query_field'], [1] * len(df), color='black', edgecolor='k', s=50, label='Pontos em Geral')
    plt.scatter(anomalies['query_field'], [1] * len(anomalies), color='red', edgecolor='k', s=100, label='Anomalias')

    # Anotar IPs e true_label dos pontos anômalos
    for i in anomalies.index:
        plt.annotate(f"{df.loc[i, 'ip_address']} (True: {df.loc[i, 'true_label']})",
                     (df.loc[i, 'query_field'], 1),
                     textcoords="offset points",
                     xytext=(5, 0),
                     ha='left',
                     fontsize=8,
                     color='black')

    plt.yticks([])
    plt.xlabel('Query Field')
    plt.title(
        f'Box Plot para a Variável Query Field \n(Proporção Esperada de Anomalias ='
        f' {contamination} ou {contamination * 100}%)')

    plt.legend()
    plt.tight_layout()
    plt.show()

    print()
