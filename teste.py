import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
import openai  

# Configurar a chave da API
openai.api_key = 'cm2djk52j00medq5p0c5dyzpo'  # Substitua pela sua chave da OpenAI

# Título da aplicação
st.markdown("<h2 style='color:lightblue;'>Preparados para uma aventura no Titanic?</h2>", unsafe_allow_html=True)
st.write("---")

# Adicionar imagem no topo
st.image("https://png.pngtree.com/png-vector/20240722/ourmid/pngtree-titanic-cruise-ship-sail-in-sea-iceberg-in-night-scene-in-png-image_13038983.png", use_column_width=True)

# Definir o estilo da página com imagem de fundo
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.cnnbrasil.com.br/wp-content/uploads/sites/12/2023/06/july9-hires.jpg");
        background-size: cover;
    }
    
    /* Estilo do balão de conversa */
    .chat-bubble {
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 80%;
        color: black;
    }

    .user-bubble {
        background-color: lightblue;
        margin-left: auto;
    }

    .bot-bubble {
        background-color: lightgray;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Adicionar música do arquivo local
st.audio("songtitanic.mp3") 

# HTML para o iframe do agente inteligente
iframe_code = '''
<iframe src="https://dash.superagentes.ai/agents/cm2djk52j00medq5p0c5dyzpo/iframe" 
        width="100%" height="600" frameborder="0" 
        allow="clipboard-write"></iframe>
'''

# Adiciona o iframe ao aplicativo
st.markdown(iframe_code, unsafe_allow_html=True)

# Carregar os dados
def carregar_dados():
    treino = pd.read_csv('train.csv')
    teste = pd.read_csv('test.csv')
    return treino, teste

# Preprocessar os dados
def preprocessar_dados(treino, teste):
    treino['Age'] = treino['Age'].fillna(treino['Age'].mean())
    teste['Age'] = teste['Age'].fillna(teste['Age'].mean())
    treino['Embarked'] = treino['Embarked'].fillna(treino['Embarked'].mode()[0])
    teste['Fare'] = teste['Fare'].fillna(teste['Fare'].mean())
    treino['MaleCheck'] = treino['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    teste['MaleCheck'] = teste['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    treino = treino.drop(['Sex', 'Ticket', 'Cabin'], axis=1)
    teste = teste.drop(['Sex', 'Ticket', 'Cabin'], axis=1)
    scaler = RobustScaler()
    treino[['Age', 'Fare']] = scaler.fit_transform(treino[['Age', 'Fare']])
    teste[['Age', 'Fare']] = scaler.transform(teste[['Age', 'Fare']])
    categorias = ['S', 'C', 'Q']
    encoder = OrdinalEncoder(categories=[categorias], dtype='int32')
    treino['Embarked'] = encoder.fit_transform(treino[['Embarked']])
    teste['Embarked'] = encoder.transform(teste[['Embarked']])
    vectorizer = CountVectorizer()
    X_train_nome = vectorizer.fit_transform(treino['Name'])
    X_teste_nome = vectorizer.transform(teste['Name'])
    X = pd.concat([treino.drop(['Name', 'PassengerId', 'Survived'], axis=1), pd.DataFrame(X_train_nome.toarray())], axis=1)
    y = treino['Survived']
    X_teste = pd.concat([teste.drop(['Name', 'PassengerId'], axis=1), pd.DataFrame(X_teste_nome.toarray())], axis=1)
    X.columns = X.columns.astype(str)
    X_teste.columns = X_teste.columns.astype(str)
    return X, y, X_teste

# Treinar modelo
def treinar_modelo(X, y):
    modelos = {
        'Regressão Logística': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'MLP Classifier (Redes Neurais)': MLPClassifier(max_iter=5000)
    }
    resultados = {}
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_val)
        resultados[nome] = accuracy_score(y_val, y_pred)
    return resultados, modelos

# Função para prever sobrevivência
def prever_sobrevivencia(modelos, X_teste, opcao):
    modelo = modelos[opcao]
    y_pred_teste = modelo.predict(X_teste)
    return y_pred_teste, modelo

# Main para rodar o resto do código
def main():
    treino, teste = carregar_dados()
    X, y, X_teste = preprocessar_dados(treino, teste)
    resultados, modelos = treinar_modelo(X, y)

    # Exibir a acurácia dos modelos
    st.subheader("Acurácia dos Modelos")
    st.write(resultados)

    # Opção para previsão
    opcao = st.selectbox("Escolha o modelo para previsão:", list(resultados.keys()))
    if st.button("Prever Sobrevivência"):
        y_pred_teste, modelo = prever_sobrevivencia(modelos, X_teste, opcao)
        st.write("Previsões de Sobrevivência (0 = Não Sobreviveu, 1 = Sobreviveu):")
        st.write(y_pred_teste)

if __name__ == "__main__":
    main()
