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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

# Função do chatbot
def chatbot_responder(pergunta):
    respostas = {
        "navio": "O navio se chama Titanic.",
        "afundou": "O Titanic afundou em 15 de abril de 1912.",
        "passageiros": "Havia cerca de 2.224 passageiros a bordo.",
        "sobreviveram": "Aproximadamente 706 passageiros sobreviveram.",
        "causa": "O Titanic colidiu com um iceberg.",
        "classes": "As classes de passageiros eram 1ª classe, 2ª classe e 3ª classe.",
        "iceberg": "Um iceberg é um grande bloco de gelo que se desprendeu de uma geleira.",
        "jack e rose": "Jack e Rose são personagens fictícios do filme Titanic.",
        "projeto": "É um projeto de machine learning que prever a sobrevivência dos passageiros do Titanic com base em dados disponíveis.",
        "modelos": "Os algoritmos utilizados incluem Regressão Logística, Random Forest e MLP Classifier (Redes Neurais).",
        "dados utilizados": "Os dados utilizados incluem informações sobre idade, gênero, classe de cabine, tarifa e outras características dos passageiros.",
        "como os dados são pré-processados?": "Os dados são limpos, preenchendo valores ausentes, transformando variáveis categóricas e normalizando valores numéricos.",
        "acurácia": "A acurácia é uma métrica que indica a proporção de previsões corretas em relação ao total de previsões feitas.",
        "validação": "A validação é realizada utilizando a divisão dos dados em conjuntos de treinamento e validação, avaliando a performance em dados não vistos.",
        "engenharia de atributos": "A engenharia de atributos é crucial para melhorar o desempenho do modelo, transformando dados brutos em características relevantes.",
        "como prever a sobrevivência?": "A sobrevivência é predita com base nas características dos passageiros, utilizando o modelo treinado para fazer previsões sobre novos dados.",
        "overfitting": "Overfitting é quando um modelo se ajusta demais aos dados de treinamento, resultando em baixa performance em dados novos.",
        "medidas": "As métricas utilizadas incluem acurácia, precisão, recall e F1-score.",
        "Oi": "Olá, como posso te ajudar? :)",
        "Precisão": "Proporção de verdadeiros positivos em relação ao total de positivos previstos. Reflita a capacidade do modelo de evitar falsos positivos."

    }

    # Verificar se a pergunta contém palavras-chave
    for palavra_chave, resposta in respostas.items():
        if palavra_chave in pergunta.lower():
            return resposta

    return "Desculpe, não tenho essa informação."

# Main para rodar o resto do código
def main():
    treino, teste = carregar_dados()
    X, y, X_teste = preprocessar_dados(treino, teste)
    resultados, modelos = treinar_modelo(X, y)

    # Seleção do modelo
    opcao = st.selectbox("Escolha o modelo para previsão:", list(modelos.keys()))

    # Botão para prever e exibir resultados
    if st.button('Prever Sobrevivência'):
        y_pred_teste, modelo_selecionado = prever_sobrevivencia(modelos, X_teste, opcao)
        teste['Survived'] = y_pred_teste
        
        # Calcular a acurácia do modelo selecionado no conjunto de treinamento
        y_train_pred = modelo_selecionado.predict(X)
        acuracia = accuracy_score(y, y_train_pred)

        # Calcular a quantidade total de passageiros, sobreviventes e não sobreviventes
        total_passageiros = teste.shape[0]
        total_sobreviventes = teste['Survived'].sum()
        total_nao_sobreviventes = total_passageiros - total_sobreviventes

        # Exibir informações destacadas com opacidade ajustada
        st.markdown(
            f"""
            <div style="padding: 10px; background-color: preto; border-radius: 5px;">
                <h4 style="margin: 0;">Resultados da previsão🔮:</h4>
                <p style="margin: 5px 0;">Acurácia do modelo '<strong>{opcao}</strong>': {acuracia:.4f}</p>
                <p style="margin: 5px 0;">Total de passageiros: <strong>{total_passageiros}</strong></p>
                <p style="margin: 5px 0;">Total de sobreviventes: <strong>{total_sobreviventes}</strong></p>
                <p style="margin: 5px 0;">Total de não sobreviventes: <strong>{total_nao_sobreviventes}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("Previsões de Sobrevivência:")
        st.dataframe(teste[['PassengerId', 'Survived']], width=600, height=400)

        # Criar gráfico para visualizar sobreviventes e não sobreviventes
        df_viz = pd.DataFrame({
            'Categoria': ['Sobreviventes', 'Não Sobreviventes'],
            'Quantidade': [total_sobreviventes, total_nao_sobreviventes]
        })
        
        # Gráfico com efeito de água e título visível e centralizado
        fig = px.bar(
            df_viz,
            x='Categoria',
            y='Quantidade',
            color='Categoria',
            title="<b>Sobreviventes vs. Não Sobreviventes</b>",
            color_discrete_sequence=["#1E90FF", "Black"],
            opacity=0.85
        )
        fig.update_layout(
            title_x=0.5,
            title_font=dict(size=24, color="white"),
            plot_bgcolor='rgba(173,216,230,0.2)',
            paper_bgcolor='rgba(255,255,255,0)',
            font=dict(size=18)
        )
        st.plotly_chart(fig)

    # Chatbot
    st.markdown("<h3 style='color:lightblue;'>Chatnic💬:</h3>", unsafe_allow_html=True)
    st.markdown("<h10 style='color:lightblue;'>ATENÇÃO⚠️. O chatbot implementado na aplicação possui um conjunto limitado de respostas, focando exclusivamente nas informações relacionadas ao projeto sobre o Titanic.</h10>", unsafe_allow_html=True)
    if "chat_active" not in st.session_state:
        st.session_state.chat_active = False
        st.session_state.conversas = []

    # Botão para iniciar o chatbot
    if st.button("Iniciar Chatbot"):
        st.session_state.chat_active = True

    # Botão para encerrar o chatbot
    if st.button("Encerrar Chatbot"):
        st.session_state.chat_active = False
        st.session_state.conversas = []  # Limpar as conversas

    if st.session_state.chat_active:
        pergunta = st.text_input("Você: ")
        if pergunta:
            resposta = chatbot_responder(pergunta)
            st.session_state.conversas.append((pergunta, resposta))

        # Exibir o histórico de conversas
        for pergunta, resposta in st.session_state.conversas:
            st.markdown(f"<div class='chat-bubble user-bubble'>{pergunta}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble bot-bubble'>{resposta}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
