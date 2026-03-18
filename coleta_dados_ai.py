import streamlit as st
import yfinance as yf
import pandas as pd
from google import genai
from dotenv import load_dotenv
import os


# ==========================================
# CONFIGURAÇÃO GERAL
# ==========================================
load_dotenv()
MINHA_CHAVE_API = os.getenv("PRIVATE_KEY_GOOGLE_AI")
client = genai.Client(api_key=MINHA_CHAVE_API)

# Configura o visual da página no navegador
st.set_page_config(page_title="Assistente IA Financeiro", page_icon="📈", layout="centered")

# Título principal do site
st.title("🤖 Assistente de Análise de Criptomoedas")
st.write("Digite o código de uma moeda para ver os dados e gerar um relatório feito por Inteligência Artificial.")

# ==========================================
# INTERFACE DO USUÁRIO
# ==========================================
# Cria uma caixa de texto onde o usuário pode digitar a moeda
ativo_escolhido = st.text_input("Qual moeda deseja analisar?", value="BTC-USD")

# Cria um botão grande na tela
if st.button("Gerar Análise Completa"):
    
    # Mostra um "carregando" na tela enquanto busca os dados
    with st.spinner(f"Buscando dados de {ativo_escolhido} no mercado..."):
        try:
            # 1. COLETA DE DADOS
            dados = yf.download(ativo_escolhido, period="30d", progress=False)

            if isinstance(dados.columns, pd.MultiIndex):
                dados.columns = ["_".join(map(str, col)).strip() for col in dados.columns.values]

            col_close = next(c for c in dados.columns if "Close" in c)
            col_high  = next(c for c in dados.columns if "High"  in c)
            col_low   = next(c for c in dados.columns if "Low"   in c)

            dados = dados.dropna(subset=[col_close, col_high, col_low])
            dados[col_close] = pd.to_numeric(dados[col_close], errors="coerce")
            dados[col_high]  = pd.to_numeric(dados[col_high],  errors="coerce")
            dados[col_low]   = pd.to_numeric(dados[col_low],   errors="coerce")
            dados = dados.dropna(subset=[col_close, col_high, col_low])

            fechamento_atual = dados[col_close].iloc[-1]
            maximo_30d = dados[col_high].max()
            minimo_30d = dados[col_low].min()

            # 2. MOSTRANDO RESULTADOS NA TELA
            st.success("Dados coletados com sucesso!")
            
            # Cria 3 caixinhas bonitas com os valores na tela
            col1, col2, col3 = st.columns(3)
            col1.metric("Preço Atual", f"US$ {fechamento_atual:.2f}")
            col2.metric("Máxima (30d)", f"US$ {maximo_30d:.2f}")
            col3.metric("Mínima (30d)", f"US$ {minimo_30d:.2f}")
            
            # Desenha um gráfico de linha interativo!
            st.subheader("Gráfico de Preços (Últimos 30 dias)")
            st.line_chart(dados[col_close])

            # 3. GERANDO O RELATÓRIO DA IA
            st.subheader("🧠 Relatório da Inteligência Artificial")
            with st.spinner("O Gemini está analisando os dados..."):
                resumo_para_ia = f"""
                Atue como um analista financeiro. O ativo {ativo_escolhido} teve nos últimos 30 dias:
                - Preço Atual: US$ {fechamento_atual:.2f}
                - Máximo: US$ {maximo_30d:.2f}
                - Mínimo: US$ {minimo_30d:.2f}
                Gere um breve relatório explicando se é um cenário de alta ou baixa e o risco.
                """
                
                resposta = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=resumo_para_ia
                )
                
                # st.info escreve o texto final da IA numa caixa com fundo azul claro
                st.info(resposta.text)

        except Exception as e:
            st.error(f"Erro ao buscar os dados. Verifique se o código da moeda está correto. Detalhe: {e}")
