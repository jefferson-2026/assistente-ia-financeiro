import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from google import genai

# ==========================================
# CONFIGURAÇÃO GERAL (NÍVEL ENTERPRISE)
# ==========================================
st.set_page_config(page_title="Terminal Crypto IA", page_icon="📊", layout="wide") # Layout 'wide' usa a tela toda
MINHA_CHAVE_API = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=MINHA_CHAVE_API)

# ==========================================
# FUNÇÕES DE BACKEND (PROCESSAMENTO)
# ==========================================

# O @st.cache_data faz o app não baixar os mesmos dados repetidas vezes (TTL = expira em 1 hora)
@st.cache_data(ttl=3600)
def coletar_dados_financeiros(ticker, periodo="60d"):
    dados = yf.download(ticker, period=period, progress=False)
    
    # Tratamento de MultiIndex do yfinance
    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = ["_".join(map(str, col)).strip() for col in dados.columns.values]
        
    col_close = next(c for c in dados.columns if "Close" in c)
    col_high  = next(c for c in dados.columns if "High"  in c)
    col_low   = next(c for c in dados.columns if "Low"   in c)
    col_open  = next(c for c in dados.columns if "Open"  in c)
    
    dados = dados.dropna(subset=[col_close, col_high, col_low, col_open])
    for col in [col_close, col_high, col_low, col_open]:
        dados[col] = pd.to_numeric(dados[col], errors="coerce")
    
    # Renomeando para facilitar nosso uso
    dados = dados.rename(columns={col_close: 'Close', col_high: 'High', col_low: 'Low', col_open: 'Open'})
    
    # --- CÁLCULO DE INDICADORES TÉCNICOS ---
    # 1. Média Móvel Simples de 20 dias (SMA)
    dados['SMA_20'] = dados['Close'].rolling(window=20).mean()
    
    # 2. RSI (Índice de Força Relativa) de 14 dias
    delta = dados['Close'].diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = ganho / perda
    dados['RSI_14'] = 100 - (100 / (1 + rs))
    
    return dados.dropna() # Removemos os primeiros dias que ficam sem média/RSI

def gerar_grafico_candlestick(dados, nome_ativo):
    fig = go.Figure(data=[go.Candlestick(x=dados.index,
                open=dados['Open'], high=dados['High'],
                low=dados['Low'], close=dados['Close'],
                name="Preço")])
    # Adicionando a linha da Média Móvel no gráfico
    fig.add_trace(go.Scatter(x=dados.index, y=dados['SMA_20'], opacity=0.7, line=dict(color='blue', width=2), name='Média Móvel 20d'))
    fig.update_layout(title=f'Gráfico de Preços e Tendência - {nome_ativo}', template='plotly_dark', xaxis_rangeslider_visible=False)
    return fig

# ==========================================
# INTERFACE DO USUÁRIO (FRONTEND)
# ==========================================
st.title("⚡ Terminal de Trading IA Avançado")

# Divide a tela de controles em colunas
col_input1, col_input2 = st.columns([1, 3])
with col_input1:
    ativo_escolhido = st.text_input("Ativo (ex: BTC-USD)", value="BTC-USD")
    btn_analisar = st.button("Executar Análise Técnica", use_container_width=True)

if btn_analisar:
    with st.spinner("Extraindo dados do mercado e calculando indicadores..."):
        try:
            dados = coletar_dados_financeiros(ativo_escolhido, periodo="60d")
            
            fechamento_atual = dados['Close'].iloc[-1]
            rsi_atual = dados['RSI_14'].iloc[-1]
            sma_atual = dados['SMA_20'].iloc[-1]
            
            # Divide a tela de resultados em duas grandes colunas
            col_grafico, col_ia = st.columns([2, 1]) # O gráfico ocupa 2/3 da tela, a IA 1/3
            
            with col_grafico:
                # Métricas Rápidas
                m1, m2, m3 = st.columns(3)
                m1.metric("Preço Atual", f"US$ {fechamento_atual:.2f}")
                m2.metric("RSI (Força Relativa)", f"{rsi_atual:.2f}", 
                          "Sobrecomprado (>70)" if rsi_atual > 70 else "Sobrevendido (<30)" if rsi_atual < 30 else "Neutro")
                m3.metric("Média Móvel (20d)", f"US$ {sma_atual:.2f}")
                
                # Exibe o Gráfico Profissional
                st.plotly_chart(gerar_grafico_candlestick(dados, ativo_escolhido), use_container_width=True)
                
            with col_ia:
                st.subheader("🧠 Conselheiro IA de Trading")
                # Prompt avançado com indicadores técnicos
                resumo_para_ia = f"""
                Atue como um Trader Sênior especialista em análise técnica de criptomoedas.
                Ativo: {ativo_escolhido}
                - Preço Atual: US$ {fechamento_atual:.2f}
                - Média Móvel de 20 dias: US$ {sma_atual:.2f}
                - RSI (14 dias): {rsi_atual:.2f}
                
                Instruções:
                1. Diga se o ativo está em tendência de alta ou baixa comparando o preço com a Média Móvel.
                2. Avalie o RSI (acima de 70 indica sobrecompra/risco de queda, abaixo de 30 indica sobrevenda/oportunidade).
                3. Conclua com um veredito curto: Compra, Venda ou Espera (Hold), justificando o risco.
                """
                
                resposta = client.models.generate_content(model="gemini-2.5-flash", contents=resumo_para_ia)
                st.info(resposta.text)

        except Exception as e:
            st.error(f"Erro na extração. Verifique o código do ativo. Detalhe: {e}")
