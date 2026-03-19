import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai

# ==========================================
# CONFIGURAÇÃO GERAL
# ==========================================
st.set_page_config(page_title="Terminal Crypto IA Pro", page_icon="⚡", layout="wide")
MINHA_CHAVE_API = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=MINHA_CHAVE_API)

# ==========================================
# FUNÇÕES DE BACKEND (PROCESSAMENTO)
# ==========================================
@st.cache_data(ttl=3600)
def coletar_dados_financeiros(ticker, periodo):
    dados = yf.download(ticker, period=period, progress=False)
    
    if isinstance(dados.columns, pd.MultiIndex):
        dados.columns = ["_".join(map(str, col)).strip() for col in dados.columns.values]
        
    col_close = next(c for c in dados.columns if "Close" in c)
    col_high  = next(c for c in dados.columns if "High"  in c)
    col_low   = next(c for c in dados.columns if "Low"   in c)
    col_open  = next(c for c in dados.columns if "Open"  in c)
    col_vol   = next(c for c in dados.columns if "Volume" in c)
    
    dados = dados.dropna(subset=[col_close, col_high, col_low, col_open])
    for col in [col_close, col_high, col_low, col_open, col_vol]:
        dados[col] = pd.to_numeric(dados[col], errors="coerce")
    
    dados = dados.rename(columns={col_close: 'Close', col_high: 'High', col_low: 'Low', col_open: 'Open', col_vol: 'Volume'})
    
    # --- INDICADORES TÉCNICOS PROFISSIONAIS ---
    # 1. Bandas de Bollinger (Volatilidade)
    dados['SMA_20'] = dados['Close'].rolling(window=20).mean()
    dados['Std_Dev'] = dados['Close'].rolling(window=20).std()
    dados['Bollinger_Upper'] = dados['SMA_20'] + (dados['Std_Dev'] * 2)
    dados['Bollinger_Lower'] = dados['SMA_20'] - (dados['Std_Dev'] * 2)
    
    # 2. MACD (Convergência/Divergência de Médias Móveis)
    ema_12 = dados['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = dados['Close'].ewm(span=26, adjust=False).mean()
    dados['MACD_Line'] = ema_12 - ema_26
    dados['MACD_Signal'] = dados['MACD_Line'].ewm(span=9, adjust=False).mean()
    dados['MACD_Hist'] = dados['MACD_Line'] - dados['MACD_Signal']
    
    # 3. RSI (Força Relativa)
    delta = dados['Close'].diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = ganho / perda
    dados['RSI_14'] = 100 - (100 / (1 + rs))
    
    return dados.dropna()

def gerar_grafico_profissional(dados, nome_ativo):
    # Cria um gráfico com 2 painéis (Preço em cima, MACD embaixo)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Painel 1: Candlestick e Bandas de Bollinger
    fig.add_trace(go.Candlestick(x=dados.index, open=dados['Open'], high=dados['High'], low=dados['Low'], close=dados['Close'], name="Preço"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Bollinger_Upper'], line=dict(color='gray', width=1, dash='dash'), name='Banda Sup.'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Bollinger_Lower'], line=dict(color='gray', width=1, dash='dash'), name='Banda Inf.', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['SMA_20'], line=dict(color='blue', width=1.5), name='Média 20d'), row=1, col=1)

    # Painel 2: MACD
    cores_macd = ['green' if val >= 0 else 'red' for val in dados['MACD_Hist']]
    fig.add_trace(go.Bar(x=dados.index, y=dados['MACD_Hist'], marker_color=cores_macd, name='MACD Hist'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['MACD_Line'], line=dict(color='orange', width=1.5), name='MACD Line'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['MACD_Signal'], line=dict(color='cyan', width=1.5), name='Signal Line'), row=2, col=1)

    fig.update_layout(title=f'Análise Técnica Avançada - {nome_ativo}', template='plotly_dark', xaxis_rangeslider_visible=False, height=700)
    return fig

# ==========================================
# INTERFACE DO USUÁRIO (FRONTEND)
# ==========================================
st.title("⚡ Terminal de Trading IA Avançado")

# Opções de Entrada na Tela
col_input1, col_input2, col_input3 = st.columns([2, 2, 1])
with col_input1:
    ativo_escolhido = st.text_input("Ativo (ex: BTC-USD, PETR4.SA)", value="BTC-USD")
with col_input2:
    # O usuário agora pode escolher o período de análise na tela
    periodo_escolhido = st.selectbox("Período de Análise", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2) # Padrão: 6 meses
with col_input3:
    st.write("") # Espaçamento
    btn_analisar = st.button("Executar Análise", use_container_width=True)

if btn_analisar:
    with st.spinner(f"Processando dados institucionais para {ativo_escolhido}..."):
        try:
            dados = coletar_dados_financeiros(ativo_escolhido, periodo_escolhido)
            
            fechamento_atual = dados['Close'].iloc[-1]
            rsi_atual = dados['RSI_14'].iloc[-1]
            macd_atual = dados['MACD_Line'].iloc[-1]
            macd_signal = dados['MACD_Signal'].iloc[-1]
            bollinger_up = dados['Bollinger_Upper'].iloc[-1]
            bollinger_down = dados['Bollinger_Lower'].iloc[-1]
            
            col_grafico, col_ia = st.columns([2.5, 1]) 
            
            with col_grafico:
                # Topo de métricas
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Preço Atual", f"US$ {fechamento_atual:.2f}")
                m2.metric("RSI (14d)", f"{rsi_atual:.2f}", "Alerta: Sobrecompra" if rsi_atual > 70 else "Alerta: Sobrevenda" if rsi_atual < 30 else "Neutro")
                m3.metric("MACD", "Alta (Bullish)" if macd_atual > macd_signal else "Baixa (Bearish)", delta_color="normal" if macd_atual > macd_signal else "inverse")
                m4.metric("Distância Banda Sup.", f"{((bollinger_up - fechamento_atual) / fechamento_atual * 100):.1f}%")
                
                # Gráfico com os 2 painéis
                st.plotly_chart(gerar_grafico_profissional(dados, ativo_escolhido), use_container_width=True)
                
            with col_ia:
                st.subheader("🧠 Conselheiro IA Sênior")
                
                # O Prompt Sênior: Enviamos TODOS os indicadores para a IA pensar como um Robô de Wall Street
                resumo_para_ia = f"""
                Atue como um Algoritmo de Trading Quantitativo Sênior. Avalie o ativo {ativo_escolhido}.
                
                DADOS TÉCNICOS ATUAIS:
                - Preço: US$ {fechamento_atual:.2f}
                - RSI (14 dias): {rsi_atual:.2f}
                - MACD Line: {macd_atual:.2f} | Signal Line: {macd_signal:.2f}
                - Banda de Bollinger Superior: US$ {bollinger_up:.2f}
                - Banda de Bollinger Inferior: US$ {bollinger_down:.2f}
                
                REGRAS DE ANÁLISE:
                1. MACD Line acima da Signal Line indica tendência de alta.
                2. Preço encostando na Banda Superior indica sobrecompra iminente; na Inferior, oportunidade de compra.
                3. RSI acima de 70 = Cuidado (correção); abaixo de 30 = Fundo (potencial alta).
                
                Forneça um relatório direto, citando os dados. Termine com um veredito claro: COMPRAR, VENDER ou MANTER.
                """
                
                resposta = client.models.generate_content(model="gemini-2.5-flash", contents=resumo_para_ia)
                st.info(resposta.text)

        except Exception as e:
            st.error(f"Erro ao processar indicadores. Tente aumentar o período de análise. Detalhe: {e}")
