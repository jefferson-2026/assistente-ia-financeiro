import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
import requests
from datetime import datetime, timedelta
from fpdf import FPDF
import tempfile
import os
import time

# ==========================================
# CONFIGURAÇÃO GERAL
# ==========================================
st.set_page_config(page_title="Terminal Crypto IA Pro", page_icon="⚡", layout="wide")
MINHA_CHAVE_API = st.secrets["GEMINI_API_KEY"]
MINHA_CHAVE_NEWS = st.secrets["NEWS_API_KEY"]
client = genai.Client(api_key=MINHA_CHAVE_API)

# ==========================================
# FUNÇÕES DE BACKEND (GRÁFICOS E NOTÍCIAS)
# ==========================================
@st.cache_data(ttl=3600)
def coletar_dados_historicos(ticker):
    dados = yf.download(ticker, period="5y", progress=False)
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
    
    dados['SMA_20'] = dados['Close'].rolling(window=20).mean()
    dados['SMA_200'] = dados['Close'].rolling(window=200).mean() 
    dados['Std_Dev'] = dados['Close'].rolling(window=20).std()
    dados['Bollinger_Upper'] = dados['SMA_20'] + (dados['Std_Dev'] * 2)
    dados['Bollinger_Lower'] = dados['SMA_20'] - (dados['Std_Dev'] * 2)
    
    ema_12 = dados['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = dados['Close'].ewm(span=26, adjust=False).mean()
    dados['MACD_Line'] = ema_12 - ema_26
    dados['MACD_Signal'] = dados['MACD_Line'].ewm(span=9, adjust=False).mean()
    dados['MACD_Hist'] = dados['MACD_Line'] - dados['MACD_Signal']
    
    delta = dados['Close'].diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = ganho / perda
    dados['RSI_14'] = 100 - (100 / (1 + rs))
    return dados.dropna()

def gerar_grafico_profissional(dados, nome_ativo):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])
    fig.add_trace(go.Candlestick(x=dados.index, open=dados['Open'], high=dados['High'], low=dados['Low'], close=dados['Close'], name="Preço"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Bollinger_Upper'], line=dict(color='gray', width=1, dash='dash'), name='Banda Sup.'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Bollinger_Lower'], line=dict(color='gray', width=1, dash='dash'), name='Banda Inf.', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['SMA_20'], line=dict(color='blue', width=1.5), name='Média 20d'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['SMA_200'], line=dict(color='purple', width=2), name='Média 200d'), row=1, col=1)
    cores_vol = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in dados.iterrows()]
    fig.add_trace(go.Bar(x=dados.index, y=dados['Volume'], marker_color=cores_vol, name='Volume'), row=2, col=1)
    cores_macd = ['green' if val >= 0 else 'red' for val in dados['MACD_Hist']]
    fig.add_trace(go.Bar(x=dados.index, y=dados['MACD_Hist'], marker_color=cores_macd, name='MACD Hist'), row=3, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['MACD_Line'], line=dict(color='orange', width=1.5), name='MACD Line'), row=3, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['MACD_Signal'], line=dict(color='cyan', width=1.5), name='Signal Line'), row=3, col=1)
    fig.update_layout(title=f'Análise Técnica - {nome_ativo}', template='plotly_dark', xaxis_rangeslider_visible=False, height=750, margin=dict(l=0, r=0, t=40, b=0))
    return fig

# NOVA FUNÇÃO: Busca manchetes de notícias do mercado financeiro mundial
@st.cache_data(ttl=1800) # Cacheia por 30 minutos para não estourar limite da API
def buscar_noticias(ticker):
    # Se for BTC-USD, busca por Bitcoin. Se for PETR4, Petrobras, etc.
    termo_busca = ticker.split('-')[0] 
    data_ontem = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    url = f"https://newsapi.org/v2/everything?q={termo_busca}&from={data_ontem}&sortBy=relevancy&language=en&apiKey={MINHA_CHAVE_NEWS}"
    try:
        resposta = requests.get(url)
        dados = resposta.json()
        if dados['status'] == 'ok':
            # Pega só as 5 notícias mais importantes
            artigos = dados['articles'][:5]
            texto_compilado = ""
            for art in artigos:
                texto_compilado += f"- Título: {art['title']}\n  Fonte: {art['source']['name']}\n\n"
            return artigos, texto_compilado
        return [], "Nenhuma notícia relevante encontrada."
    except Exception as e:
        return [], f"Erro na coleta de notícias: {e}"

# ==========================================
# INTERFACE DO USUÁRIO
# ==========================================
st.title("⚡ Terminal Institucional Crypto & IA")

st.markdown("---")
col_input1, col_input2, col_input3, col_input4 = st.columns([2, 2, 2, 2])
with col_input1:
    ativo_escolhido = st.text_input("Ticker do Ativo", value="BTC-USD")
with col_input2:
    opcoes_periodo = {"1 Mês": 30, "3 Meses": 90, "6 Meses": 180, "1 Ano": 365, "2 Anos": 730, "5 Anos": 1825, "Máximo": 99999}
    selecao_texto = st.selectbox("Período de Visualização", list(opcoes_periodo.keys()), index=2)
    dias_plot = opcoes_periodo[selecao_texto]
with col_input3:
    st.write("") 
    usar_ia = st.toggle("🤖 Usar IA (Análise Gráfica + Sentimento)", value=True)
with col_input4:
    st.write("") 
    btn_analisar = st.button("Executar Scanner Global", type="primary", use_container_width=True)
st.markdown("---")

if btn_analisar:
    with st.spinner(f"Processando gráficos e vasculhando a mídia global por {ativo_escolhido}..."):
        try:
            # 1. PROCESSAMENTO DE DADOS
            dados_completos = coletar_dados_historicos(ativo_escolhido)
            dados_plot = dados_completos.tail(dias_plot)
            
            fechamento_atual = dados_plot['Close'].iloc[-1]
            rsi_atual = dados_plot['RSI_14'].iloc[-1]
            macd_atual = dados_plot['MACD_Line'].iloc[-1]
            macd_signal = dados_plot['MACD_Signal'].iloc[-1]
            bollinger_up = dados_plot['Bollinger_Upper'].iloc[-1]
            bollinger_down = dados_plot['Bollinger_Lower'].iloc[-1]
            sma_20 = dados_plot['SMA_20'].iloc[-1]
            sma_200 = dados_plot['SMA_200'].iloc[-1]
            
            # 2. BUSCA DE NOTÍCIAS
            lista_noticias, string_noticias = buscar_noticias(ativo_escolhido)
            
            # 3. INTERFACE COM ABAS (TABS) PARA ORGANIZAÇÃO LIMPA
            tab1, tab2 = st.tabs(["📈 Gráficos & Métricas", "📰 Feed de Notícias Globais"])
            
            with tab1:
                # Métricas Rápidas
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Preço Atual", f"US$ {fechamento_atual:.2f}")
                m2.metric("RSI (14d)", f"{rsi_atual:.2f}", "Alerta: Sobrecompra" if rsi_atual > 70 else "Alerta: Sobrevenda" if rsi_atual < 30 else "Neutro", delta_color="off")
                m3.metric("MACD Status", "Alta (Bullish)" if macd_atual > macd_signal else "Baixa (Bearish)", delta_color="normal" if macd_atual > macd_signal else "inverse")
                m4.metric("SMA 200 (Macro)", f"US$ {sma_200:.2f}")
                
                # Gráfico
                fig = gerar_grafico_profissional(dados_plot, ativo_escolhido)
                st.plotly_chart(fig, use_container_width=True)
                
            with tab2:
                st.subheader(f"Últimas manchetes sobre {ativo_escolhido.split('-')[0]}")
                if len(lista_noticias) > 0:
                    for art in lista_noticias:
                        with st.container(border=True):
                            st.markdown(f"**{art['title']}**")
                            st.caption(f"Fonte: {art['source']['name']} | Data: {art['publishedAt'][:10]}")
                else:
                    st.info("Nenhuma notícia de alto impacto encontrada nos últimos 3 dias.")
            
            # 4. RELATÓRIO DUPLO DA IA (TÉCNICO + FUNDAMENTALISTA)
            if usar_ia:
                st.markdown("---")
                st.markdown("### 🧠 Veredito Final (Technical & Fundamental Analysis)")
                with st.spinner("O Head Trader está cruzando gráficos com o sentimento das notícias..."):
                    
                    # PROMPT SÊNIOR EVOLUÍDO
                    prompt = f"""
                    Atue como o Head Trader (Gestor Principal) de um Fundo Hedge. 
                    Você recebeu os dados técnicos dos analistas quantitativos e o resumo de notícias globais.
                    Gere um relatório executivo cruzando as duas informações para o ativo {ativo_escolhido}.
                    
                    1. DADOS TÉCNICOS:
                    Preço: US${fechamento_atual:.2f} | SMA 20: US${sma_20:.2f} | SMA 200: US${sma_200:.2f}
                    RSI: {rsi_atual:.2f} | MACD Line: {macd_atual:.2f} (Signal: {macd_signal:.2f})
                    Bollinger: Teto US${bollinger_up:.2f}, Piso US${bollinger_down:.2f}
                    
                    2. MANCHETES RECENTES (Sentimento Fundamentalista):
                    {string_noticias}
                    
                    Sua Tarefa (Retorne em Markdown limpo):
                    ## 🎯 Resumo Executivo
                    (1 parágrafo juntando preço e sentimento macroeconômico)
                    
                    ## 📊 Leitura Gráfica
                    (Análise curta dos indicadores)
                    
                    ## 📰 Termômetro de Sentimento
                    (As notícias atuais ajudam ou atrapalham a tendência do gráfico?)
                    
                    ## ⚖️ Veredito de Operação
                    (COMPRA FORTE, COMPRA, MANTER, VENDA ou VENDA FORTE. E qual o Risco da operação).
                    """
                    
                    resposta = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    st.success("Análise de Múltiplos Fatores Concluída!")
                    st.markdown(resposta.text)

        except Exception as e:
            st.error(f"Erro no processamento global. Detalhe: {e}")
