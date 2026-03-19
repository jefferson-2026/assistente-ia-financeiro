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
def coletar_dados_historicos(ticker):
    # TRUQUE SÊNIOR: Buscamos 5 anos escondido para calcular tudo perfeitamente
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
    
    # --- CÁLCULO DE INDICADORES (Com Histórico Completo) ---
    dados['SMA_20'] = dados['Close'].rolling(window=20).mean()
    dados['SMA_200'] = dados['Close'].rolling(window=200).mean() # Nova: Tendência Macro
    
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
    # Gráfico agora tem 3 painéis (Preço, Volume, MACD)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])

    # Painel 1: Preço e Médias
    fig.add_trace(go.Candlestick(x=dados.index, open=dados['Open'], high=dados['High'], low=dados['Low'], close=dados['Close'], name="Preço"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Bollinger_Upper'], line=dict(color='gray', width=1, dash='dash'), name='Banda Sup.'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Bollinger_Lower'], line=dict(color='gray', width=1, dash='dash'), name='Banda Inf.', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['SMA_20'], line=dict(color='blue', width=1.5), name='Média 20d'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['SMA_200'], line=dict(color='purple', width=2), name='Média 200d'), row=1, col=1)

    # Painel 2: Volume
    cores_vol = ['green' if row['Close'] >= row['Open'] else 'red' for index, row in dados.iterrows()]
    fig.add_trace(go.Bar(x=dados.index, y=dados['Volume'], marker_color=cores_vol, name='Volume'), row=2, col=1)

    # Painel 3: MACD
    cores_macd = ['green' if val >= 0 else 'red' for val in dados['MACD_Hist']]
    fig.add_trace(go.Bar(x=dados.index, y=dados['MACD_Hist'], marker_color=cores_macd, name='MACD Hist'), row=3, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['MACD_Line'], line=dict(color='orange', width=1.5), name='MACD Line'), row=3, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados['MACD_Signal'], line=dict(color='cyan', width=1.5), name='Signal Line'), row=3, col=1)

    fig.update_layout(title=f'Análise Técnica Institucional - {nome_ativo}', template='plotly_dark', xaxis_rangeslider_visible=False, height=850)
    return fig

# ==========================================
# INTERFACE DO USUÁRIO (FRONTEND)
# ==========================================
st.title("⚡ Terminal Institucional Crypto & IA")

# Barra de controles
st.markdown("---")
col_input1, col_input2, col_input3, col_input4 = st.columns([2, 2, 2, 2])
with col_input1:
    ativo_escolhido = st.text_input("Ticker do Ativo", value="BTC-USD")
with col_input2:
    # Removemos 1d e 5d pois velas diárias ficam estranhas para períodos tão curtos
    opcoes_periodo = {"1 Mês": 30, "3 Meses": 90, "6 Meses": 180, "1 Ano": 365, "2 Anos": 730, "5 Anos": 1825, "Máximo": 99999}
    selecao_texto = st.selectbox("Período de Visualização", list(opcoes_periodo.keys()), index=2)
    dias_plot = opcoes_periodo[selecao_texto]
with col_input3:
    st.write("") # Alinhamento
    # CHAVE DE ATIVAÇÃO DA IA
    usar_ia = st.toggle("🤖 Usar IA Especialista", value=True, help="Ative para gerar o relatório usando tokens.")
with col_input4:
    st.write("") 
    btn_analisar = st.button("Executar Análise", type="primary", use_container_width=True)
st.markdown("---")

if btn_analisar:
    with st.spinner(f"Processando histórico de dados para {ativo_escolhido}..."):
        try:
            # 1. Puxa todos os dados e calcula tudo
            dados_completos = coletar_dados_historicos(ativo_escolhido)
            
            # 2. Corta (slice) apenas a quantidade de dias que o usuário quer ver
            dados_plot = dados_completos.tail(dias_plot)
            
            # 3. Pega os valores da última linha (dados mais recentes de hoje)
            fechamento_atual = dados_plot['Close'].iloc[-1]
            rsi_atual = dados_plot['RSI_14'].iloc[-1]
            macd_atual = dados_plot['MACD_Line'].iloc[-1]
            macd_signal = dados_plot['MACD_Signal'].iloc[-1]
            bollinger_up = dados_plot['Bollinger_Upper'].iloc[-1]
            bollinger_down = dados_plot['Bollinger_Lower'].iloc[-1]
            sma_20 = dados_plot['SMA_20'].iloc[-1]
            sma_200 = dados_plot['SMA_200'].iloc[-1]
            volume_atual = dados_plot['Volume'].iloc[-1]
            
            # Ajuste de layout dinâmico (Se a IA estiver desligada, o gráfico ocupa 100%)
            col_grafico, col_ia = st.columns([2.5, 1.2]) if usar_ia else st.columns([1, 0.01])
            
            with col_grafico:
                # Métricas Rápidas
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Preço Atual", f"US$ {fechamento_atual:.2f}")
                m2.metric("RSI (14d)", f"{rsi_atual:.2f}", "Alerta: Sobrecompra" if rsi_atual > 70 else "Alerta: Sobrevenda" if rsi_atual < 30 else "Neutro", delta_color="off")
                m3.metric("MACD", "Alta" if macd_atual > macd_signal else "Baixa", delta_color="normal" if macd_atual > macd_signal else "inverse")
                m4.metric("SMA 200 (Macro)", f"US$ {sma_200:.2f}")
                
                st.plotly_chart(gerar_grafico_profissional(dados_plot, ativo_escolhido), use_container_width=True)
                
            if usar_ia:
                with col_ia:
                    st.subheader("🧠 Relatório Quantitativo IA")
                    with st.spinner("Analisando indicadores..."):
                        # O Novo Prompt Profissional
                        prompt = f"""
                        Você é um Trader Quantitativo Institucional Sênior, especialista em Criptomoedas e Ações.
                        Gere um relatório de análise técnica avançado e altamente profissional em Markdown para o ativo {ativo_escolhido}.
                        
                        DADOS TÉCNICOS EXTRAÍDOS HOJE:
                        - Preço Atual: US$ {fechamento_atual:.2f}
                        - SMA (20 dias): US$ {sma_20:.2f} (Tendência Curta)
                        - SMA (200 dias): US$ {sma_200:.2f} (Tendência Macro)
                        - RSI (14 dias): {rsi_atual:.2f}
                        - MACD Line: {macd_atual:.2f} | Signal Line: {macd_signal:.2f}
                        - Bandas de Bollinger: Teto US$ {bollinger_up:.2f} | Piso US$ {bollinger_down:.2f}
                        
                        ESTRUTURA OBRIGATÓRIA DA SUA RESPOSTA:
                        1. 📊 **Visão Geral:** Resumo executivo do momento atual.
                        2. 🔬 **Leitura Técnica:** Interpretação direta do RSI, cruzamento MACD e médias.
                        3. 🎯 **Zonas de Interesse:** Estipule possíveis suportes e resistências.
                        4. ⚖️ **Veredito:** [COMPRA FORTE / COMPRA / MANTER / VENDA / VENDA FORTE] com justificativa de risco.
                        
                        Seja direto, frio e analítico. Sem enrolação.
                        """
                        
                        resposta = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                        st.markdown(resposta.text)

        except Exception as e:
            st.error(f"Erro inesperado ao buscar dados. Detalhe: {e}")
