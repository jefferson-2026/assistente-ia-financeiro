import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
from fpdf import FPDF
import tempfile
import os
import time

# ==========================================
# CONFIGURAÇÃO GERAL
# ==========================================
st.set_page_config(page_title="Terminal Crypto IA Pro", page_icon="⚡", layout="wide")
MINHA_CHAVE_API = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=MINHA_CHAVE_API)

# ==========================================
# FUNÇÕES DE BACKEND E PDF
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

    fig.update_layout(title=f'Análise Técnica Institucional - {nome_ativo}', template='plotly_dark', xaxis_rangeslider_visible=False, height=850)
    return fig

def gerar_pdf_relatorio(ticker, fechamento, rsi, macd, sma200, texto_ia, fig_original):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cabeçalho do PDF
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, f"Relatorio de Analise Oficial: {ticker}", 0, 1, 'C')
    pdf.ln(5)
    
    # Detecção Inteligente: Estamos na nuvem do Streamlit ou Local?
    rodando_na_nuvem = "STREAMLIT_SERVER_PORT" in os.environ or "KUBERNETES_PORT" in os.environ
    
    if not rodando_na_nuvem:
        # ---- MODO LOCAL: GERA A IMAGEM DO GRÁFICO ----
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                fig_pdf = go.Figure(fig_original)
                fig_pdf.update_layout(template='plotly_white', height=450) 
                fig_pdf.write_image(tmp.name, format="png", engine="kaleido")
                time.sleep(0.5) 
                pdf.image(tmp.name, x=10, w=190)
                os.unlink(tmp.name)
        except Exception as e:
            # Se der erro mesmo localmente, força a cair no plano da tabela
            rodando_na_nuvem = True 
    
    if rodando_na_nuvem:
        # ---- MODO NUVEM: GERA UMA TABELA FINANCEIRA ELEGANTE ----
        pdf.set_font("helvetica", 'B', 12)
        pdf.cell(0, 10, "Indicadores Tecnicos de Curto e Longo Prazo", 0, 1)
        
        pdf.set_font("helvetica", '', 10)
        with pdf.table(col_widths=(40, 60), text_align="LEFT") as table:
            row = table.row()
            row.cell("Preco Atual:")
            row.cell(f"US$ {fechamento:.2f}")
            
            row = table.row()
            row.cell("MME 200 Dias:")
            row.cell(f"US$ {sma200:.2f}")
            
            row = table.row()
            row.cell("RSI (14 Dias):")
            row.cell(f"{rsi:.2f} " + ("(Sobrecomprado)" if rsi > 70 else "(Sobrevendido)" if rsi < 30 else "(Neutro)"))
            
            row = table.row()
            row.cell("MACD Line:")
            row.cell(f"{macd:.2f}")
        pdf.ln(5)

    # TEXTO DA INTELIGÊNCIA ARTIFICIAL
    pdf.ln(5)
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "Veredito do Algoritmo IA (Fundo Hedge)", 0, 1)
    pdf.ln(5)
    
    # Limpeza do Markdown para formato puro de texto no PDF
    pdf.set_font("helvetica", '', 11)
    texto_limpo = str(texto_ia).replace("**", "").replace("*", "").replace("#", "")
    texto_limpo = texto_limpo.encode('latin-1', 'replace').decode('latin-1') 
    
    pdf.multi_cell(0, 6, texto_limpo)
        
    return bytes(pdf.output())

# ==========================================
# INTERFACE DO USUÁRIO (FRONTEND TELA CHEIA)
# ==========================================
st.title("⚡ Terminal Institucional Crypto & IA")

# Barra de controles
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
    usar_ia = st.toggle("🤖 Usar IA Especialista", value=True)
with col_input4:
    st.write("") 
    btn_analisar = st.button("Executar Análise Completa", type="primary", use_container_width=True)
st.markdown("---")

if btn_analisar:
    with st.spinner(f"Processando algoritmo institucional para {ativo_escolhido}..."):
        try:
            # 1. COLETA E PROCESSAMENTO
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
            
            # 2. MÉTRICAS EM TELA CHEIA
            st.markdown("### 📊 Indicadores Quantitativos Atuais")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Preço Atual", f"US$ {fechamento_atual:.2f}")
            m2.metric("RSI (14d)", f"{rsi_atual:.2f}", "Alerta: Sobrecompra" if rsi_atual > 70 else "Alerta: Sobrevenda" if rsi_atual < 30 else "Neutro", delta_color="off")
            m3.metric("MACD Status", "Alta (Bullish)" if macd_atual > macd_signal else "Baixa (Bearish)", delta_color="normal" if macd_atual > macd_signal else "inverse")
            m4.metric("SMA 200 (Macro)", f"US$ {sma_200:.2f}")
            
            # 3. GRÁFICO EM TELA CHEIA
            st.markdown("### 📈 Gráfico de Preços, Volume e Tendência")
            fig = gerar_grafico_profissional(dados_plot, ativo_escolhido)
            st.plotly_chart(fig, use_container_width=True)
            
            # 4. RELATÓRIO DA IA E BOTÃO PDF (ABAIXO DO GRÁFICO)
            if usar_ia:
                st.markdown("---")
                st.markdown("### 🧠 Veredito do Fundo (Inteligência Artificial)")
                with st.spinner("O Head Trader IA está redigindo o relatório..."):
                    
                    # PROMPT ESPECIALIZADO DE TRADE
                    prompt = f"""
                    Atue como um Head Trader de um Fundo Hedge de Criptomoedas. 
                    Escreva um relatório executivo detalhado para o ativo {ativo_escolhido}.
                    
                    DADOS TÉCNICOS EXTRAÍDOS:
                    - Preço Atual: US$ {fechamento_atual:.2f}
                    - SMA 20 (Curto Prazo): US$ {sma_20:.2f}
                    - SMA 200 (Macro): US$ {sma_200:.2f}
                    - RSI (14d): {rsi_atual:.2f}
                    - MACD Line: {macd_atual:.2f} | Signal Line: {macd_signal:.2f}
                    - Bandas de Bollinger: Teto US$ {bollinger_up:.2f} | Piso US$ {bollinger_down:.2f}
                    
                    FORMATO DE SAÍDA (Use títulos e seja altamente técnico):
                    ## 🎯 Resumo Executivo
                    (1 parágrafo direto com a tese principal)
                    
                    ## 📊 Análise de Price Action e Momento
                    (Interpretação do preço atual perante as médias e Bandas de Bollinger)
                    
                    ## 🔬 Osciladores (RSI e MACD)
                    (Avaliação clara de sobrecompra/sobrevenda e força de cruzamentos)
                    
                    ## ⚖️ Veredito Oficial
                    (Conclua claramente: COMPRA FORTE, COMPRA, MANTER, VENDA ou VENDA FORTE. Justifique o risco.)
                    """
                    
                    resposta = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    st.markdown(resposta.text)
                    
                    # 5. GERAÇÃO E DOWNLOAD DO PDF
                    with st.spinner("Gerando arquivo de PDF para download..."):
                        pdf_bytes = gerar_pdf_relatorio(
                            ativo_escolhido, fechamento_atual, rsi_atual, 
                            macd_atual, sma_200, resposta.text, fig
                        )
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.download_button(
                            label="📥 Baixar Relatório Oficial em PDF",
                            data=pdf_bytes,
                            file_name=f"Relatorio_Trading_{ativo_escolhido}.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"Erro inesperado ao buscar dados. Detalhe: {e}")
