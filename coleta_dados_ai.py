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
import vectorbt as vbt 

# ==========================================
# CONFIGURAÇÃO GERAL E CHAVES
# ==========================================
st.set_page_config(page_title="Terminal Crypto IA Pro", page_icon="⚡", layout="wide")
MINHA_CHAVE_API = st.secrets.get("GEMINI_API_KEY", "")
MINHA_CHAVE_NEWS = st.secrets.get("NEWS_API_KEY", "")
client = genai.Client(api_key=MINHA_CHAVE_API) if MINHA_CHAVE_API else None

# ==========================================
# FUNÇÕES DE DADOS E GRÁFICOS
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

@st.cache_data(ttl=1800) 
def buscar_noticias(ticker):
    if not MINHA_CHAVE_NEWS:
        return [], "API Key de notícias não configurada."
    termo_busca = ticker.split('-')[0] 
    data_ontem = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    url = f"https://newsapi.org/v2/everything?q={termo_busca}&from={data_ontem}&sortBy=relevancy&language=en&apiKey={MINHA_CHAVE_NEWS}"
    try:
        resposta = requests.get(url)
        dados = resposta.json()
        if dados['status'] == 'ok':
            artigos = dados['articles'][:5]
            texto_compilado = "".join([f"- Título: {art['title']} (Fonte: {art['source']['name']})\n" for art in artigos])
            return artigos, texto_compilado
        return [], "Nenhuma notícia relevante encontrada."
    except Exception as e:
        return [], f"Erro na coleta: {e}"

# --- BACKTEST MELHORADO COM ALAVANCAGEM ---
def executar_backtest_macd(dados, margem, alavancagem, taxa_corretora):
    entradas = dados['MACD_Line'] > dados['MACD_Signal']
    saidas = dados['MACD_Line'] < dados['MACD_Signal']
    
    tamanho_operacao = margem * alavancagem # Ex: US$ 100 com 10x = Posição de US$ 1000
    caixa_virtual = tamanho_operacao * 10 # Dinheiro infinito para simulação não quebrar
    
    portfolio = vbt.Portfolio.from_signals(
        dados['Close'], 
        entradas, 
        saidas, 
        size=tamanho_operacao, 
        size_type='value', 
        init_cash=caixa_virtual, 
        fees=taxa_corretora
    )
    
    fig_backtest = portfolio.plot(subplots=['orders', 'trade_pnl', 'cum_returns'])
    fig_backtest.update_layout(template='plotly_dark', height=600, title=f"Simulação: Operando US$ {tamanho_operacao:.2f} por sinal")
    
    lucro_total = portfolio.total_profit()
    win_rate = portfolio.trades.win_rate() * 100 if len(portfolio.trades) > 0 else 0
    return fig_backtest, lucro_total, win_rate

def gerar_pdf_relatorio(ticker, fechamento, rsi, macd, texto_ia):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, f"Relatorio Oficial do Fundo: {ticker}", 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 10, "Indicadores Tecnicos de Curto e Longo Prazo", 0, 1)
    pdf.set_font("helvetica", '', 10)
    with pdf.table(col_widths=(40, 60), text_align="LEFT") as table:
        row = table.row(); row.cell("Preco Atual:"); row.cell(f"US$ {fechamento:.2f}")
        row = table.row(); row.cell("RSI (14 Dias):"); row.cell(f"{rsi:.2f}")
        row = table.row(); row.cell("MACD Status:"); row.cell(f"{macd:.2f}")
    pdf.ln(5)
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "Veredito Hibrido da IA", 0, 1)
    pdf.ln(5)
    pdf.set_font("helvetica", '', 11)
    texto_limpo = str(texto_ia).replace("**", "").replace("*", "").replace("#", "")
    texto_limpo = texto_limpo.encode('latin-1', 'replace').decode('latin-1') 
    pdf.multi_cell(0, 6, texto_limpo)
    return bytes(pdf.output())

# ==========================================
# INTERFACE DO USUÁRIO
# ==========================================
st.title("⚡ Terminal Quantitativo & Algorítmico")

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
    usar_ia = st.toggle("🤖 Usar IA Híbrida", value=True)
with col_input4:
    st.write("") 
    btn_analisar = st.button("Analisar & Backtest", type="primary", use_container_width=True)

# NOVOS INPUTS DE RISCO (FORA DO BOTÃO PARA SALVAR O ESTADO)
with st.expander("⚙️ Configurações de Risco do Backtesting (Clique para abrir)", expanded=False):
    c_r1, c_r2, c_r3 = st.columns(3)
    val_margem = c_r1.number_input("Sua Margem/Banca por Trade (US$)", min_value=10.0, value=100.0, step=10.0)
    val_alavancagem = c_r2.number_input("Alavancagem (x)", min_value=1, max_value=125, value=1, step=1)
    val_taxa = c_r3.number_input("Taxa da Corretora (%)", min_value=0.0, value=0.1, step=0.05) / 100

st.markdown("---")

if btn_analisar:
    with st.spinner("Processando Wall Street..."):
        try:
            dados_completos = coletar_dados_historicos(ativo_escolhido)
            dados_plot = dados_completos.tail(dias_plot)
            
            fechamento_atual = dados_plot['Close'].iloc[-1]
            rsi_atual = dados_plot['RSI_14'].iloc[-1]
            macd_atual = dados_plot['MACD_Line'].iloc[-1]
            macd_signal = dados_plot['MACD_Signal'].iloc[-1]
            
            lista_noticias, string_noticias = buscar_noticias(ativo_escolhido)
            
            tab1, tab2, tab3 = st.tabs(["📈 Gráficos", "📰 Notícias Globais", "⏳ Máquina do Tempo (Backtest)"])
            
            with tab1:
                st.plotly_chart(gerar_grafico_profissional(dados_plot, ativo_escolhido), use_container_width=True)
                
            with tab2:
                if len(lista_noticias) > 0:
                    for art in lista_noticias:
                        with st.container(border=True):
                            st.markdown(f"**{art['title']}**")
                            st.caption(f"{art['source']['name']} | Data: {art['publishedAt'][:10]}")
                else:
                    st.info("Nenhuma notícia de alto impacto encontrada.")
                    
            with tab3:
                posicao_total = val_margem * val_alavancagem
                st.markdown(f"### Resultado da Estratégia MACD (Operando US$ {posicao_total:.2f})")
                st.write(f"Você comprometeu **US$ {val_margem:.2f}** com alavancagem de **{val_alavancagem}x** (pagando {(val_taxa*100):.2f}% de taxa).")
                
                fig_bt, lucro, winrate = executar_backtest_macd(dados_plot, val_margem, val_alavancagem, val_taxa)
                
                col_bt1, col_bt2, col_bt3 = st.columns(3)
                col_bt1.metric("Lucro Líquido Simulado", f"US$ {lucro:.2f}", delta="Gain" if lucro > 0 else "Loss", delta_color="normal" if lucro > 0 else "inverse")
                col_bt2.metric("Retorno sobre a Margem (ROE)", f"{(lucro / val_margem) * 100:.1f}%")
                col_bt3.metric("Taxa de Acertos (Win Rate)", f"{winrate:.1f}%")
                
                st.plotly_chart(fig_bt, use_container_width=True)

            if usar_ia and client:
                st.markdown("---")
                st.subheader("🧠 Veredito Final (Technical & Fundamental Analysis)")
                with st.spinner("Head Trader redigindo o relatório final..."):
                    prompt = f"""
                    Atue como o Head Trader de um Fundo Hedge. Gere um relatório executivo para {ativo_escolhido}.
                    Preço: US${fechamento_atual:.2f} | RSI: {rsi_atual:.2f} | MACD: {macd_atual:.2f}
                    NOTÍCIAS: {string_noticias}
                    
                    Formato (Markdown):
                    ## 🎯 Resumo Executivo
                    ## 📊 Leitura Gráfica
                    ## 📰 Termômetro de Sentimento (Notícias)
                    ## ⚖️ Veredito de Operação (COMPRA FORTE/COMPRA/MANTER/VENDA/VENDA FORTE).
                    """
                    resposta = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    st.markdown(resposta.text)
                    
                    with st.spinner("Gerando arquivo PDF..."):
                        pdf_bytes = gerar_pdf_relatorio(ativo_escolhido, fechamento_atual, rsi_atual, macd_atual, resposta.text)
                        st.download_button("📥 Baixar Relatório em PDF", data=pdf_bytes, file_name=f"Relatorio_{ativo_escolhido}.pdf", mime="application/pdf", type="primary")

        except Exception as e:
            st.error(f"Erro na execução. Detalhe: {e}")
