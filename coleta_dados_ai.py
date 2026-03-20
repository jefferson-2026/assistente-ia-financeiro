import os
import math
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
import vectorbt as vbt
from fpdf import FPDF
from google import genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# CONFIG GERAL E CHAVES
# ==========================================
st.set_page_config(page_title="Terminal Crypto Pro", page_icon="⚡", layout="wide")

GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
NEWS_KEY = st.secrets.get("NEWS_API_KEY", "")
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BYBIT_FUTURES_BASE = "https://api.bybit.com"

# ==========================================
# MOTOR DE INDICADORES
# ==========================================
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["Bollinger_Upper"] = df["SMA_20"] + (df["Std_Dev"] * 2)
    df["Bollinger_Lower"] = df["SMA_20"] - (df["Std_Dev"] * 2)

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD_Line"] - df["MACD_Signal"]

    delta = df["Close"].diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = ganho / perda
    df["RSI_14"] = 100 - (100 / (1 + rs))

    return df

# ==========================================
# COLETAS DE DADOS
# ==========================================
def coletar_dados_binance_futuros(ticker: str, intervalo: str) -> pd.DataFrame:
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/klines"
    params = {"symbol": ticker.upper().strip(), "interval": intervalo, "limit": 1500}
    
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    payload = r.json()
    
    if isinstance(payload, dict) and "code" in payload:
        raise ValueError(f"Binance recusou: {payload.get('msg')}")
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("Binance não tem dados para este ativo/timeframe.")

    df = pd.DataFrame(payload, columns=["open_time", "Open", "High", "Low", "Close", "Volume", "close_time", "qav", "num_trades", "tbb", "tbq", "ignore"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    return calcular_indicadores(df)

def coletar_dados_bybit_futuros(ticker: str, intervalo: str) -> pd.DataFrame:
    mapa_bybit = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "6h": "360", "8h": "480",
        "12h": "720", "1d": "D", "3d": "D", "1w": "W", "1M": "M"
    }
    intervalo_bybit = mapa_bybit.get(intervalo, "60")
    
    url = f"{BYBIT_FUTURES_BASE}/v5/market/kline"
    params = {"category": "linear", "symbol": ticker.upper().strip(), "interval": intervalo_bybit, "limit": 1000}
    
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    payload = r.json()
    
    if payload.get("retCode") != 0:
        raise ValueError(f"Bybit recusou: {payload.get('retMsg')}")
        
    lista_velas = payload.get("result", {}).get("list", [])
    if not lista_velas:
        raise ValueError("Bybit não tem dados para este ativo/timeframe.")
        
    lista_velas = lista_velas[::-1]
    
    df = pd.DataFrame(lista_velas, columns=["startTime", "Open", "High", "Low", "Close", "Volume", "turnover"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df["open_time"] = pd.to_datetime(df["startTime"], unit="ms")
    df = df.set_index("open_time")
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    return calcular_indicadores(df)

@st.cache_data(ttl=300)
def orquestrador_de_dados(ticker: str, intervalo: str) -> pd.DataFrame:
    erros = []
    try:
        df = coletar_dados_binance_futuros(ticker, intervalo)
        st.session_state["fonte_dados"] = "Binance Futures (Oficial)"
        return df
    except Exception as e_bin:
        erros.append(f"Binance: {str(e_bin)}")
        
    try:
        df = coletar_dados_bybit_futuros(ticker, intervalo)
        st.session_state["fonte_dados"] = "Bybit Linear Futures (Fallback)"
        return df
    except Exception as e_byb:
        erros.append(f"Bybit: {str(e_byb)}")
        
    raise ValueError(" | ".join(erros))

# ==========================================
# RESTANTE DO SISTEMA (Gráficos, News, Backtest, PDF)
# ==========================================
def formatar_moeda(valor):
    """Detecta se a moeda é muito barata e ajusta as casas decimais."""
    if valor < 0.01:
        return f"{valor:.6f}"
    elif valor < 1:
        return f"{valor:.4f}"
    else:
        return f"{valor:.2f}"

def gerar_grafico_profissional(dados, nome_ativo, intervalo):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.15, 0.25])

    fig.add_trace(go.Candlestick(x=dados.index, open=dados["Open"], high=dados["High"], low=dados["Low"], close=dados["Close"], name="Preço"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados["Bollinger_Upper"], line=dict(color="gray", width=1, dash="dash"), name="Banda Sup."), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados["Bollinger_Lower"], line=dict(color="gray", width=1, dash="dash"), name="Banda Inf.", fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados["SMA_20"], line=dict(color="blue", width=1.5), name="Média 20"), row=1, col=1)

    if dados["SMA_200"].notna().any():
        fig.add_trace(go.Scatter(x=dados.index, y=dados["SMA_200"], line=dict(color="purple", width=2), name="Média 200"), row=1, col=1)

    cores_vol = ["green" if c >= o else "red" for c, o in zip(dados["Close"], dados["Open"])]
    fig.add_trace(go.Bar(x=dados.index, y=dados["Volume"], marker_color=cores_vol, name="Volume"), row=2, col=1)

    cores_macd = ["green" if v >= 0 else "red" for v in dados["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=dados.index, y=dados["MACD_Hist"], marker_color=cores_macd, name="MACD Hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados["MACD_Line"], line=dict(color="orange", width=1.5), name="MACD Line"), row=3, col=1)
    fig.add_trace(go.Scatter(x=dados.index, y=dados["MACD_Signal"], line=dict(color="cyan", width=1.5), name="Signal Line"), row=3, col=1)

    fig.update_layout(title=f"Ação do Preço - {nome_ativo} ({intervalo})", template="plotly_dark", xaxis_rangeslider_visible=False, height=750, margin=dict(l=0, r=0, t=40, b=0))
    return fig

@st.cache_data(ttl=1800)
def buscar_noticias(ticker):
    if not NEWS_KEY: return [], "API Key não configurada."
    termo_busca = ticker.upper().replace("1000", "").replace("USDT", "").replace("USD", "")
    data_ontem = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    
    query = f"({termo_busca} crypto) OR (Federal Reserve rate)"
    url = f"https://newsapi.org/v2/everything?q={query}&from={data_ontem}&sortBy=relevancy&language=en&apiKey={NEWS_KEY}"
    
    try:
        r = requests.get(url, timeout=10)
        dados = r.json()
        if dados.get("status") == "ok":
            artigos = dados.get("articles", [])[:5]
            texto = "".join([f"- Título: {a['title']} ({a['source']['name']})\n" for a in artigos])
            return artigos, texto if texto else "Sem manchetes relevantes no momento."
        return [], "Nenhuma notícia relevante."
    except Exception as e:
        return [], f"Erro na coleta: {e}"

def executar_backtest_macd(dados, margem, alavancagem, taxa_corretora):
    base = dados.dropna(subset=["Close", "MACD_Line", "MACD_Signal"]).copy()
    if base.empty or len(base) < 10:
        raise ValueError("Poucas velas para backtest seguro.")

    entradas = base["MACD_Line"] > base["MACD_Signal"]
    saidas = base["MACD_Line"] < base["MACD_Signal"]
    tamanho_operacao = margem * alavancagem
    caixa_virtual = tamanho_operacao * 5

    portfolio = vbt.Portfolio.from_signals(
        base["Close"], entries=entradas, exits=saidas,
        size=tamanho_operacao, size_type="value", init_cash=caixa_virtual,
        fees=taxa_corretora, direction="longonly"
    )

    fig_bt = portfolio.plot(subplots=["orders", "trade_pnl", "cum_returns"])
    fig_bt.update_layout(template="plotly_dark", height=600)
    
    lucro = float(portfolio.total_profit())
    win_rate = float(portfolio.trades.win_rate() * 100) if portfolio.trades.count() > 0 else 0.0
    return fig_bt, lucro, win_rate

def gerar_pdf_relatorio(ticker, fechamento, max_preco, min_preco, rsi, texto_ia):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, f"Relatorio Oficial de Operacao: {ticker}", 0, 1, "C")
    pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Raio-X de Preco e Indicadores", 0, 1)
    pdf.set_font("helvetica", "", 10)
    
    with pdf.table(col_widths=(45, 70), text_align="LEFT") as table:
        row = table.row(); row.cell("Preco Atual:"); row.cell(f"US$ {formatar_moeda(fechamento)}")
        row = table.row(); row.cell("Maxima (Periodo):"); row.cell(f"US$ {formatar_moeda(max_preco)}")
        row = table.row(); row.cell("Minima (Periodo):"); row.cell(f"US$ {formatar_moeda(min_preco)}")
        row = table.row(); row.cell("RSI 14:"); row.cell(f"{rsi:.2f}")
        
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Veredito Institucional da Inteligencia Artificial", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("helvetica", "", 11)
    texto_limpo = str(texto_ia).replace("**", "").replace("*", "").replace("#", "")
    texto_limpo = texto_limpo.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, texto_limpo)
    return bytes(pdf.output())

# ==========================================
# FRONTEND - INTERFACE PRINCIPAL
# ==========================================
st.title("⚡ Terminal Institucional de Futuros")

st.markdown("---")
# A linha principal de inputs: agora com o Toggle da IA!
col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1.5, 1, 1.2])
with col1:
    ativo = st.text_input("Ticker Futuros", value="BTCUSDT")
with col2:
    intervalos = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    sel_intervalo = st.selectbox("Timeframe", intervalos, index=4)
with col3:
    qtd_velas = st.slider("Qtd. Velas no Gráfico", min_value=50, max_value=1000, value=200, step=50)
with col4:
    st.write("")
    st.write("")
    # NOVO: CHAVE PARA LIGAR/DESLIGAR A IA E POUPAR TOKENS
    ativar_ia = st.toggle("🧠 Usar IA", value=False) 
with col5:
    st.write("")
    btn = st.button("Analisar", type="primary", use_container_width=True)

with st.expander("⚙️ Configurações de Risco do Backtesting", expanded=False):
    r1, r2, r3 = st.columns(3)
    margem = r1.number_input("Sua Margem por Trade (US$)", min_value=10.0, value=100.0, step=10.0)
    alavancagem = r2.number_input("Alavancagem (x)", min_value=1, max_value=125, value=10, step=1)
    taxa = r3.number_input("Taxa Corretora Maker/Taker (%)", min_value=0.0, value=0.04, step=0.01) / 100

st.markdown("---")

if btn:
    with st.spinner(f"Extraindo dados via API Institucional para {ativo}..."):
        try:
            # Chama Orquestrador
            dados_completos = orquestrador_de_dados(ativo, sel_intervalo)

            if dados_completos.empty:
                st.error(f"Não há dados disponíveis.")
                st.stop()

            # Corta para o tamanho do slider
            dados_plot = dados_completos.tail(qtd_velas).copy()
            if dados_plot.empty:
                st.error("Sem dados no recorte.")
                st.stop()

            # CÁLCULOS NOVOS DE MÁXIMAS, MÍNIMAS E VARIAÇÃO
            fechamento_atual = float(dados_plot["Close"].dropna().iloc[-1])
            abertura_inicial = float(dados_plot["Open"].dropna().iloc[0])
            
            # Pega o maior 'High' e menor 'Low' de TODAS as velas que estão na tela
            maxima_periodo = float(dados_plot["High"].max())
            minima_periodo = float(dados_plot["Low"].min())
            
            # Variação em % do começo do gráfico até agora
            variacao_perc = ((fechamento_atual - abertura_inicial) / abertura_inicial) * 100

            # Indicadores clássicos
            rsi_series = dados_plot["RSI_14"].dropna()
            macd_series = dados_plot["MACD_Line"].dropna()
            macd_signal_series = dados_plot["MACD_Signal"].dropna()
            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
            macd = float(macd_series.iloc[-1]) if not macd_series.empty else 0.0
            macd_signal = float(macd_signal_series.iloc[-1]) if not macd_signal_series.empty else 0.0

            # Puxa notícias (sempre rodamos para o backtest, mas a IA só usa se ligada)
            noticias_lista, noticias_texto = buscar_noticias(ativo)

            # --- NOVO PAINEL DE MÉTRICAS (MÁXIMA E MÍNIMA) NO TOPO ---
            st.success(f"Fonte de Dados: **{st.session_state.get('fonte_dados')}**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Preço Atual", f"US$ {formatar_moeda(fechamento_atual)}", f"{variacao_perc:.2f}%", delta_color="normal")
            m2.metric("Máxima do Período", f"US$ {formatar_moeda(maxima_periodo)}", "Topo Resistência", delta_color="off")
            m3.metric("Mínima do Período", f"US$ {formatar_moeda(minima_periodo)}", "Fundo Suporte", delta_color="off")
            m4.metric("Tendência MACD", "Compradora (Bullish)" if macd > macd_signal else "Vendedora (Bearish)")

            st.markdown("<br>", unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["📈 Gráficos", "📰 Notícias & Macro", "⏳ Backtest Strategy"])

            with tab1:
                st.plotly_chart(gerar_grafico_profissional(dados_plot, ativo, sel_intervalo), use_container_width=True)

            with tab2:
                if noticias_lista:
                    for art in noticias_lista:
                        with st.container(border=True):
                            st.markdown(f"**{art['title']}**")
                            st.caption(f"{art['source']['name']} | Data: {art['publishedAt'][:10]}")
                else:
                    st.info(noticias_texto)

            with tab3:
                pos_total = margem * alavancagem
                st.markdown(f"### Backtest: Operando US$ {pos_total:.2f} nas últimas {qtd_velas} velas ({sel_intervalo})")

                fig_bt, lucro, winrate = executar_backtest_macd(dados_plot, margem, alavancagem, taxa)

                c1, c2, c3 = st.columns(3)
                c1.metric("Lucro Líquido Real", f"US$ {lucro:.2f}", delta="Gain" if lucro > 0 else "Loss", delta_color="normal" if lucro > 0 else "inverse")
                roe = (lucro / margem) * 100 if margem > 0 else 0
                c2.metric("Retorno s/ Margem (ROE)", f"{roe:.1f}%")
                c3.metric("Taxa de Acerto (Win Rate)", f"{winrate:.1f}%")

                st.plotly_chart(fig_bt, use_container_width=True)

            # --- SÓ RODA A IA SE O USUÁRIO ATIVOU O BOTÃO DE TOGGLE ---
            if ativar_ia and client:
                st.markdown("---")
                st.subheader("🧠 Relatório do Head Trader (IA)")
                with st.spinner("IA processando dados em nuvem e redigindo relatório..."):
                    prompt = f"""
                    Atue como Head Trader de Cripto Futuros. Ativo: {ativo} | Timeframe: {sel_intervalo}.
                    Preço atual: US$ {formatar_moeda(fechamento_atual)}. Máxima do período: US$ {formatar_moeda(maxima_periodo)}. Mínima: US$ {formatar_moeda(minima_periodo)}.
                    RSI: {rsi:.2f} | MACD: {macd:.6f}
                    Contexto Macro: {noticias_texto}

                    Retorne em Markdown limpo:
                    ## 🎯 Resumo Executivo
                    ## 📊 Análise Gráfica e Pontos de Controle
                    ## 📰 Contexto Macro / Notícias
                    ## ⚖️ Setup de Trade (LONG, SHORT ou AGUARDAR e justifique o Risco)
                    """
                    resposta = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    st.markdown(resposta.text)

                    with st.spinner("Gerando Relatório PDF..."):
                        pdf_bytes = gerar_pdf_relatorio(ativo, fechamento_atual, maxima_periodo, minima_periodo, rsi, resposta.text)
                        st.download_button("📥 Baixar PDF do Relatório", data=pdf_bytes, file_name=f"Trade_{ativo}.pdf", mime="application/pdf", type="primary")
            elif not ativar_ia:
                st.info("A Inteligência Artificial não foi ativada nesta busca para economizar tokens. Se quiser o relatório completo, ligue o Toggle '🧠 Usar IA' lá em cima e clique em Analisar.")

        except Exception as e:
            st.error(f"Erro na obtenção de dados:\n{e}")
