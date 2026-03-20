import math
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
import vectorbt as vbt
import yfinance as yf
from fpdf import FPDF
from google import genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# CONFIG GERAL E CHAVES
# ==========================================
st.set_page_config(page_title="Terminal Crypto IA Pro", page_icon="⚡", layout="wide")

GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
NEWS_KEY = st.secrets.get("NEWS_API_KEY", "")
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

BINANCE_BASE = "https://api.binance.com"

# ==========================================
# FUNÇÕES DE DADOS (BINANCE + YAHOO)
# ==========================================

def _binance_interval(periodo_dias: int):
    if periodo_dias <= 1:
        return "1h", periodo_dias * 24
    elif periodo_dias <= 7:
        return "4h", math.ceil(periodo_dias * 24 / 4)
    elif periodo_dias <= 30:
        return "1d", periodo_dias
    elif periodo_dias <= 365:
        return "1d", periodo_dias
    else:
        return "1d", 365

def coletar_dados_binance(ticker: str, periodo_dias: int) -> pd.DataFrame:
    intervalo, limit = _binance_interval(periodo_dias)
    params = {"symbol": ticker.upper(), "interval": intervalo, "limit": limit}
    url = f"{BINANCE_BASE}/api/v3/klines"
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    klines = r.json()
    if not klines:
        raise ValueError("Sem dados retornados pela Binance")

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time","Open","High","Low","Close","Volume",
            "close_time","qav","num_trades","taker_base_vol",
            "taker_quote_vol","ignore",
        ],
    )
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    df.index.name = "Date"
    df = df[["Open","High","Low","Close","Volume"]].dropna()

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

    return df.dropna()

def coletar_dados_yahoo(ticker: str, periodo_dias: int) -> pd.DataFrame:
    if periodo_dias <= 7:
        period_str = "7d"
    elif periodo_dias <= 30:
        period_str = "1mo"
    elif periodo_dias <= 90:
        period_str = "3mo"
    elif periodo_dias <= 180:
        period_str = "6mo"
    elif periodo_dias <= 365:
        period_str = "1y"
    else:
        period_str = "5y"

    dados = yf.download(ticker, period=period_str, progress=False)
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

    df = dados.rename(
        columns={
            col_close: "Close",
            col_high: "High",
            col_low: "Low",
            col_open: "Open",
            col_vol: "Volume",
        }
    )[["Open","High","Low","Close","Volume"]]

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

    return df.dropna()

@st.cache_data(ttl=3600)
def coletar_dados_historicos(ticker: str, periodo_dias: int = 365) -> pd.DataFrame:
    try:
        df = coletar_dados_binance(ticker, periodo_dias)
        fonte = "Binance (spot)"
    except Exception:
        df = coletar_dados_yahoo(ticker, periodo_dias)
        fonte = "Yahoo Finance"
    st.session_state["fonte_dados"] = fonte
    return df

# ==========================================
# GRÁFICO PRINCIPAL
# ==========================================
def gerar_grafico_profissional(dados, nome_ativo):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.15, 0.25],
    )
    fig.add_trace(
        go.Candlestick(
            x=dados.index,
            open=dados["Open"],
            high=dados["High"],
            low=dados["Low"],
            close=dados["Close"],
            name="Preço",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["Bollinger_Upper"],
            line=dict(color="gray", width=1, dash="dash"),
            name="Banda Sup.",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["Bollinger_Lower"],
            line=dict(color="gray", width=1, dash="dash"),
            name="Banda Inf.",
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.1)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["SMA_20"],
            line=dict(color="blue", width=1.5),
            name="Média 20d",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["SMA_200"],
            line=dict(color="purple", width=2),
            name="Média 200d",
        ),
        row=1,
        col=1,
    )

    cores_vol = [
        "green" if row["Close"] >= row["Open"] else "red"
        for _, row in dados.iterrows()
    ]
    fig.add_trace(
        go.Bar(
            x=dados.index,
            y=dados["Volume"],
            marker_color=cores_vol,
            name="Volume",
        ),
        row=2,
        col=1,
    )

    cores_macd = ["green" if v >= 0 else "red" for v in dados["MACD_Hist"]]
    fig.add_trace(
        go.Bar(
            x=dados.index,
            y=dados["MACD_Hist"],
            marker_color=cores_macd,
            name="MACD Hist",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["MACD_Line"],
            line=dict(color="orange", width=1.5),
            name="MACD Line",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["MACD_Signal"],
            line=dict(color="cyan", width=1.5),
            name="Signal Line",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title=f"Análise Técnica - {nome_ativo}",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=750,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

# ==========================================
# NOTÍCIAS
# ==========================================
@st.cache_data(ttl=1800)
def buscar_noticias(ticker):
    if not NEWS_KEY:
        return [], "API Key de notícias não configurada."
    termo_busca = ticker.split("-")[0]
    data_ontem = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={termo_busca}&from={data_ontem}&sortBy=relevancy&language=en&apiKey={NEWS_KEY}"
    )
    try:
        r = requests.get(url, timeout=10)
        dados = r.json()
        if dados.get("status") == "ok":
            artigos = dados.get("articles", [])[:5]
            texto = "".join(
                [
                    f"- Título: {a['title']} (Fonte: {a['source']['name']})\n"
                    for a in artigos
                ]
            )
            return artigos, texto if texto else "Sem manchetes relevantes."
        return [], "Nenhuma notícia relevante encontrada."
    except Exception as e:
        return [], f"Erro na coleta: {e}"

# ==========================================
# BACKTEST COM ALAVANCAGEM
# ==========================================
def executar_backtest_macd(dados, margem, alavancagem, taxa_corretora):
    entradas = dados["MACD_Line"] > dados["MACD_Signal"]
    saidas = dados["MACD_Line"] < dados["MACD_Signal"]

    tamanho_operacao = margem * alavancagem
    caixa_virtual = tamanho_operacao * 10

    portfolio = vbt.Portfolio.from_signals(
        dados["Close"],
        entradas,
        saidas,
        size=tamanho_operacao,
        size_type="value",
        init_cash=caixa_virtual,
        fees=taxa_corretora,
    )

    fig_bt = portfolio.plot(subplots=["orders", "trade_pnl", "cum_returns"])
    fig_bt.update_layout(
        template="plotly_dark",
        height=600,
        title=f"Backtest MACD (Operando US$ {tamanho_operacao:.2f} por sinal)",
    )

    lucro_total = portfolio.total_profit()
    win_rate = portfolio.trades.win_rate() * 100 if len(portfolio.trades) > 0 else 0
    return fig_bt, lucro_total, win_rate

# ==========================================
# PDF
# ==========================================
def gerar_pdf_relatorio(ticker, fechamento, rsi, macd, texto_ia):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, f"Relatorio Oficial do Fundo: {ticker}", 0, 1, "C")
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Indicadores Tecnicos", 0, 1)
    pdf.set_font("helvetica", "", 10)
    with pdf.table(col_widths=(40, 60), text_align="LEFT") as table:
        row = table.row()
        row.cell("Preco Atual:")
        row.cell(f"US$ {fechamento:.2f}")
        row = table.row()
        row.cell("RSI (14d):")
        row.cell(f"{rsi:.2f}")
        row = table.row()
        row.cell("MACD Line:")
        row.cell(f"{macd:.2f}")

    pdf.ln(5)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Veredito da IA (Grafico + Noticias)", 0, 1)
    pdf.ln(5)

    pdf.set_font("helvetica", "", 11)
    texto_limpo = str(texto_ia).replace("**", "").replace("*", "").replace("#", "")
    texto_limpo = texto_limpo.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, texto_limpo)

    return bytes(pdf.output())

# ==========================================
# INTERFACE STREAMLIT
# ==========================================
st.title("⚡ Terminal Quantitativo & Algorítmico (Cripto Nativo)")

st.markdown("---")
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
with col1:
    ativo = st.text_input("Ticker do Ativo", value="BTCUSDT")  # Binance style
with col2:
    opcoes_periodo = {
        "1 Mês": 30,
        "3 Meses": 90,
        "6 Meses": 180,
        "1 Ano": 365,
        "2 Anos": 730,
        "5 Anos": 1825,
        "Máximo (limite)": 999,
    }
    sel_periodo = st.selectbox(
        "Período de Visualização", list(opcoes_periodo.keys()), index=2
    )
    dias_plot = opcoes_periodo[sel_periodo]
with col3:
    st.write("")
    usar_ia = st.toggle("🤖 Usar IA Híbrida", value=True)
with col4:
    st.write("")
    btn = st.button("Analisar & Backtest", type="primary", use_container_width=True)

with st.expander("⚙️ Configurações de Risco do Backtesting", expanded=False):
    r1, r2, r3 = st.columns(3)
    margem = r1.number_input(
        "Margem/Banca por Trade (US$)", min_value=10.0, value=100.0, step=10.0
    )
    alavancagem = r2.number_input(
        "Alavancagem (x)", min_value=1, max_value=125, value=1, step=1
    )
    taxa = r3.number_input(
        "Taxa da Corretora (%)", min_value=0.0, value=0.1, step=0.05
    ) / 100

st.markdown("---")

if btn:
    with st.spinner("Buscando dados e processando indicadores..."):
        try:
            dados_completos = coletar_dados_historicos(ativo, periodo_dias=dias_plot)
            dados_plot = dados_completos.tail(dias_plot)

            fechamento = dados_plot["Close"].iloc[-1]
            rsi = dados_plot["RSI_14"].iloc[-1]
            macd = dados_plot["MACD_Line"].iloc[-1]
            macd_signal = dados_plot["MACD_Signal"].iloc[-1]

            noticias_lista, noticias_texto = buscar_noticias(ativo)

            tab1, tab2, tab3 = st.tabs(
                ["📈 Gráficos", "📰 Notícias Globais", "⏳ Backtest MACD"]
            )

            with tab1:
                fonte = st.session_state.get("fonte_dados", "Desconhecida")
                st.caption(f"Fonte de dados usada: {fonte}")
                st.plotly_chart(
                    gerar_grafico_profissional(dados_plot, ativo),
                    use_container_width=True,
                )

            with tab2:
                if noticias_lista:
                    for art in noticias_lista:
                        with st.container(border=True):
                            st.markdown(f"**{art['title']}**")
                            st.caption(
                                f"{art['source']['name']} | Data: {art['publishedAt'][:10]}"
                            )
                else:
                    st.info(noticias_texto)

            with tab3:
                pos_total = margem * alavancagem
                st.markdown(
                    f"### Resultado da Estratégia MACD (Operando US$ {pos_total:.2f})"
                )
                st.write(
                    f"Margem: US$ {margem:.2f} | Alavancagem: {alavancagem}x | Taxa: {(taxa*100):.2f}%"
                )
                fig_bt, lucro, winrate = executar_backtest_macd(
                    dados_plot, margem, alavancagem, taxa
                )
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Lucro Líquido Simulado",
                    f"US$ {lucro:.2f}",
                    delta="Gain" if lucro > 0 else "Loss",
                    delta_color="normal" if lucro > 0 else "inverse",
                )
                roe = (lucro / margem) * 100 if margem > 0 else 0
                c2.metric("Retorno sobre a Margem (ROE)", f"{roe:.1f}%")
                c3.metric("Win Rate", f"{winrate:.1f}%")
                st.plotly_chart(fig_bt, use_container_width=True)

            if usar_ia and client:
                st.markdown("---")
                st.subheader("🧠 Veredito Final (Gráfico + Notícias)")
                with st.spinner("IA do Fundo analisando..."):
                    prompt = f"""
                    Atue como Head Trader de um Fundo Hedge focado em criptomoedas.
                    Ativo: {ativo}
                    Dados técnicos atuais:
                    - Preço: US$ {fechamento:.2f}
                    - RSI (14d): {rsi:.2f}
                    - MACD Line: {macd:.2f} | Signal: {macd_signal:.2f}

                    Manchetes recentes:
                    {noticias_texto}

                    Gere um relatório em Markdown com:
                    ## 🎯 Resumo Executivo
                    ## 📊 Leitura Gráfica
                    ## 📰 Termômetro de Sentimento
                    ## ⚖️ Veredito de Operação (COMPRA FORTE/COMPRA/MANTER/VENDA/VENDA FORTE).
                    """
                    resposta = client.models.generate_content(
                        model="gemini-2.5-flash", contents=prompt
                    )
                    st.markdown(resposta.text)

                    with st.spinner("Gerando PDF..."):
                        pdf_bytes = gerar_pdf_relatorio(
                            ativo, fechamento, rsi, macd, resposta.text
                        )
                        st.download_button(
                            "📥 Baixar Relatório em PDF",
                            data=pdf_bytes,
                            file_name=f"Relatorio_{ativo}.pdf",
                            mime="application/pdf",
                            type="primary",
                        )

        except Exception as e:
            st.error(f"Erro na execução. Detalhe: {e}")
