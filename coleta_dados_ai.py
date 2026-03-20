import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
import vectorbt as vbt
from fpdf import FPDF
from google import genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Terminal Crypto IA Pro", page_icon="⚡", layout="wide")

GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
NEWS_KEY = st.secrets.get("NEWS_API_KEY", "")
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

BINANCE_FUTURES_BASE = "https://fapi.binance.com"

def validar_resposta_binance(payload):
    if isinstance(payload, dict):
        msg = payload.get("msg", "Erro desconhecido da Binance")
        code = payload.get("code", "N/A")
        raise ValueError(f"Binance retornou erro [{code}]: {msg}")
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("A Binance não retornou velas para esse ativo/timeframe.")
    return True

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

@st.cache_data(ttl=300)
def coletar_dados_binance_futuros(ticker: str, intervalo: str, limite: int = 1500) -> pd.DataFrame:
    url = f"{BINANCE_FUTURES_BASE}/fapi/v1/klines"
    params = {
        "symbol": ticker.upper().strip(),
        "interval": intervalo,
        "limit": limite
    }

    r = requests.get(url, params=params, timeout=15)
    payload = r.json()
    validar_resposta_binance(payload)

    df = pd.DataFrame(
        payload,
        columns=[
            "open_time", "Open", "High", "Low", "Close", "Volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
    )

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    if df.empty:
        raise ValueError("A Binance retornou candles, mas nenhum candle válido pôde ser processado.")

    df = calcular_indicadores(df)
    return df

def gerar_grafico_profissional(dados, nome_ativo, intervalo):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.15, 0.25]
    )

    fig.add_trace(
        go.Candlestick(
            x=dados.index,
            open=dados["Open"],
            high=dados["High"],
            low=dados["Low"],
            close=dados["Close"],
            name="Preço"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["Bollinger_Upper"],
            line=dict(color="gray", width=1, dash="dash"),
            name="Banda Sup."
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["Bollinger_Lower"],
            line=dict(color="gray", width=1, dash="dash"),
            name="Banda Inf.",
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.1)"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["SMA_20"],
            line=dict(color="blue", width=1.5),
            name="Média 20"
        ),
        row=1, col=1
    )

    if dados["SMA_200"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=dados.index,
                y=dados["SMA_200"],
                line=dict(color="purple", width=2),
                name="Média 200"
            ),
            row=1, col=1
        )

    cores_vol = ["green" if c >= o else "red" for c, o in zip(dados["Close"], dados["Open"])]
    fig.add_trace(
        go.Bar(
            x=dados.index,
            y=dados["Volume"],
            marker_color=cores_vol,
            name="Volume"
        ),
        row=2, col=1
    )

    cores_macd = ["green" if v >= 0 else "red" for v in dados["MACD_Hist"].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=dados.index,
            y=dados["MACD_Hist"],
            marker_color=cores_macd,
            name="MACD Hist"
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["MACD_Line"],
            line=dict(color="orange", width=1.5),
            name="MACD Line"
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dados.index,
            y=dados["MACD_Signal"],
            line=dict(color="cyan", width=1.5),
            name="Signal Line"
        ),
        row=3, col=1
    )

    fig.update_layout(
        title=f"Análise de Futuros - {nome_ativo} ({intervalo})",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=750,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

@st.cache_data(ttl=1800)
def buscar_noticias(ticker):
    if not NEWS_KEY:
        return [], "API Key não configurada."

    termo_busca = ticker.replace("USDT", "").replace("USD", "")
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
            texto = "".join([f"- Título: {a['title']} ({a['source']['name']})\n" for a in artigos])
            return artigos, texto if texto else "Sem manchetes relevantes."
        return [], "Nenhuma notícia relevante."
    except Exception as e:
        return [], f"Erro na coleta: {e}"

def executar_backtest_macd(dados, margem, alavancagem, taxa_corretora):
    base = dados.dropna(subset=["Close", "MACD_Line", "MACD_Signal"]).copy()

    if base.empty or len(base) < 30:
        raise ValueError("Não há velas suficientes para executar o backtest com segurança.")

    entradas = base["MACD_Line"] > base["MACD_Signal"]
    saidas = base["MACD_Line"] < base["MACD_Signal"]

    tamanho_operacao = margem * alavancagem
    caixa_virtual = tamanho_operacao * 10

    portfolio = vbt.Portfolio.from_signals(
        base["Close"],
        entries=entradas,
        exits=saidas,
        size=tamanho_operacao,
        size_type="value",
        init_cash=caixa_virtual,
        fees=taxa_corretora,
        direction="longonly"
    )

    fig_bt = portfolio.plot(subplots=["orders", "trade_pnl", "cum_returns"])
    fig_bt.update_layout(template="plotly_dark", height=600)

    lucro_total = float(portfolio.total_profit())
    total_trades = portfolio.trades.count()
    win_rate = float(portfolio.trades.win_rate() * 100) if total_trades > 0 else 0.0

    return fig_bt, lucro_total, win_rate

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

    with pdf.table(col_widths=(45, 70), text_align="LEFT") as table:
        row = table.row()
        row.cell("Preco Atual:")
        row.cell(f"US$ {fechamento:.6f}" if fechamento < 1 else f"US$ {fechamento:.2f}")
        row = table.row()
        row.cell("RSI:")
        row.cell(f"{rsi:.2f}")
        row = table.row()
        row.cell("MACD:")
        row.cell(f"{macd:.6f}" if abs(macd) < 1 else f"{macd:.2f}")

    pdf.ln(5)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Veredito da IA", 0, 1)
    pdf.ln(5)

    pdf.set_font("helvetica", "", 11)
    texto_limpo = str(texto_ia).replace("**", "").replace("*", "").replace("#", "")
    texto_limpo = texto_limpo.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, texto_limpo)

    return bytes(pdf.output())

st.title("⚡ Terminal Institucional de Futuros & IA")

st.markdown("---")
col1, col2, col3, col4 = st.columns([2, 1.5, 2, 2])

with col1:
    ativo = st.text_input("Ticker Futuros Binance", value="BTCUSDT")

with col2:
    intervalos_binance = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
    sel_intervalo = st.selectbox("Timeframe (Vela)", intervalos_binance, index=5)

with col3:
    qtd_velas = st.slider("Qtd. Velas no Gráfico", min_value=50, max_value=1500, value=200, step=50)

with col4:
    st.write("")
    btn = st.button("Analisar Mercado", type="primary", use_container_width=True)

with st.expander("⚙️ Configurações de Risco do Backtesting", expanded=False):
    r1, r2, r3 = st.columns(3)
    margem = r1.number_input("Margem/Banca por Trade (US$)", min_value=10.0, value=100.0, step=10.0)
    alavancagem = r2.number_input("Alavancagem (x)", min_value=1, max_value=125, value=10, step=1)
    taxa = r3.number_input("Taxa Corretora Maker/Taker (%)", min_value=0.0, value=0.04, step=0.01) / 100

st.markdown("---")

if btn:
    with st.spinner(f"Extraindo dados de {ativo} na Binance Futures..."):
        try:
            dados_completos = coletar_dados_binance_futuros(ativo, sel_intervalo, limite=1500)

            if dados_completos.empty:
                st.error(f"Não há velas disponíveis para {ativo} em {sel_intervalo}.")
                st.stop()

            dados_plot = dados_completos.tail(qtd_velas).copy()

            if dados_plot.empty:
                st.error("O recorte de velas ficou vazio. Reduza a quantidade de velas ou troque o timeframe.")
                st.stop()

            fechamento = float(dados_plot["Close"].dropna().iloc[-1])

            rsi_series = dados_plot["RSI_14"].dropna()
            macd_series = dados_plot["MACD_Line"].dropna()
            macd_signal_series = dados_plot["MACD_Signal"].dropna()

            rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
            macd = float(macd_series.iloc[-1]) if not macd_series.empty else 0.0
            macd_signal = float(macd_signal_series.iloc[-1]) if not macd_signal_series.empty else 0.0

            noticias_lista, noticias_texto = buscar_noticias(ativo)

            tab1, tab2, tab3 = st.tabs(["📈 Gráficos Perpétuos", "📰 Notícias & Macro", "⏳ Simulador de Lucros"])

            with tab1:
                st.caption("Fonte de dados: Binance Futures API")
                st.plotly_chart(
                    gerar_grafico_profissional(dados_plot, ativo, sel_intervalo),
                    use_container_width=True
                )

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
                c1.metric(
                    "Lucro Líquido Real",
                    f"US$ {lucro:.2f}",
                    delta="Gain" if lucro > 0 else "Loss",
                    delta_color="normal" if lucro > 0 else "inverse"
                )
                roe = (lucro / margem) * 100 if margem > 0 else 0
                c2.metric("Retorno s/ Banca (ROE)", f"{roe:.1f}%")
                c3.metric("Taxa de Acerto (Win Rate)", f"{winrate:.1f}%")

                st.plotly_chart(fig_bt, use_container_width=True)

            if usar_ia := True:
                st.markdown("---")
                st.subheader("🧠 Veredito Final (IA Institucional)")
                if client:
                    with st.spinner("IA do Fundo cruzando dados técnicos e notícias..."):
                        prompt = f"""
                        Você é um Head Trader de Cripto Futuros.
                        Ativo: {ativo}
                        Timeframe: {sel_intervalo}

                        Dados técnicos:
                        - Preço: US$ {fechamento:.6f}
                        - RSI: {rsi:.2f}
                        - MACD Line: {macd:.6f}
                        - MACD Signal: {macd_signal:.6f}

                        Notícias:
                        {noticias_texto}

                        Retorne em Markdown:
                        ## 🎯 Resumo Executivo
                        ## 📊 Ação do Preço
                        ## 📰 Sentimento de Mercado
                        ## ⚖️ Setup de Trade (LONG, SHORT ou AGUARDAR)
                        """
                        resposta = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                        st.markdown(resposta.text)

                        pdf_bytes = gerar_pdf_relatorio(ativo, fechamento, rsi, macd, resposta.text)
                        st.download_button(
                            "📥 Baixar Relatório",
                            data=pdf_bytes,
                            file_name=f"Trade_{ativo}_{sel_intervalo}.pdf",
                            mime="application/pdf",
                            type="primary"
                        )
                else:
                    st.warning("GEMINI_API_KEY não configurada.")

        except Exception as e:
            st.error(f"Erro ao processar {ativo}: {e}")
