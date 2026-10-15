import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
import concurrent.futures
import datetime
import numpy as np
import time
import requests

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Intraday Quant & News Dashboard", layout="wide", page_icon="⚡")

# ==========================================
# 1. MODUL BERITA REAL-TIME (MASSIVE RSS AGGREGATOR)
# ==========================================
@st.cache_data(ttl=300) # Update setiap 5 menit
def fetch_all_news():
    rss_feeds = {
        "CNBC Market": "https://www.cnbcindonesia.com/market/rss",
        "CNBC Rekomendasi": "https://www.cnbcindonesia.com/mymoney/rss",
        "Kontan Investasi": "https://rss.kontan.co.id/news/investasi",
        "Kontan Industri": "https://rss.kontan.co.id/news/industri",
        "Kontan Keuangan": "https://rss.kontan.co.id/news/keuangan",
        "Bisnis Market": "https://www.bisnis.com/rss/market",
        "Antara Ekonomi": "https://www.antaranews.com/rss/ekonomi.xml",
        "Sindonews Ekbis": "https://ekbis.sindonews.com/rss",
        "Detik Finance": "https://rss.detik.com/index.php/finance",
        "Republika": "https://www.republika.co.id/rss/ekonomi",
        "IDX Channel": "https://www.idxchannel.com/rss",
        "EmitenNews": "https://emitennews.com/feed",
        "SahamGO": "https://sahamgo.com/feed/",
        "SahamOK": "https://www.sahamok.net/feed/",
        "PintarSaham": "https://pintarsaham.id/feed/",
        "Stockwise": "https://stockwise.id/feed/",
        "Emtrade": "https://emtrade.id/feed/",
        "CreativeTrader": "https://www.creativetrader.id/feed/",
        "Bandarmology": "https://www.bandarmology.com/feed/",
        "Google IDX": "https://news.google.com/rss/search?q=saham+BEI+OR+IHSG+when:1d&hl=id&gl=ID&ceid=ID:id",
        "Google Rekomendasi": "https://news.google.com/rss/search?q=(site:cnbcindonesia.com+OR+site:kontan.co.id+OR+site:market.bisnis.com)+(\"rekomendasi+saham\"+OR+\"target+harga\")+when:1d&hl=id&gl=ID&ceid=ID:id",
        "Google UMA/Suspensi": "https://news.google.com/rss/search?q=\"Unusual+Market+Activity\"+OR+\"Suspensi+Saham\"+BEI+when:1d&hl=id&gl=ID&ceid=ID:id",
        "Google Bloomberg Tech": "https://news.google.com/rss/search?q=site:bloombergtechnoz.com+(saham+OR+IHSG+OR+emiten)+when:1d&hl=id&gl=ID&ceid=ID:id",
        "Google IDN/Investor": "https://news.google.com/rss/search?q=site:idnfinancials.com+OR+site:investor.id+when:1d&hl=id&gl=ID&ceid=ID:id"
    }
    
    news_data = []
    
    def parse_feed(source, url):
        try:
            response = requests.get(url, timeout=5)
            feed = feedparser.parse(response.content)
            limit = 10 if "Google" in source else 5
            for entry in feed.entries[:limit]: 
                published_time = entry.published if hasattr(entry, 'published') else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                news_data.append({
                    "Waktu": published_time,
                    "Kategori": source,
                    "Judul Berita": entry.title,
                    "Tautan": entry.link
                })
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(parse_feed, source, url) for source, url in rss_feeds.items()]
        concurrent.futures.wait(futures, timeout=12)
        
    df_news = pd.DataFrame(news_data)
    if not df_news.empty:
        df_news['Waktu'] = pd.to_datetime(df_news['Waktu'], utc=True, errors='coerce')
        df_news = df_news.dropna(subset=['Waktu']).sort_values(by='Waktu', ascending=False)
        df_news['Waktu'] = df_news['Waktu'].dt.tz_convert('Asia/Jakarta').dt.strftime("%d %b %Y %H:%M")
        df_news = df_news.drop_duplicates(subset=['Judul Berita'], keep='first')
    return df_news

# ==========================================
# 2. MODUL NLP: SENTIMEN BERITA MENDALAM
# ==========================================
def get_news_catalyst(ticker, df_news):
    if df_news is None or df_news.empty:
        return "-", "⚪ Netral", 0
        
    clean_ticker = ticker.replace(".JK", "").strip()
    pattern = fr'\b{clean_ticker}\b'
    related_news = df_news[df_news['Judul Berita'].str.contains(pattern, case=False, regex=True, na=False)]
    
    if related_news.empty:
        return "-", "⚪ Netral", 0
        
    bullish_kw = r'(?i)laba|dividen|akuisisi|rekomendasi|buyback|naik|melonjak|meroket|cuan|ekspansi|kontrak|prospek|terbang|lonjak|rekor|akumulasi asing|net buy|target harga'
    bearish_kw = r'(?i)rugi|anjlok|suspensi|uma|delisting|fca|turun|merosot|tekanan|pkpu|pantauan khusus|hindari|gagal bayar|jeblok|ARBs|net sell|distribusi|buka gembok'
    
    score = 0
    for title in related_news['Judul Berita']:
        if pd.Series([title]).str.contains(bullish_kw, regex=True).any():
            score += 1
        if pd.Series([title]).str.contains(bearish_kw, regex=True).any():
            score -= 1
            
    latest_title = related_news.iloc[0]['Judul Berita']
    
    if score > 0:
        sentiment = "🟢 Positif"
    elif score < 0:
        sentiment = "🔴 Negatif / Bahaya"
    else:
        sentiment = "🟡 Sentimen Campuran"
        
    return latest_title, sentiment, score

# ==========================================
# 3. MODUL TECHNICAL ANALYSIS (Berdasarkan "Mastering the Trade")
# ==========================================
def calculate_quant_indicators(df):
    if len(df) < 40:
        return df
        
    # VWAP
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI (14) - Klasik
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # RSI (7) - John Carter's Bearish Divergence Rule
    ema_up_7 = up.ewm(com=6, adjust=False).mean()
    ema_down_7 = down.ewm(com=6, adjust=False).mean()
    rs_7 = ema_up_7 / ema_down_7
    df['RSI_7'] = 100 - (100 / (1 + rs_7))
    
    # ATR & TTM Squeeze
    df['TR'] = np.maximum((df['High'] - df['Low']), 
                          np.maximum(abs(df['High'] - df['Close'].shift()), 
                                     abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = df['TR'].rolling(window=14).mean()

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['MA20'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['MA20'] - (df['BB_std'] * 2)
    
    df['KC_upper'] = df['MA20'] + (df['ATR'] * 1.5)
    df['KC_lower'] = df['MA20'] - (df['ATR'] * 1.5)
    
    df['Squeeze_On'] = (df['BB_lower'] > df['KC_lower']) & (df['BB_upper'] < df['KC_upper'])
    
    return df

@st.cache_data(ttl=900)
def run_quant_screener(scan_limit, df_news):
    try:
        emiten_df = pd.read_csv("daftar_lengkap_saham_indonesia_bei.csv")
        tickers = emiten_df['Ticker_Yahoo_Finance'].dropna().unique().tolist()
    except Exception as e:
        st.error(f"Gagal memuat file CSV: {e}")
        return pd.DataFrame()

    tickers_to_scan = tickers[:scan_limit] 
    results = []
    
    def process_ticker(ticker):
        try:
            data = yf.download(ticker, period="10d", interval="15m", progress=False)
            if len(data) < 40:
                return None
                
            df = calculate_quant_indicators(data)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # =========================================================
            # STRATEGI 1: CARTER'S BEARISH DIVERGENCE (FAKE ORGASM SETUP)
            # Harga Higher High, RSI(7) Lower High, Volume Menurun
            # =========================================================
            higher_high = latest['High'] > prev['High']
            rsi_lower = latest['RSI_7'] < prev['RSI_7']
            volume_drop = latest['Volume'] < prev['Volume']
            carter_bearish_div = higher_high and rsi_lower and volume_drop
            
            # =========================================================
            # STRATEGI 2: TTM SQUEEZE FIRING & BUILDING
            # =========================================================
            squeeze_firing = (prev['Squeeze_On'] == True) and (latest['Squeeze_On'] == False) and (latest['MACD_Hist'] > 0)
            squeeze_building = (latest['Squeeze_On'] == True) and (float(latest['MACD_Hist']) > float(prev['MACD_Hist']))
            
            # =========================================================
            # STRATEGI 3: STRICT OPENING GAP
            # =========================================================
            gap_up_strict = latest['Open'] > prev['Close'] * 1.015
            volume_spike_strict = latest['Volume'] > df['Volume'].rolling(20).mean().iloc[-1] * 2
            
            # Filter Keamanan Dasar
            above_vwap = latest['Close'] > latest['VWAP']
            rsi_safe = latest['RSI_14'] < 70

            tech_signal = "HOLD / SKIP"
            logic_used = "-"
            
            # Logika Deteksi
            if carter_bearish_div:
                tech_signal = "🛑 STRONG SELL"
                logic_used = "Carter's Bearish Divergence (Take Profit!)"
            elif gap_up_strict and volume_spike_strict and above_vwap and rsi_safe:
                tech_signal = "STRONG BUY"
                logic_used = "Strict Gap Breakout"
            elif squeeze_firing and volume_spike_strict and above_vwap:
                tech_signal = "BUY"
                logic_used = "Carter's TTM Squeeze Firing"
            elif squeeze_building and above_vwap:
                tech_signal = "👀 RADAR"
                logic_used = "Squeeze Building Up"
            elif latest['RSI_14'] >= 80 or latest['Close'] < latest['VWAP'] * 0.99:
                tech_signal = "SELL / CUT"
                logic_used = "Overbought / Breakdown VWAP"

            # --- INTEGRASI SENTIMEN BERITA ---
            news_title, sentiment_label, news_score = get_news_catalyst(ticker, df_news)
            
            final_signal = tech_signal
            
            # Matriks Sinergi dengan Bobot Lebih Pintar
            if tech_signal == "🛑 STRONG SELL":
                final_signal = "🛑 STRONG SELL / SHORT (Technical Breakdown)"
            elif tech_signal in ["STRONG BUY", "BUY"] and news_score > 0:
                final_signal = "🔥 SUPER BUY (Katalis + Teknikal Valid)"
            elif tech_signal in ["STRONG BUY", "BUY"] and news_score < 0:
                final_signal = "⚠️ BUY BERISIKO (Peringatan UMA/News)"
            elif tech_signal == "👀 RADAR" and news_score > 0:
                final_signal = "🎯 RADAR PRIORITAS (Konsolidasi + Sentimen Positif)"
            elif tech_signal == "HOLD / SKIP" and news_score > 1:
                 final_signal = "🚨 PANTAU NEWS (Technical Sideways, Fundamental Hot)"

            # Tampilkan saham yang masuk kriteria aksi
            if final_signal not in ["HOLD / SKIP", "SELL / CUT"]:
                atr = float(latest['ATR'])
                entry_price = float(latest['Close'])
                stop_loss = entry_price - (atr * 1.5)
                take_profit = entry_price + (atr * 2.0)
                
                return {
                    "Ticker": ticker.replace(".JK", ""),
                    "Sinyal Eksekusi": final_signal,
                    "Setup Teknikal": logic_used,
                    "Sentimen Berita": sentiment_label,
                    "Headline Terkini": news_title,
                    "Harga": f"Rp {entry_price:,.0f}",
                    "VWAP": f"Rp {latest['VWAP']:,.0f}",
                    "RSI (7)": round(float(latest['RSI_7']), 1),
                    "Target (TP)": f"Rp {take_profit:,.0f}",
                    "Stop (SL)": f"Rp {stop_loss:,.0f}"
                }
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(process_ticker, t) for t in tickers_to_scan]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                results.append(res)
                
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['SortRank'] = df_results['Sinyal Eksekusi'].apply(
            lambda x: 1 if "SUPER BUY" in x else (2 if "STRONG SELL" in x else (3 if "RADAR PRIOR" in x else 4))
        )
        df_results = df_results.sort_values(by="SortRank").drop(columns=['SortRank'])
        
    return df_results

# ==========================================
# 4. TAMPILAN ANTARMUKA STREAMLIT
# ==========================================
st.title("⚡ Quantitative Scalping & Mega-News Catalyst Dashboard")
st.markdown("""
Pusat komando *trading* pintar yang menggabungkan aliran berita *real-time* dengan metode teknikal **John F. Carter**. 
Algoritma ini menggunakan filter **TTM Squeeze** untuk probabilitas lonjakan, serta **RSI-7 Bearish Divergence** untuk mendeteksi jebakan *false breakout* yang kehabisan volume.
""")

df_news_live = fetch_all_news()

tab_screener, tab_news = st.tabs(["🎯 Sinergi: Quant Screener + News", "📰 Mega News Aggregator"])

with tab_screener:
    st.header("Sinyal Scalping Saham Indonesia (Interval 15m)")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        scan_limit = st.selectbox("Jumlah saham di-scan:", [50, 100, 200, 500, 800], index=1)
    with col2:
        st.write("")
        st.write("")
        run_scan = st.button("🚀 Pindai Pasar & Analisis Katalis", use_container_width=True, type="primary")
        
    if run_scan:
        with st.spinner(f'Mengkalibrasi sinyal teknikal {scan_limit} saham terhadap ratusan indeks berita Live...'):
            start_time = time.time()
            df_screener = run_quant_screener(scan_limit, df_news_live)
            elapsed_time = round(time.time() - start_time, 2)
            
            if not df_screener.empty:
                st.success(f"Pemindaian selesai dalam {elapsed_time} detik.")
                st.dataframe(df_screener, use_container_width=True, hide_index=True)
            else:
                st.info("Market sedang berkonsolidasi. Tidak ada saham dengan probabilitas tinggi saat ini.")

with tab_news:
    st.header("Aliran Berita & Sentimen Pasar (Live Database)")
    
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("🔄 Refresh Data Berita"):
            st.cache_data.clear()
            st.rerun()
            
    with st.container():
        if not df_news_live.empty:
            st.data_editor(
                df_news_live,
                column_config={
                    "Tautan": st.column_config.LinkColumn("Buka Artikel")
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )
        else:
            st.error("Gagal menarik data RSS. Pastikan koneksi internet server stabil.")
