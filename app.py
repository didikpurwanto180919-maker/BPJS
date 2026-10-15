import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
import concurrent.futures
import datetime
import numpy as np
import time

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Intraday Quant & News Dashboard", layout="wide", page_icon="⚡")

# ==========================================
# 1. MODUL BERITA REAL-TIME (RSS AGGREGATOR)
# ==========================================
@st.cache_data(ttl=300) # Update setiap 5 menit
def fetch_all_news():
    # Daftar RSS Feed Prioritas (Berdasarkan sumber finansial terverifikasi)
    rss_feeds = {
        "CNBC Market": "https://www.cnbcindonesia.com/market/rss",
        "CNBC Rekomendasi": "https://www.cnbcindonesia.com/mymoney/rss",
        "Kontan Investasi": "https://rss.kontan.co.id/news/investasi",
        "Bisnis Market": "https://www.bisnis.com/rss/market",
        "Antara Ekonomi": "https://www.antaranews.com/rss/ekonomi.xml",
        "Sindonews Ekbis": "https://ekbis.sindonews.com/rss",
        "Detik Finance": "https://rss.detik.com/index.php/finance",
        "IDX Channel": "https://www.idxchannel.com/rss",
        "Google News (IHSG)": "https://news.google.com/rss/search?q=saham+BEI+OR+IHSG+when:1d&hl=id&gl=ID&ceid=ID:id",
        "SahamGO": "https://sahamgo.com/feed/"
    }
    
    news_data = []
    
    def parse_feed(source, url):
        try:
            feed = feedparser.parse(url)
            # Ambil 10 berita terbaru per sumber untuk menjaga kecepatan
            for entry in feed.entries[:10]: 
                published_time = entry.published if hasattr(entry, 'published') else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                news_data.append({
                    "Waktu": published_time,
                    "Kategori": source,
                    "Judul Berita": entry.title,
                    "Tautan": entry.link
                })
        except Exception:
            pass

    # Threading untuk konkurensi IO-bound tasks agar scraping lebih cepat
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(parse_feed, source, url) for source, url in rss_feeds.items()]
        concurrent.futures.wait(futures, timeout=15)
        
    df_news = pd.DataFrame(news_data)
    if not df_news.empty:
        df_news['Waktu'] = pd.to_datetime(df_news['Waktu'], utc=True, errors='coerce')
        df_news = df_news.dropna(subset=['Waktu']).sort_values(by='Waktu', ascending=False)
        # Format waktu ke Waktu Lokal (WIB)
        df_news['Waktu'] = df_news['Waktu'].dt.tz_convert('Asia/Jakarta').dt.strftime("%d %b %Y %H:%M")
    return df_news

# ==========================================
# 2. MODUL NLP SEDERHANA: SENTIMEN BERITA
# ==========================================
def get_news_catalyst(ticker, df_news):
    """
    Menganalisis sentimen berita berdasarkan judul yang menyebutkan kode ticker.
    Mengembalikan (Judul Terbaru, Sentimen Teks, Skor Sentimen)
    """
    if df_news is None or df_news.empty:
        return "-", "⚪ Netral", 0
        
    # Ekstrak 4 digit ticker tanpa .JK (contoh: BBCA)
    clean_ticker = ticker.replace(".JK", "").strip()
    
    # Cari judul berita yang mengandung 4 huruf ticker eksak (menggunakan word boundary \b)
    pattern = fr'\b{clean_ticker}\b'
    related_news = df_news[df_news['Judul Berita'].str.contains(pattern, case=False, regex=True, na=False)]
    
    if related_news.empty:
        return "-", "⚪ Netral", 0
        
    # Daftar Kata Kunci Sentimen
    bullish_kw = r'(?i)laba|dividen|akuisisi|rekomendasi|buyback|naik|melonjak|meroket|cuan|ekspansi|kontrak|prospek|terbang|lonjak|rekor'
    bearish_kw = r'(?i)rugi|anjlok|suspensi|uma|delisting|fca|turun|merosot|tekanan|pkpu|pantauan khusus|hindari|gagal bayar|jeblok|ARBs'
    
    score = 0
    # Evaluasi semua berita terkait emiten ini untuk kalkulasi skor akhir
    for title in related_news['Judul Berita']:
        if pd.Series([title]).str.contains(bullish_kw, regex=True).any():
            score += 1
        if pd.Series([title]).str.contains(bearish_kw, regex=True).any():
            score -= 1
            
    latest_title = related_news.iloc[0]['Judul Berita']
    
    if score > 0:
        sentiment = "🟢 Positif"
    elif score < 0:
        sentiment = "🔴 Negatif / UMA"
    else:
        sentiment = "🟡 Sentimen Campuran"
        
    return latest_title, sentiment, score

# ==========================================
# 3. MODUL TECHNICAL ANALYSIS & QUANT SCREENER
# ==========================================
def calculate_quant_indicators(df):
    """Menghitung indikator teknikal untuk strategi Intraday (VWAP, MACD, Squeeze, ATR)"""
    if len(df) < 40:
        return df
        
    # VWAP (Volume Weighted Average Price) Intraday kumulatif aproksimasi
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # MACD (12, 26, 9)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI (14)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Average True Range (ATR)
    df['TR'] = np.maximum((df['High'] - df['Low']), 
                          np.maximum(abs(df['High'] - df['Close'].shift()), 
                                     abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # TTM Squeeze (John Carter)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['MA20'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['MA20'] - (df['BB_std'] * 2)
    
    df['KC_upper'] = df['MA20'] + (df['ATR'] * 1.5)
    df['KC_lower'] = df['MA20'] - (df['ATR'] * 1.5)
    
    # Squeeze menyala ketika BB masuk ke dalam KC
    df['Squeeze_On'] = (df['BB_lower'] > df['KC_lower']) & (df['BB_upper'] < df['KC_upper'])
    
    return df

@st.cache_data(ttl=900) # Caching screener setiap 15 menit
def run_quant_screener(scan_limit, df_news):
    try:
        # Load dataset saham BEI
        emiten_df = pd.read_csv("daftar_lengkap_saham_indonesia_bei.csv")
        tickers = emiten_df['Ticker_Yahoo_Finance'].dropna().unique().tolist()
    except Exception as e:
        st.error(f"Gagal memuat file CSV: {e}")
        return pd.DataFrame()

    tickers_to_scan = tickers[:scan_limit] 
    results = []
    
    def process_ticker(ticker):
        try:
            # Unduh data intraday 15-menitan
            data = yf.download(ticker, period="5d", interval="15m", progress=False)
            if len(data) < 40:
                return None
                
            df = calculate_quant_indicators(data)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # --- STRATEGI TEKNIKAL ---
            # 1: The Opening Gap Breakout
            gap_up = latest['Open'] > prev['Close'] * 1.015
            
            # 2: TTM Squeeze Firing (Momentum Expansion)
            squeeze_firing = (prev['Squeeze_On'] == True) and (latest['Squeeze_On'] == False) and (latest['MACD_Hist'] > 0)
            
            # 3: Pullback & VWAP Bounce
            vwap_bounce = (latest['Low'] <= latest['VWAP'] * 1.005) and (latest['Close'] > latest['VWAP']) and (latest['RSI'] < 45)

            # Validasi Volume (Minimal 2x lipat volume rata-rata)
            volume_spike = latest['Volume'] > df['Volume'].rolling(20).mean().iloc[-1] * 2

            tech_signal = "HOLD / SKIP"
            logic_used = "-"
            
            if gap_up and volume_spike:
                tech_signal = "STRONG BUY"
                logic_used = "Opening Gap Breakout"
            elif squeeze_firing:
                tech_signal = "BUY"
                logic_used = "Squeeze Firing UP"
            elif vwap_bounce:
                tech_signal = "BUY"
                logic_used = "VWAP Bounce / Pullback"
            elif latest['RSI'] > 80 or latest['Close'] < latest['VWAP']:
                tech_signal = "SELL / CUT"
                logic_used = "Overbought / Breakdown VWAP"

            # --- INTEGRASI SENTIMEN BERITA ---
            news_title, sentiment_label, news_score = get_news_catalyst(ticker, df_news)
            
            final_signal = tech_signal
            
            # Matriks Keputusan Sinergi Teknikal + Fundamental News
            if tech_signal in ["STRONG BUY", "BUY"] and news_score > 0:
                final_signal = "🔥 SUPER BUY (Katalis + Teknikal)"
            elif tech_signal in ["STRONG BUY", "BUY"] and news_score < 0:
                final_signal = "⚠️ BUY BERISIKO (Teknikal Beli tapi Berita Buruk)"
            elif tech_signal in ["HOLD / SKIP", "SELL / CUT"] and news_score > 0:
                final_signal = "👀 PANTAU (Breakout Tertunda / Ada Katalis)"
            elif tech_signal in ["HOLD / SKIP", "SELL / CUT"] and news_score < 0:
                final_signal = "❌ HINDARI (Teknikal & Berita Buruk)"

            # Filter: Hanya tampilkan saham yang memiliki aksi (Sinyal Teknikal ATAU Berita Katalis)
            if final_signal not in ["HOLD / SKIP", "SELL / CUT"]:
                atr = latest['ATR']
                entry_price = latest['Close']
                stop_loss = entry_price - (atr * 1.5)
                take_profit = entry_price + (atr * 2.0)
                
                return {
                    "Ticker": ticker.replace(".JK", ""),
                    "Sinyal Eksekusi": final_signal,
                    "Katalis Teknikal": logic_used,
                    "Sentimen Berita": sentiment_label,
                    "Headline Berita Terkini": news_title,
                    "Harga": f"Rp {entry_price:,.0f}",
                    "VWAP": f"Rp {latest['VWAP']:,.0f}",
                    "RSI": round(latest['RSI'], 1),
                    "Target Jual (TP)": f"Rp {take_profit:,.0f}",
                    "Stop Loss (SL)": f"Rp {stop_loss:,.0f}"
                }
        except Exception:
            return None

    # Eksekusi paralel untuk memindai ratusan saham secara serentak
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(process_ticker, t) for t in tickers_to_scan]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                results.append(res)
                
    # Urutkan agar saham dengan SUPER BUY (🔥) muncul paling atas
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by="Sinyal Eksekusi", ascending=False)
        
    return df_results

# ==========================================
# 4. TAMPILAN ANTARMUKA STREAMLIT
# ==========================================
st.title("⚡ Quantitative Scalping & News Catalyst Dashboard")
st.markdown("""
Pusat komando *trading* pintar yang menggabungkan aliran berita *real-time* dengan deteksi teknikal mesin.
Algoritma ini menyeleksi sinyal beli berdasarkan momentum teknikal dan melakukan *cross-check* terhadap berita yang beredar untuk mendeteksi jebakan atau mengkonfirmasi katalis.
""")

# Pre-fetch Data Berita di Latar Belakang
df_news_live = fetch_all_news()

tab_screener, tab_news = st.tabs(["🎯 Sinergi: Quant Screener + News", "📰 Global & Lokal News Aggregator"])

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
        with st.spinner(f'Menganalisis {scan_limit} saham teratas dan mencocokkan dengan puluhan sumber berita Live...'):
            start_time = time.time()
            # Pass dataframe berita yang sudah difetch ke screener
            df_screener = run_quant_screener(scan_limit, df_news_live)
            elapsed_time = round(time.time() - start_time, 2)
            
            if not df_screener.empty:
                st.success(f"Pemindaian selesai dalam {elapsed_time} detik. Menampilkan saham potensial yang digerakkan oleh teknikal atau sentimen.")
                st.dataframe(df_screener, use_container_width=True, hide_index=True)
            else:
                st.info("Pasar sedang lesu. Tidak ada saham dengan kombinasi sinyal teknikal atau katalis berita yang memadai saat ini.")

with tab_news:
    st.header("Aliran Berita & Sentimen Pasar (Live)")
    st.markdown("Database Mentah yang digunakan oleh *Quant Screener* untuk menilai katalis fundamental emiten.")
    
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
                    "Tautan": st.column_config.LinkColumn("Baca Selengkapnya")
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )
        else:
            st.error("Gagal menarik data RSS. Periksa koneksi internet atau ketersediaan endpoint web berita.")
