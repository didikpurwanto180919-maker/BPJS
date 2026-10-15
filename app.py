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
st.set_page_config(page_title="Intraday Quant Dashboard", layout="wide", page_icon="⚡")

# ==========================================
# 1. MODUL BERITA REAL-TIME (RSS AGGREGATOR)
# ==========================================
@st.cache_data(ttl=300) # Update setiap 5 menit
def fetch_all_news():
    # Daftar RSS Feed Komprehensif (Berdasarkan sumber terverifikasi)
    rss_feeds = {
        "CNBC Market": "https://www.cnbcindonesia.com/market/rss",
        "CNBC Rekomendasi": "https://www.cnbcindonesia.com/mymoney/rss",
        "Kontan Investasi": "https://rss.kontan.co.id/news/investasi",
        "Bisnis Market": "https://www.bisnis.com/rss/market",
        "Antara Ekonomi": "https://www.antaranews.com/rss/ekonomi.xml",
        "Sindonews Ekbis": "https://ekbis.sindonews.com/rss",
        "Detik Finance": "https://rss.detik.com/index.php/finance",
        "Republika Ekonomi": "https://www.republika.co.id/rss/ekonomi",
        "IDX Channel": "https://www.idxchannel.com/rss",
        "Google News (IHSG)": "https://news.google.com/rss/search?q=saham+BEI+OR+IHSG+when:1d&hl=id&gl=ID&ceid=ID:id",
        "Google News (Rekomendasi)": "https://news.google.com/rss/search?q=(site:cnbcindonesia.com+OR+site:kontan.co.id)+(\"rekomendasi+saham\"+OR+\"saham+pilihan\")+when:1d&hl=id&gl=ID&ceid=ID:id",
        "SahamGO": "https://sahamgo.com/feed/",
        "PintarSaham": "https://pintarsaham.id/feed/"
    }
    
    news_data = []
    
    def parse_feed(source, url):
        try:
            feed = feedparser.parse(url)
            # Ambil 5 berita terbaru per sumber untuk menjaga kecepatan
            for entry in feed.entries[:5]: 
                published_time = entry.published if hasattr(entry, 'published') else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                news_data.append({
                    "Waktu": published_time,
                    "Kategori": source,
                    "Judul Berita": entry.title,
                    "Tautan": entry.link
                })
        except Exception:
            pass

    # Threading untuk konkurensi IO-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(parse_feed, source, url) for source, url in rss_feeds.items()]
        concurrent.futures.wait(futures, timeout=10)
        
    df_news = pd.DataFrame(news_data)
    if not df_news.empty:
        df_news['Waktu'] = pd.to_datetime(df_news['Waktu'], utc=True, errors='coerce')
        df_news = df_news.dropna(subset=['Waktu']).sort_values(by='Waktu', ascending=False)
        # Format waktu ke Waktu Lokal (WIB)
        df_news['Waktu'] = df_news['Waktu'].dt.tz_convert('Asia/Jakarta').dt.strftime("%d %b %Y %H:%M")
    return df_news.head(100) # Batasi 100 berita agregat terbaru

# ==========================================
# 2. MODUL TECHNICAL ANALYSIS & QUANT SCREENER
# ==========================================
def calculate_quant_indicators(df):
    """Menghitung indikator teknikal untuk strategi Intraday (VWAP, MACD, Squeeze, ATR)"""
    if len(df) < 40:
        return df
        
    # VWAP (Volume Weighted Average Price) - Reset harian secara ideal, namun didekati dengan kumulatif rolling untuk dataset kecil
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
    
    # Average True Range (ATR) untuk kalkulasi Stop Loss & Take Profit
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
    
    # Squeeze menyala ketika Bollinger Bands masuk ke dalam Keltner Channels
    df['Squeeze_On'] = (df['BB_lower'] > df['KC_lower']) & (df['BB_upper'] < df['KC_upper'])
    
    return df

@st.cache_data(ttl=900) # Caching screener setiap 15 menit
def run_quant_screener(scan_limit=50):
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
            
            # --- STRATEGI 1: THE OPENING GAP ---
            gap_up = latest['Open'] > prev['Close'] * 1.015
            
            # --- STRATEGI 2: TTM SQUEEZE FIRING ---
            # Squeeze 'On' pada bar sebelumnya, dan 'Off' (firing) pada bar saat ini dengan momentum positif
            squeeze_firing = (prev['Squeeze_On'] == True) and (latest['Squeeze_On'] == False) and (latest['MACD_Hist'] > 0)
            
            # --- STRATEGI 3: PULLBACK & VWAP BOUNCE ---
            # Harga terkoreksi tapi tertahan di atas VWAP, dengan RSI oversold namun MACD mulai menyempit
            vwap_bounce = (latest['Low'] <= latest['VWAP'] * 1.005) and (latest['Close'] > latest['VWAP']) and (latest['RSI'] < 45)

            # Validasi Likuiditas & Volume (Minimal 2x rata-rata 20 periode terakhir)
            volume_spike = latest['Volume'] > df['Volume'].rolling(20).mean().iloc[-1] * 2

            signal = "NO ACTION"
            logic_used = "-"
            
            if gap_up and volume_spike:
                signal = "STRONG BUY"
                logic_used = "Opening Gap Breakout"
            elif squeeze_firing:
                signal = "BUY"
                logic_used = "Squeeze Firing UP"
            elif vwap_bounce:
                signal = "BUY"
                logic_used = "VWAP Bounce / Pullback"
            elif latest['RSI'] > 80 or latest['Close'] < latest['VWAP']:
                signal = "SELL / CUT"
                logic_used = "Overbought / Breakdown VWAP"

            if signal in ["STRONG BUY", "BUY"]:
                # Kalkulasi Intraday Target dan Stop Loss berdasarkan ATR
                atr = latest['ATR']
                entry_price = latest['Close']
                stop_loss = entry_price - (atr * 1.5)
                take_profit = entry_price + (atr * 2.0)
                
                return {
                    "Ticker": ticker.replace(".JK", ""),
                    "Harga": f"Rp {entry_price:,.0f}",
                    "Sinyal": signal,
                    "Katalis Teknikal": logic_used,
                    "VWAP": f"Rp {latest['VWAP']:,.0f}",
                    "RSI": round(latest['RSI'], 1),
                    "Target Jual (TP)": f"Rp {take_profit:,.0f}",
                    "Stop Loss (SL)": f"Rp {stop_loss:,.0f}"
                }
        except Exception:
            return None

    # Eksekusi paralel
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(process_ticker, t) for t in tickers_to_scan]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                results.append(res)
                
    return pd.DataFrame(results)

# ==========================================
# 3. TAMPILAN ANTARMUKA STREAMLIT
# ==========================================
st.title("⚡ Quantitative Scalping & Intraday Dashboard")
st.markdown("Pusat komando *trading* harian yang menggabungkan aliran berita *real-time* dengan deteksi anomali teknikal dan struktur pasar secara otomatis.")

tab_screener, tab_news = st.tabs(["🎯 Intraday Quant Screener", "📰 Global & Lokal News Aggregator"])

with tab_screener:
    st.header("Sinyal Scalping Saham Indonesia (Interval 15m)")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        scan_limit = st.selectbox("Jumlah saham di-scan:", [50, 100, 200, 500], index=0)
    with col2:
        st.write("")
        st.write("")
        run_scan = st.button("🚀 Pindai Pasar Sekarang", use_container_width=True)
        
    if run_scan:
        with st.spinner(f'Menganalisis {scan_limit} saham teratas berdasarkan struktur algoritma intraday...'):
            start_time = time.time()
            df_screener = run_quant_screener(scan_limit=scan_limit)
            elapsed_time = round(time.time() - start_time, 2)
            
            if not df_screener.empty:
                st.success(f"Pemindaian selesai dalam {elapsed_time} detik. Ditemukan {len(df_screener)} sinyal aksi.")
                st.dataframe(df_screener, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada saham yang memenuhi probabilitas eksekusi tinggi saat ini.")

with tab_news:
    st.header("Aliran Berita & Sentimen Pasar (Live)")
    st.markdown("Berita dikompilasi dari portal finansial prioritas dan diurutkan secara kronologis untuk mendeteksi katalis fundamental seketika.")
    
    if st.button("🔄 Refresh Berita", type="primary"):
        st.cache_data.clear()
        
    with st.spinner('Mengambil feed RSS paralel...'):
        df_news = fetch_all_news()
        if not df_news.empty:
            # Gunakan st.data_editor untuk tampilan tabel yang interaktif
            st.data_editor(
                df_news,
                column_config={
                    "Tautan": st.column_config.LinkColumn("Baca Selengkapnya")
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )
        else:
            st.error("Gagal menarik data RSS. Periksa koneksi internet atau ketersediaan endpoint.")
