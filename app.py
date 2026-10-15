import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
import time

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Value Investing Screener (Kiplinger Method)", layout="wide", page_icon="🏦")

# ==========================================
# 1. MODUL FUNDAMENTAL ANALYSIS (Kiplinger's Method)
# ==========================================
def calculate_fundamental_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Ekstrak data fundamental (Jika data tidak tersedia, kembalikan None)
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        pe_ratio = info.get('trailingPE', None)
        forward_pe = info.get('forwardPE', None)
        dividend_yield = info.get('dividendYield', 0)
        debt_to_equity = info.get('debtToEquity', None)
        market_cap = info.get('marketCap', 0)
        book_value = info.get('bookValue', None)
        beta = info.get('beta', None)
        
        # Validasi Kelayakan Data Dasar
        if current_price == 0 or pe_ratio is None or debt_to_equity is None:
            return None
            
        # Konversi Debt to Equity ke desimal (Yahoo Finance mengembalikan format persentase x 100, misal 35.5 untuk 35.5%)
        # Kita sesuaikan dengan standar Kiplinger (< 35%)
        debt_to_equity_pct = debt_to_equity / 100 if debt_to_equity > 10 else debt_to_equity
        
        # =========================================================
        # FILTER 1: BUKAN SAHAM PENNY / MICROCAP (Kiplinger Rule)
        # Asumsi untuk BEI: Kapitalisasi pasar > Rp 2 Triliun & Harga > Rp 100
        # =========================================================
        is_not_penny = (market_cap > 2e12) and (current_price > 100)
        
        # =========================================================
        # FILTER 2: HUTANG RENDAH (Kiplinger Rule: D/E < 35%)
        # =========================================================
        is_low_debt = debt_to_equity_pct < 0.35
        
        # =========================================================
        # FILTER 3: P/E RATIO MASUK AKAL DAN BERTUMBUH
        # Kiplinger Rule: Cari P/E rendah relatif thd pasar/industri. 
        # Batasan konservatif: P/E antara 5 hingga 15.
        # =========================================================
        is_good_pe = 5 <= pe_ratio <= 15
        
        # =========================================================
        # FILTER 4: DIVIDEN (Kiplinger Rule: Punya dividen & bertumbuh)
        # =========================================================
        # Cukup cek apakah ada dividen (Yield > 1%)
        is_dividend_paying = (dividend_yield is not None) and (dividend_yield > 0.01)

        # Matriks Penilaian Total
        score = 0
        reasons = []
        
        if not is_not_penny:
            return None # Langsung diskualifikasi jika saham gorengan
            
        if is_good_pe:
            score += 1
            reasons.append("P/E Undervalued")
        if is_low_debt:
            score += 1
            reasons.append("Hutang Sehat (<35%)")
        if is_dividend_paying:
            score += 1
            reasons.append(f"Dividen Ada ({(dividend_yield*100):.1f}%)")
        if book_value and (current_price < book_value * 1.5): # PBV wajar
            score += 1
            reasons.append("PBV Wajar")
            
        signal = "HOLD / PANTAU"
        if score == 4:
            signal = "🌟 VALUE STOCK (Kriteria Kiplinger Terpenuhi)"
        elif score == 3:
            signal = "👍 POTENSIAL VALUE (Layak Dicicil)"
            
        if score >= 3:
            pbv_calc = current_price / book_value if book_value else 0
            return {
                "Ticker": ticker.replace(".JK", ""),
                "Rekomendasi": signal,
                "Harga": f"Rp {current_price:,.0f}",
                "P/E Ratio": f"{pe_ratio:.1f}x",
                "PBV": f"{pbv_calc:.1f}x",
                "Debt/Equity": f"{(debt_to_equity_pct*100):.1f}%",
                "Div Yield": f"{(dividend_yield*100 if dividend_yield else 0):.1f}%",
                "Beta": f"{beta:.2f}" if beta else "-",
                "Katalis Value": " + ".join(reasons)
            }
        return None
        
    except Exception:
        return None

@st.cache_data(ttl=3600) # Caching fundamental 1 jam (tidak perlu detik-ke-detik)
def run_value_screener(scan_limit):
    try:
        emiten_df = pd.read_csv("daftar_lengkap_saham_indonesia_bei.csv")
        tickers = emiten_df['Ticker_Yahoo_Finance'].dropna().unique().tolist()
    except Exception as e:
        st.error(f"Gagal memuat file CSV: {e}")
        return pd.DataFrame()

    results = []
    
    # Threading di limit ke jumlah yang lebih kecil untuk API Fundamental Yahoo Finance
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(calculate_fundamental_metrics, t) for t in tickers[:scan_limit]]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res: results.append(res)
                
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['SortRank'] = df_results['Rekomendasi'].apply(
            lambda x: 1 if "VALUE STOCK" in x else 2
        )
        df_results = df_results.sort_values(by="SortRank").drop(columns=['SortRank'])
    return df_results

# ==========================================
# 2. TAMPILAN ANTARMUKA STREAMLIT
# ==========================================
st.title("🏦 Kiplinger's Value & Growth Screener")
st.markdown("""
Screener ini menganalisis fundamental perusahaan berdasarkan buku **"The Basics for Investing in Stocks"** oleh Kiplinger. 
Sistem mencari saham yang cocok untuk *Long-Term Hold* dan *Dollar-Cost Averaging* dengan kriteria:
1. **Harga Murah (Value)**: Price-to-Earnings (P/E) antara 5x hingga 15x.
2. **Kesehatan Finansial**: Rasio Hutang terhadap Ekuitas (D/E) di bawah 35%.
3. **Penghasilan Pasif**: Rutin membagikan dividen dengan *yield* di atas 1%.
4. **Bukan Saham Gorengan**: Dilarang membeli saham *penny* (Microcap).
""")

scan_limit = st.selectbox("Jumlah saham di-scan (Fundamental butuh waktu API lebih lama):", [50, 100, 200, 500], index=1)
run_scan = st.button("🔍 Filter Saham Value", use_container_width=True, type="primary")
    
if run_scan:
    with st.spinner(f'Mengunduh Laporan Keuangan (P/E, Dividen, Hutang) untuk {scan_limit} saham...'):
        start_time = time.time()
        df_screener = run_value_screener(scan_limit)
        elapsed_time = round(time.time() - start_time, 2)
        
        if not df_screener.empty:
            st.success(f"Pemindaian selesai dalam {elapsed_time} detik. Ditemukan {len(df_screener)} saham Blue-Chip / Value yang layak untuk dicicil (Dollar-Cost Averaging).")
            st.dataframe(df_screener, use_container_width=True, hide_index=True)
        else:
            st.warning("Tidak ada saham yang memenuhi kriteria ketat Kiplinger's saat ini. Pasar mungkin sedang 'Overvalued'.")
