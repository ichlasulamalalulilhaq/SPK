"""
Gaya visual antarmuka Streamlit.

Murni presentasi: tidak ada perhitungan di sini. Dipisahkan dari app/app.py
agar perubahan tampilan tidak bercampur dengan logika keputusan.

Arah desain: dokumen keuangan koperasi. Latar kertas, tinta gelap, satu
aksen hijau tua; judul berkait (serif), teks tanpa kait, angka berlebar
tetap agar kolom nominal sejajar. Tanpa gradien, emoji, atau bayangan.
"""

from __future__ import annotations

from html import escape

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --tinta: #1c1b19;
  --redup: #6b665c;
  --garis: #d9d3c5;
  --kertas: #f7f5f0;
  --kartu: #fffdf8;
  --aksen: #1f5c4a;
  --aksen-muda: #e2ebe5;
  --merah: #9b2c1f;
  --merah-muda: #f4e3df;
  --kuning: #7d5f0c;
  --kuning-muda: #f3ead2;
  --biru: #2f4a6b;
  --biru-muda: #e3e8ef;
  --sans: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --serif: 'IBM Plex Serif', Georgia, 'Times New Roman', serif;
  --mono: 'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace;
}

/* ---- dasar --------------------------------------------------------- */
html, body, .stApp, [data-testid="stMarkdownContainer"], input, textarea, button {
  font-family: var(--sans);
}
.stApp { color: var(--tinta); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }
[data-testid="stDecoration"], footer, #MainMenu { display: none; }
[data-testid="stHeader"] { background: transparent; }

h1, h2, h3, h4 {
  font-family: var(--serif) !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em;
  color: var(--tinta);
}
h3 { font-size: 1.2rem !important; padding-top: .4rem !important; }
h4 { font-size: 1.05rem !important; }

/* h5 dipakai sebagai label bagian bergaya formulir */
h5 {
  font-family: var(--sans) !important;
  font-size: .72rem !important;
  font-weight: 600 !important;
  letter-spacing: .09em;
  text-transform: uppercase;
  color: var(--redup) !important;
  border-bottom: 1px solid var(--garis);
  padding: 1.1rem 0 .35rem !important;
  margin-bottom: .5rem !important;
}

[data-testid="stCaptionContainer"], .stCaption { color: var(--redup); }
code { font-family: var(--mono); background: transparent; color: var(--tinta); }
hr { border-color: var(--garis) !important; }

/* ---- kop halaman --------------------------------------------------- */
.kop { border-bottom: 2px solid var(--tinta); padding-bottom: .9rem; margin-bottom: .4rem; }
.kop-atas {
  font-size: .72rem; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; color: var(--aksen);
}
.kop h1 {
  font-size: clamp(1.55rem, 3.4vw, 2.2rem) !important;
  line-height: 1.15; margin: .25rem 0 .35rem !important; padding: 0 !important;
}
.kop-sub { color: var(--redup); font-size: .92rem; max-width: 60ch; margin: 0; }
.kop-fakta {
  display: flex; flex-wrap: wrap; gap: .4rem 2.2rem;
  margin: 1rem 0 0; padding-top: .8rem; border-top: 1px solid var(--garis);
}
.kop-fakta div { min-width: 7.5rem; }
.kop-fakta dt {
  font-size: .68rem; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--redup);
}
.kop-fakta dd {
  margin: .1rem 0 0; font-family: var(--mono); font-size: 1.02rem;
  font-variant-numeric: tabular-nums; color: var(--tinta);
}

/* ---- metrik -------------------------------------------------------- */
[data-testid="stMetric"] {
  background: var(--kartu);
  border: 1px solid var(--garis);
  border-top: 3px solid var(--tinta);
  border-radius: 2px;
  padding: .7rem .85rem .6rem;
}
[data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] {
  font-size: .7rem !important; font-weight: 600; letter-spacing: .07em;
  text-transform: uppercase; color: var(--redup) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--mono); font-size: 1.4rem !important;
  font-variant-numeric: tabular-nums; color: var(--tinta);
}
[data-testid="stMetricDelta"] { font-family: var(--mono); font-size: .78rem; }

/* ---- tab ----------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 0; border-bottom: 1px solid var(--garis); overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
  padding: .65rem 1rem; height: auto; color: var(--redup);
  font-weight: 500; font-size: .9rem; background: transparent;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--tinta); }
.stTabs [aria-selected="true"] { color: var(--tinta) !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--aksen); height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }

/* ---- sidebar ------------------------------------------------------- */
[data-testid="stSidebar"] {
  background: #efebe2; border-right: 1px solid var(--garis);
}
[data-testid="stSidebar"] h2 {
  font-size: 1.05rem !important; border-bottom: 2px solid var(--tinta);
  padding-bottom: .4rem !important;
}
[data-testid="stSidebar"] h3 {
  font-family: var(--sans) !important; font-size: .72rem !important;
  letter-spacing: .09em; text-transform: uppercase; color: var(--redup);
}

/* ---- tombol -------------------------------------------------------- */
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
  border-radius: 2px; font-weight: 500; box-shadow: none;
  border: 1px solid var(--tinta); transition: background .12s;
}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
  background: var(--aksen); border-color: var(--aksen); color: #fff;
}

/* ---- isian, formulir, expander, tabel ------------------------------ */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
  border-radius: 2px !important;
}
[data-testid="stForm"] {
  background: var(--kartu); border: 1px solid var(--garis);
  border-radius: 2px; padding: .4rem 1.2rem 1.2rem;
}
[data-testid="stExpander"] details, [data-testid="stExpander"] {
  border-radius: 2px; border-color: var(--garis); background: var(--kartu);
}
[data-testid="stDataFrame"] { border: 1px solid var(--garis); border-radius: 2px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 2px; }

/* ---- pesan --------------------------------------------------------- */
[data-testid="stAlert"] { border-radius: 2px; }
[data-testid="stAlert"] > div { border-left-width: 3px; }

/* ---- lencana status ------------------------------------------------ */
.lencana {
  display: inline-block; font-family: var(--sans); font-size: .7rem;
  font-weight: 600; letter-spacing: .07em; text-transform: uppercase;
  padding: .15rem .5rem; border: 1px solid currentColor; border-radius: 2px;
  vertical-align: middle; white-space: nowrap;
}
.lencana-setuju { color: var(--aksen); background: var(--aksen-muda); }
.lencana-tunda  { color: var(--biru);  background: var(--biru-muda); }
.lencana-tolak  { color: var(--merah); background: var(--merah-muda); }
.lencana-verif  { color: var(--kuning); background: var(--kuning-muda); }

/* ---- kepala bagian ------------------------------------------------- */
.bagian { margin: .2rem 0 .7rem; }
.bagian-label {
  font-size: .68rem; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; color: var(--aksen);
}
.bagian h3 { margin: .1rem 0 .15rem !important; padding: 0 !important; }
.bagian p { margin: 0; color: var(--redup); font-size: .9rem; max-width: 70ch; }

/* ---- pita ringkasan ------------------------------------------------ */
.pita {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr));
  border: 1px solid var(--garis); border-top: 3px solid var(--tinta);
  background: var(--kartu); margin: .2rem 0 1.1rem;
}
.pita > div { padding: .7rem .9rem .75rem; border-right: 1px solid var(--garis); }
.pita > div:last-child { border-right: 0; }
.pita dt {
  font-size: .66rem; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--redup);
}
.pita dd {
  margin: .15rem 0 0; font-family: var(--mono); font-size: 1.35rem;
  font-variant-numeric: tabular-nums; line-height: 1.2; color: var(--tinta);
}
.pita small { display: block; font-size: .74rem; color: var(--redup); margin-top: .15rem; }
.pita .baik { color: var(--aksen); }
.pita .buruk { color: var(--merah); }
.pita .eval dt::after {
  content: "evaluasi"; margin-left: .4rem; padding: 0 .3rem;
  border: 1px solid var(--kuning); color: var(--kuning); font-size: .58rem;
}
@media (max-width: 640px) {
  .pita > div { border-right: 0; border-bottom: 1px solid var(--garis); }
}

/* ---- rapor kriteria ------------------------------------------------ */
.rapor { width: 100%; border-collapse: collapse; margin: .3rem 0 .8rem; font-size: .88rem; }
.rapor th {
  text-align: left; font-size: .66rem; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--redup);
  border-bottom: 1px solid var(--tinta); padding: .35rem .4rem;
}
.rapor td { padding: .45rem .4rem; border-bottom: 1px solid var(--garis); vertical-align: middle; }
.rapor .kode { font-family: var(--mono); color: var(--redup); width: 2.4rem; }
.rapor .nilai {
  font-family: var(--mono); text-align: right; white-space: nowrap;
  font-variant-numeric: tabular-nums; width: 7.5rem;
}
.rapor .arah { font-size: .72rem; color: var(--redup); white-space: nowrap; width: 4.5rem; }
.rapor .batang { width: 32%; }
.rapor .batang span { display: block; height: 6px; background: var(--garis); position: relative; }
.rapor .batang i { position: absolute; left: 0; top: 0; bottom: 0; background: var(--aksen); }
.rapor .batang.cost i { background: var(--merah); }

/* ---- kaki halaman -------------------------------------------------- */
.kaki {
  margin-top: 2.5rem; padding-top: .8rem; border-top: 1px solid var(--garis);
  font-size: .78rem; color: var(--redup); line-height: 1.55;
}

@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .kop-fakta { gap: .5rem 1.4rem; }
  .stTabs [data-baseweb="tab"] { padding: .55rem .7rem; font-size: .84rem; }
}
</style>
"""

_KELAS_STATUS = {
    "DISETUJUI": "lencana-setuju",
    "DITUNDA": "lencana-tunda",
    "DITOLAK": "lencana-tolak",
    "PERLU VERIFIKASI": "lencana-verif",
}


def pasang_gaya(st) -> None:
    """Sisipkan lembar gaya. Dipanggil sekali di awal setiap render."""
    st.markdown(_CSS, unsafe_allow_html=True)


def lencana(status: str) -> str:
    """HTML lencana berwarna untuk status keputusan."""
    kelas = _KELAS_STATUS.get(status, "lencana-verif")
    return f'<span class="lencana {kelas}">{escape(str(status))}</span>'


def kop(st, atas: str, judul: str, sub: str, fakta: list[tuple[str, str]]) -> None:
    """Kop halaman: label lembaga, judul, keterangan, dan baris angka ringkas."""
    isi_fakta = "".join(
        f"<div><dt>{escape(k)}</dt><dd>{escape(v)}</dd></div>" for k, v in fakta
    )
    st.markdown(
        f'<header class="kop"><div class="kop-atas">{escape(atas)}</div>'
        f"<h1>{escape(judul)}</h1><p class=\"kop-sub\">{escape(sub)}</p>"
        f'<dl class="kop-fakta">{isi_fakta}</dl></header>',
        unsafe_allow_html=True,
    )


def kaki(st, teks: str) -> None:
    """Catatan kaki halaman."""
    st.markdown(f'<div class="kaki">{escape(teks)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# KOMPONEN
# --------------------------------------------------------------------------
# Palet grafik: aksen hijau, lalu warna redup yang tetap terbedakan.
PALET = ["#1F5C4A", "#9B2C1F", "#2F4A6B", "#7D5F0C", "#5E4B8B",
         "#3D7A8A", "#8A5A44", "#6B665C"]

# Warna latar dan teks sel status pada tabel keputusan.
_WARNA_STATUS = {
    "DISETUJUI": ("#E2EBE5", "#1F5C4A"),
    "DITUNDA": ("#E3E8EF", "#2F4A6B"),
    "DITOLAK": ("#F4E3DF", "#9B2C1F"),
    "PERLU VERIFIKASI": ("#F3EAD2", "#7D5F0C"),
}


def bagian(st, label: str, judul: str, inti: str = "",
           penjelasan: str | None = None) -> None:
    """
    Kepala bagian seragam: label kecil, judul, satu kalimat inti.

    Keterangan panjang tidak dibuang, melainkan dipindah ke expander
    "Cara membaca" agar halaman tetap ringkas.
    """
    st.markdown(
        f'<div class="bagian"><div class="bagian-label">{escape(label)}</div>'
        f"<h3>{escape(judul)}</h3>"
        + (f"<p>{escape(inti)}</p>" if inti else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    if penjelasan:
        with st.expander("Cara membaca"):
            st.markdown(penjelasan)


def pita(st, butir: list[dict]) -> None:
    """
    Pita ringkasan angka. Tiap butir: label, nilai, catatan (opsional),
    nada ("baik"/"buruk", opsional), evaluasi (bool), bantuan (tooltip).
    """
    isi = []
    for b in butir:
        kelas = "eval" if b.get("evaluasi") else ""
        nada = b.get("nada") or ""
        catatan = f'<small class="{nada}">{escape(b["catatan"])}</small>' if b.get("catatan") else ""
        judul = f' title="{escape(b["bantuan"])}"' if b.get("bantuan") else ""
        isi.append(
            f'<div class="{kelas}"{judul}><dt>{escape(b["label"])}</dt>'
            f'<dd>{escape(str(b["nilai"]))}</dd>{catatan}</div>'
        )
    st.markdown(f'<dl class="pita">{"".join(isi)}</dl>', unsafe_allow_html=True)


def rapor(st, baris: list[dict]) -> None:
    """
    Tabel rapor kriteria. Tiap baris: kode, nama, nilai (teks), porsi (0-1
    untuk panjang batang), benefit (bool).
    """
    isi = []
    for b in baris:
        porsi = max(0.0, min(1.0, float(b["porsi"])))
        arah = "makin tinggi makin baik" if b["benefit"] else "makin rendah makin baik"
        isi.append(
            f'<tr><td class="kode">{escape(b["kode"])}</td>'
            f'<td>{escape(b["nama"])}</td>'
            f'<td class="nilai">{escape(b["nilai"])}</td>'
            f'<td class="batang{"" if b["benefit"] else " cost"}" title="{arah}">'
            f'<span><i style="width:{porsi * 100:.0f}%"></i></span></td>'
            f'<td class="arah">{"benefit" if b["benefit"] else "cost"}</td></tr>'
        )
    st.markdown(
        '<table class="rapor"><thead><tr><th></th><th>Kriteria</th>'
        '<th style="text-align:right">Nilai</th><th>Posisi</th><th>Arah</th>'
        f'</tr></thead><tbody>{"".join(isi)}</tbody></table>',
        unsafe_allow_html=True,
    )


def warnai_status(df, kolom: str = "Status"):
    """Kembalikan Styler dengan sel status berwarna; data tidak berubah."""
    def gaya(nilai):
        latar, teks = _WARNA_STATUS.get(nilai, ("transparent", "inherit"))
        return f"background-color: {latar}; color: {teks}; font-weight: 600"

    styler = df.style
    # Styler.map tersedia sejak pandas 2.1; applymap untuk versi lebih lama.
    terapkan = getattr(styler, "map", None) or styler.applymap
    return terapkan(gaya, subset=[kolom])
