"""
Antarmuka web SPK Prioritisasi Penyaluran Kredit Mikro.

Jalankan dari akar proyek:
    streamlit run app/app.py

Bobot kriteria dapat diubah dan seluruh peringkat dihitung ulang seketika,
tiap peringkat disertai alasan, dan tiap skenario dapat disimpan untuk
ditelusuri kembali.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Akar proyek ditambahkan ke sys.path agar modul src dapat diimpor ketika
# Streamlit dijalankan dari direktori mana pun.
AKAR = Path(__file__).resolve().parents[1]
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))

from src import config as cfg                      # noqa: E402
from src import evaluasi, knowledge, ml_model, pipeline, sensitivity  # noqa: E402
from src import input_pemohon as inp               # noqa: E402
from src import tampilan                           # noqa: E402
from src.mcdm import ahp, ranking                  # noqa: E402

SUMBER_GABUNG = "40 pemohon contoh + pemohon input"
SUMBER_INPUT = "Hanya pemohon input"

def rupiah(nilai: float, singkat: bool = False) -> str:
    """
    Format nilai rupiah secara seragam di seluruh antarmuka.

    Dibuat terpusat untuk menutup temuan HE-09, yaitu pemisah ribuan yang
    berbeda antartab karena sebagian memakai format bawaan Python.
    """
    if singkat and abs(nilai) >= 1_000_000:
        # Format Indonesia: titik pemisah ribuan, koma desimal.
        angka = f"{nilai / 1_000_000:,.1f}".replace(",", "_").replace(".", ",")
        return f"Rp{angka.replace('_', '.')} jt"
    return f"Rp{nilai:,.0f}".replace(",", ".")


st.set_page_config(
    page_title="SPK Prioritisasi Kredit Mikro",
    layout="wide",
)


# --------------------------------------------------------------------------
# PEMUATAN DATA
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Menjalankan pipeline (ETL, pelatihan model, MCDM)...")
def muat_pipeline() -> dict:
    """
    Jalankan pipeline sekali, lalu simpan hasilnya di cache.

    cache_resource dipakai, bukan cache_data, karena hasil memuat objek
    scikit-learn yang tidak dapat di-hash maupun diserialkan oleh Streamlit.
    """
    return pipeline.jalankan(verbose=False)


@st.cache_data(show_spinner="Menilai pemohon input...")
def nilai_pemohon_input(isian: tuple, _konteks: dict) -> pd.DataFrame:
    """
    Hitung C1-C6 pemohon input, disimpan di cache per isi daftar.

    Awalan garis bawah pada _konteks membuat Streamlit tidak mencoba
    meng-hash objek scikit-learn di dalamnya.
    """
    return inp.nilai_pemohon([dict(d) for d in isian], _konteks)


def susun_konteks_aktif(konteks: dict, sumber: str) -> dict:
    """
    Gabungkan kohort contoh dengan pemohon input menjadi matriks keputusan.

    Seluruh tab membaca konteks hasil fungsi ini, sehingga pemohon input ikut
    dipertimbangkan pada keputusan, sensitivitas, what-if, dan riwayat.
    """
    bagian = []
    if sumber == SUMBER_GABUNG:
        contoh = konteks["df_periode"].copy()
        contoh["sumber"] = "Contoh"
        contoh["perlu_verifikasi"] = False
        bagian.append(contoh)

    daftar = st.session_state.get("pemohon_baru", [])
    if daftar:
        isian = tuple(tuple(d.items()) for d in daftar)
        bagian.append(nilai_pemohon_input(isian, konteks))

    df = (pd.concat(bagian, ignore_index=True) if bagian
          else konteks["df_periode"].iloc[0:0].copy())
    if "perlu_verifikasi" not in df.columns:
        df["perlu_verifikasi"] = False
    df["perlu_verifikasi"] = df["perlu_verifikasi"].fillna(False).astype(bool)

    matriks, _, benefit = pipeline.bentuk_matriks_keputusan(df)
    return {
        **konteks,
        "df_periode": df,
        "matriks": matriks,
        "benefit": benefit,
        "id_alternatif": df["id_pemohon"].tolist(),
    }


def hitung_ulang(konteks: dict, bobot: np.ndarray, metode: str) -> pd.DataFrame:
    """Hitung ulang peringkat dengan bobot pilihan pengguna."""
    matriks = konteks["matriks"]
    benefit = konteks["benefit"]

    fungsi = {"TOPSIS": ranking.topsis, "SAW": ranking.saw, "VIKOR": ranking.vikor}
    skor = fungsi[metode](matriks, bobot, benefit)["skor"]

    df = konteks["df_periode"].copy()
    df["skor"] = skor
    df["rank_TOPSIS"] = (-skor).argsort().argsort() + 1   # nama kolom dipertahankan
    return df


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
def gambar_sidebar(bobot_ahp: np.ndarray) -> dict:
    st.sidebar.header("Parameter Keputusan")

    st.sidebar.markdown("##### Pemohon")
    n_input = len(st.session_state.get("pemohon_baru", []))
    sumber = st.sidebar.radio(
        "Pemohon yang diperingkat", [SUMBER_GABUNG, SUMBER_INPUT],
        help=f"Pemohon input saat ini: {n_input}. Tambahkan lewat tab Input Pemohon.",
    )

    # Batas DITURUNKAN dari konfigurasi, tidak ditulis tangan. Versi
    # sebelumnya memakai batas skala Deutsche Mark yang tertinggal saat
    # satuan diubah ke Rupiah, sehingga nilai bawaan Rp60 juta melampaui
    # batas atas Rp200 ribu dan aplikasi gagal dijalankan. Menurunkan batas
    # dari sumber yang sama membuat kesalahan semacam itu tidak bisa terulang.
    st.sidebar.markdown("##### Dana dan batas")
    plafon_awal = int(cfg.PLAFON_PERIODE)
    langkah = max(1, 10 ** (len(str(plafon_awal)) - 3))   # 2 digit pertama
    plafon = st.sidebar.number_input(
        "Plafon periode (Rp)",
        min_value=int(plafon_awal * 0.1),
        max_value=int(plafon_awal * 5),
        value=plafon_awal,
        step=langkah,
        help=f"Nilai bawaan {rupiah(plafon_awal)} berasal dari config.PLAFON_PERIODE.",
    )
    st.sidebar.caption(f"Plafon aktif: {rupiah(plafon)}")
    pd_batas = st.sidebar.slider(
        "Batas keras PD", 0.30, 0.95, float(cfg.PD_HARD_LIMIT), 0.01,
        help="Pemohon dengan PD di atas nilai ini ditolak berapa pun peringkatnya.",
    )
    st.sidebar.markdown("##### Metode")
    metode = st.sidebar.selectbox("Metode pemeringkatan", ["TOPSIS", "SAW", "VIKOR"])

    st.sidebar.markdown("##### Bobot kriteria")

    if st.sidebar.button("Kembalikan ke bobot AHP", use_container_width=True):
        for i, kode in enumerate(cfg.CRITERIA_CODES):
            st.session_state[f"w_{kode}"] = float(bobot_ahp[i])

    bobot_mentah = []
    for i, kode in enumerate(cfg.CRITERIA_CODES):
        arah = "benefit" if cfg.CRITERIA_BENEFIT[kode] else "cost"
        bobot_mentah.append(
            st.sidebar.slider(
                f"{kode} · {cfg.CRITERIA_NAMES[kode]} ({arah})",
                0.0, 0.60, float(bobot_ahp[i]), 0.005, key=f"w_{kode}",
            )
        )

    bobot = np.asarray(bobot_mentah, dtype=float)

    # Perbaikan temuan HE-03 (keparahan 4). Sebelumnya seluruh bobot dapat
    # digeser ke nol dan sistem tetap menampilkan peringkat, padahal
    # peringkat tanpa bobot tidak punya makna sama sekali. Kasus ini kini
    # dihentikan dengan pesan yang menjelaskan akibatnya, bukan sekadar
    # dinormalisasi diam-diam.
    if bobot.sum() <= 0:
        st.sidebar.error(
            "Seluruh bobot bernilai nol. Pemeringkatan tidak dapat dihitung "
            "karena tidak ada kriteria yang dipakai membandingkan pemohon. "
            "Naikkan setidaknya satu bobot, atau tekan 'Kembalikan ke bobot AHP'."
        )
        st.stop()

    # Bobot dinormalisasi agar selalu menjumlah 1. Tanpa ini, menaikkan satu
    # slider akan menaikkan skor semua alternatif dan perbandingan jadi tidak sah.
    bobot = bobot / bobot.sum()

    st.sidebar.caption(
        "Bobot dinormalisasi otomatis agar berjumlah 1. "
        f"Total sebelum normalisasi: {sum(bobot_mentah):.3f}"
    )

    # Perbaikan temuan HE-08 (keparahan 4): mode evaluasi dinyatakan eksplisit.
    # st.toggle baru tersedia sejak Streamlit 1.26. checkbox dipakai sebagai
    # cadangan agar aplikasi tetap berjalan pada pemasangan yang lebih lama.
    st.sidebar.markdown("##### Mode")
    sakelar = getattr(st.sidebar, "toggle", st.sidebar.checkbox)
    mode_evaluasi = sakelar(
        "Mode evaluasi", value=True,
        help="Menampilkan tingkat gagal bayar sebenarnya. Hanya mungkin pada "
             "data uji berlabel. Pada penyaluran nyata, angka ini belum ada "
             "sampai masa angsuran berjalan.",
    )
    return {"plafon": plafon, "pd_batas": pd_batas, "metode": metode,
            "bobot": bobot, "mode_evaluasi": mode_evaluasi, "sumber": sumber}


# --------------------------------------------------------------------------
# TAB: KEPUTUSAN
# --------------------------------------------------------------------------
def tab_keputusan(df_keputusan: pd.DataFrame, par: dict) -> None:
    disetujui = df_keputusan[df_keputusan["status_keputusan"] == "DISETUJUI"]

    ada = len(disetujui) > 0
    tersalur = df_keputusan["nominal_disetujui"].sum()
    selisih_pd = disetujui["pd_model"].mean() - df_keputusan["pd_model"].mean() if ada else 0.0
    selisih_iks = disetujui["iks_skor"].mean() - df_keputusan["iks_skor"].mean() if ada else 0.0
    butir = [
        {"label": "Disetujui", "nilai": len(disetujui),
         "catatan": f"dari {len(df_keputusan)} pemohon"},
        {"label": "Dana tersalur", "nilai": rupiah(tersalur, singkat=True),
         "catatan": f"{tersalur / par['plafon'] * 100:.1f}% plafon".replace(".", ",")},
        {"label": "PD rata-rata penerima",
         "nilai": f"{disetujui['pd_model'].mean():.3f}" if ada else "–",
         "catatan": f"{selisih_pd:+.3f} vs semua pemohon" if ada else None,
         "nada": "baik" if selisih_pd <= 0 else "buruk"},
        {"label": "IKS rata-rata penerima",
         "nilai": f"{disetujui['iks_skor'].mean():.2f}" if ada else "–",
         "catatan": f"{selisih_iks:+.2f} vs semua pemohon" if ada else None,
         "nada": "baik" if selisih_iks >= 0 else "buruk"},
    ]
    # Perbaikan temuan HE-08 (keparahan 4). Kolom target_macet hanya tersedia
    # karena berkas periode diambil dari data uji berlabel. Pada penyaluran
    # nyata metrik ini MUSTAHIL ada sebelum masa angsuran berjalan, sehingga
    # menampilkannya tanpa keterangan membuat pengguna menyangka sistem dapat
    # memastikan hasil di muka.
    # Pemohon input belum punya label macet, sehingga metrik evaluasi hanya
    # dihitung dari pemohon contoh yang berlabel.
    disetujui_label = disetujui[disetujui["target_macet"].notna()]
    semua_label = df_keputusan[df_keputusan["target_macet"].notna()]
    if par.get("mode_evaluasi") and len(disetujui_label):
        macet = disetujui_label["target_macet"].mean()
        selisih_macet = macet - semua_label["target_macet"].mean()
        butir.append({
            "label": "Macet aktual", "evaluasi": True,
            "nilai": f"{macet:.1%}".replace(".", ","),
            "catatan": f"{selisih_macet:+.1%} vs semua pemohon".replace(".", ","),
            "nada": "baik" if selisih_macet <= 0 else "buruk",
            "bantuan": "Hanya tersedia pada data uji berlabel. Pada penyaluran "
                       "nyata angka ini belum ada saat keputusan diambil. Pemohon "
                       f"input tidak ikut dihitung ({len(disetujui) - len(disetujui_label)} "
                       "penerima tanpa label).",
        })
    elif ada:
        butir.append({
            "label": "Macet aktual", "nilai": "–",
            "catatan": "di luar mode evaluasi",
            "bantuan": "Tidak tersedia di luar mode evaluasi atau tanpa pemohon berlabel.",
        })
    tampilan.pita(st, butir)

    tampilan.bagian(
        st, "Daftar keputusan", "Peringkat dan status pemohon",
        "Diurutkan dari skor tertinggi. Dana dibagikan menurut urutan ini "
        "sampai plafon habis.",
    )
    semua_status = ["DISETUJUI", "DITUNDA", "DITOLAK", "PERLU VERIFIKASI"]
    saring = st.multiselect("Tampilkan status", semua_status, default=semua_status)
    tampil = df_keputusan[df_keputusan["status_keputusan"].isin(saring)]

    tabel = (
        tampil[[
            "rank_TOPSIS", "id_pemohon", "sumber", "skor", "iks_skor", "kapasitas_angsur",
            "skor_riwayat", "skor_produktif", "pd_model", "nilai_pengajuan_rp",
            "status_keputusan", "alasan",
        ]].rename(columns={
            "rank_TOPSIS": "Peringkat", "id_pemohon": "ID", "sumber": "Sumber",
            "skor": "Skor",
            "iks_skor": "C1 IKS", "kapasitas_angsur": "C2 Kapasitas",
            "skor_riwayat": "C3 Riwayat", "skor_produktif": "C4 Produktif",
            "pd_model": "C5 PD", "nilai_pengajuan_rp": "C6 Nilai (Rp)",
            "status_keputusan": "Status", "alasan": "Alasan",
        }).sort_values("Peringkat")
    )
    st.dataframe(
        tampilan.warnai_status(tabel),
        use_container_width=True, hide_index=True, height=430,
        column_config={
            "Skor": st.column_config.NumberColumn(format="%.4f"),
            "C5 PD": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0),
            "C6 Nilai (Rp)": st.column_config.NumberColumn(format="%.0f"),
            "C2 Kapasitas": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    tampilan.bagian(
        st, "Rincian", "Mengapa pemohon ini di peringkat tersebut",
        "Pilih satu pemohon untuk melihat nilai tiap kriteria, alasan "
        "keputusan, dan beban angsurannya.",
    )
    daftar_id = tampil.sort_values("rank_TOPSIS")["id_pemohon"].tolist()
    if not daftar_id:
        st.info("Tidak ada pemohon dengan status yang dipilih.")
        return
    terakhir = st.session_state.get("terakhir")
    pilihan = st.selectbox(
        "Pilih pemohon", daftar_id,
        index=daftar_id.index(terakhir) if terakhir in daftar_id else 0,
    )
    baris = df_keputusan[df_keputusan["id_pemohon"] == pilihan].iloc[0]

    kiri, kanan = st.columns([1, 1])
    with kiri:
        st.markdown(f"**Penjelasan peringkat**\n\n{baris['penjelasan']}")
        st.markdown(
            f"**Keputusan** {tampilan.lencana(baris['status_keputusan'])} "
            f"{baris['alasan']}",
            unsafe_allow_html=True,
        )
        tampilkan_beban(baris)
        if baris.get("sumber") == "Input":
            tampilkan_verifikasi(baris)
    with kanan:
        rapor_pemohon(baris, df_keputusan)
        st.markdown(f"**Indikator deprivasi yang terpenuhi (IKS {int(baris['iks_skor'])}/{cfg.IKS_MAX})**")
        alasan_iks = knowledge.jelaskan_iks(baris)
        st.markdown("\n".join(f"- {a}" for a in alasan_iks) if alasan_iks
                    else "- Tidak ada indikator deprivasi terpenuhi")


def rapor_pemohon(r: pd.Series, df: pd.DataFrame) -> None:
    """Nilai C1-C6 satu pemohon dalam bentuk lembar rapor."""
    nilai_maks = float(df["nilai_pengajuan_rp"].max()) or 1.0
    tampilan.rapor(st, [
        {"kode": "C1", "nama": cfg.CRITERIA_NAMES["C1"],
         "nilai": f"{int(r['iks_skor'])} / {cfg.IKS_MAX}",
         "porsi": r["iks_skor"] / cfg.IKS_MAX, "benefit": True},
        {"kode": "C2", "nama": cfg.CRITERIA_NAMES["C2"],
         "nilai": f"{r['kapasitas_angsur']:.0f} / 100",
         "porsi": r["kapasitas_angsur"] / 100, "benefit": True},
        {"kode": "C3", "nama": cfg.CRITERIA_NAMES["C3"],
         "nilai": f"{int(r['skor_riwayat'])} / 4",
         "porsi": r["skor_riwayat"] / 4, "benefit": True},
        {"kode": "C4", "nama": cfg.CRITERIA_NAMES["C4"],
         "nilai": f"{int(r['skor_produktif'])} / 4",
         "porsi": r["skor_produktif"] / 4, "benefit": True},
        {"kode": "C5", "nama": cfg.CRITERIA_NAMES["C5"],
         "nilai": f"{r['pd_model']:.3f}",
         "porsi": r["pd_model"], "benefit": False},
        {"kode": "C6", "nama": cfg.CRITERIA_NAMES["C6"],
         "nilai": rupiah(r["nilai_pengajuan_rp"], singkat=True),
         "porsi": r["nilai_pengajuan_rp"] / nilai_maks, "benefit": False},
    ])


def tampilkan_beban(baris: pd.Series) -> None:
    """Angsuran, beban terhadap pendapatan, dan status kesejahteraan."""
    if pd.isna(baris.get("angsuran_bulanan")):
        return
    perkiraan = " (perkiraan dari jenis pekerjaan)" if baris.get("sumber") != "Input" else ""
    rasio = float(baris["rasio_angsuran_pendapatan"])
    st.markdown(
        f"**Angsuran:** {rupiah(baris['angsuran_bulanan'])}/bulan selama "
        f"{int(baris['durasi_bulan'])} bulan · **{rasio:.1%}** dari pendapatan "
        f"{rupiah(baris['pendapatan_rt_rp'])}{perkiraan} · "
        f"**Kesejahteraan:** {baris['status_kesejahteraan']}"
    )
    if rasio > 0.30:
        st.warning(
            f"Beban angsuran {rasio:.1%} melampaui batas kehati-hatian 30% "
            "pendapatan. Pertimbangkan tenor lebih panjang atau nilai lebih kecil."
        )


def tampilkan_verifikasi(baris: pd.Series) -> None:
    """Jaminan dan hasil verifikasi lapangan pemohon input."""
    st.markdown(
        f"**Jaminan:** penjamin {str(baris['penjamin_label']).lower()}, "
        f"aset {str(baris['kepemilikan_aset_label']).lower()}  \n"
        f"**Survei lapangan:** {baris['sudah_survei']} · "
        f"**Karakter:** {baris['karakter']}"
    )
    if baris.get("catatan_petugas"):
        st.markdown(f"**Catatan petugas:** {baris['catatan_petugas']}")
    if baris["karakter"] == "Kurang":
        st.error(
            "Penilaian karakter KURANG. Sistem tetap memeringkat berdasarkan "
            "enam kriteria, tetapi petugas perlu mempertimbangkan ini sebelum "
            "pencairan."
        )
    if baris["sudah_survei"] != "Sudah":
        st.warning(
            "Belum disurvei. Pemohon tidak memperoleh dana sampai data "
            "diverifikasi ke lapangan."
        )


# --------------------------------------------------------------------------
# TAB: INPUT PEMOHON
# --------------------------------------------------------------------------
def _kode_berikutnya(terpakai: set[str]) -> str:
    n = 1
    while f"BARU-{n:03d}" in terpakai:
        n += 1
    return f"BARU-{n:03d}"


def _id_terpakai(konteks_asal: dict) -> set[str]:
    return ({d["id_pemohon"] for d in st.session_state.pemohon_baru}
            | set(konteks_asal["df_periode"]["id_pemohon"]))


def tampilkan_hasil_terakhir(df_keputusan: pd.DataFrame | None) -> None:
    """Kartu hasil penilaian pemohon yang terakhir ditambahkan."""
    id_akhir = st.session_state.get("terakhir")
    if df_keputusan is None or id_akhir not in set(df_keputusan["id_pemohon"]):
        return
    r = df_keputusan[df_keputusan["id_pemohon"] == id_akhir].iloc[0]
    status = r["status_keputusan"]

    with st.container(border=True):
        st.markdown(f"#### Hasil penilaian {id_akhir}")
        st.markdown(
            f"{tampilan.lencana(status)} &nbsp;Peringkat "
            f"**{int(r['rank_TOPSIS'])}** dari {len(df_keputusan)} · {r['alasan']}",
            unsafe_allow_html=True,
        )

        lebih_aman = (df_keputusan["pd_model"] < r["pd_model"]).mean()
        rapor_pemohon(r, df_keputusan)
        st.caption(
            f"Risiko gagal bayar (C5) lebih tinggi daripada {lebih_aman:.0%} "
            "pemohon lain. Dibaca sebagai peringkat risiko relatif, bukan "
            "peluang pasti."
        )

        tampilkan_beban(r)
        tampilkan_verifikasi(r)
        for p in st.session_state.get("peringatan_terakhir", []):
            st.warning(p)
        with st.expander("Indikator kebutuhan sosial yang terpenuhi"):
            alasan = knowledge.jelaskan_iks(r)
            st.markdown("\n".join(f"- {x}" for x in alasan) if alasan
                        else "- Tidak ada indikator deprivasi terpenuhi")


def tab_input(konteks_asal: dict, df_keputusan: pd.DataFrame | None) -> None:
    daftar = st.session_state.pemohon_baru
    tampilan.bagian(
        st, "Berkas pengajuan", "Input pemohon baru",
        "Isi data berkas pengajuan. Sistem menghitung C1–C6 dengan aturan dan "
        "model yang sama persis dengan pemohon lain, lalu memasukkannya ke "
        "peringkat. Identitas cukup berupa kode, tanpa nama maupun NIK.",
    )

    pesan_csv = st.session_state.pop("pesan_csv", None)
    if pesan_csv:
        n_sah, galat_csv = pesan_csv
        if n_sah:
            st.success(f"{n_sah} pemohon dari CSV berhasil ditambahkan.")
        if galat_csv:
            st.error("Baris yang ditolak:\n\n" + "\n".join(f"- {g}" for g in galat_csv))

    tampilkan_hasil_terakhir(df_keputusan)

    kode = _kode_berikutnya(_id_terpakai(konteks_asal))
    with st.form("form_pemohon"):
        st.markdown(f"**Kode pemohon:** `{kode}` (otomatis)")

        st.markdown("##### 1. Data diri dan rumah tangga")
        a, b, c = st.columns(3)
        usia = a.number_input("Usia (tahun)", inp.USIA_MIN, inp.USIA_MAKS, 35)
        jenis_kelamin = b.selectbox("Jenis kelamin", inp.JENIS_KELAMIN)
        status_nikah = c.selectbox("Status pernikahan", inp.STATUS_NIKAH, index=1)
        a, b, c = st.columns(3)
        anggota_rt = a.number_input("Jumlah anggota keluarga serumah", 1, 20, 4)
        tanggungan = b.number_input(
            "Jumlah tanggungan", 0, 20, 2,
            help="Orang yang biaya hidupnya ditanggung pemohon.")
        status_rumah = c.selectbox("Status rumah", list(inp.STATUS_RUMAH))
        a, b = st.columns(2)
        lama_domisili = a.selectbox(
            "Lama tinggal di alamat ini", list(inp.LAMA_DOMISILI), index=3)
        telepon = b.radio("Punya nomor HP aktif", list(inp.TELEPON), horizontal=True)

        st.markdown("##### 2. Pekerjaan dan keuangan")
        a, b, c = st.columns(3)
        jenis_pekerjaan = a.selectbox("Jenis pekerjaan", list(inp.JENIS_PEKERJAAN), index=1)
        lama_bekerja = b.selectbox("Lama bekerja", list(inp.LAMA_BEKERJA), index=2)
        pendapatan = c.number_input(
            "Pendapatan rumah tangga per bulan (Rp)", 0, 1_000_000_000,
            2_500_000, step=100_000)
        a, b = st.columns(2)
        rekening = a.selectbox("Rekening bank", list(inp.REKENING))
        tabungan = b.selectbox("Tabungan", list(inp.TABUNGAN))
        a, b, c = st.columns(3)
        cicilan_lain = a.selectbox("Cicilan di tempat lain", list(inp.CICILAN_LAIN))
        pinjaman_aktif = b.number_input(
            "Jumlah pinjaman berjalan (termasuk ini)", 1, 4, 1)
        riwayat = c.selectbox("Riwayat pinjaman", list(inp.RIWAYAT), index=2)

        st.markdown("##### 3. Pengajuan")
        a, b, c = st.columns(3)
        nilai_pinjaman = a.number_input(
            "Nilai pinjaman (Rp)", 0, 1_000_000_000, 3_000_000, step=50_000)
        tenor = b.number_input(
            "Jangka waktu (bulan)", inp.TENOR_MIN, inp.TENOR_MAKS, 12)
        tujuan = c.selectbox("Tujuan pinjaman", list(inp.TUJUAN))
        st.caption(
            "Angsuran per bulan dan beban terhadap pendapatan dihitung otomatis "
            "(anuitas, bunga 18% per tahun) setelah tombol ditekan."
        )

        st.markdown("##### 4. Jaminan")
        a, b = st.columns(2)
        penjamin = a.selectbox("Penjamin", list(inp.PENJAMIN))
        agunan = b.selectbox("Aset / agunan", list(inp.AGUNAN), index=3)

        st.markdown("##### 5. Verifikasi lapangan")
        a, b = st.columns(2)
        sudah_survei = a.radio(
            "Sudah disurvei ke lapangan?", inp.SUDAH_SURVEI, horizontal=True,
            help="Pemohon yang belum disurvei berstatus PERLU VERIFIKASI dan "
                 "tidak memperoleh dana.")
        karakter = b.radio("Penilaian karakter", inp.KARAKTER, horizontal=True)
        catatan = st.text_area("Catatan petugas (opsional)", "")

        kirim = st.form_submit_button(
            "Nilai pemohon", type="primary", use_container_width=True)

    if kirim:
        data = {
            "id_pemohon": kode, "usia": usia, "jenis_kelamin": jenis_kelamin,
            "status_nikah": status_nikah, "anggota_rt": anggota_rt,
            "tanggungan": tanggungan, "status_rumah": status_rumah,
            "lama_domisili": lama_domisili, "telepon": telepon,
            "jenis_pekerjaan": jenis_pekerjaan, "lama_bekerja": lama_bekerja,
            "pendapatan_rp": pendapatan, "rekening": rekening,
            "tabungan": tabungan, "cicilan_lain": cicilan_lain,
            "pinjaman_aktif": pinjaman_aktif, "riwayat": riwayat,
            "nilai_pinjaman_rp": nilai_pinjaman, "tenor_bulan": tenor,
            "tujuan": tujuan, "penjamin": penjamin, "agunan": agunan,
            "sudah_survei": sudah_survei, "karakter": karakter,
            "catatan": catatan,
        }
        galat = inp.validasi(data)
        if galat:
            st.error("Data belum bisa dinilai:\n\n" + "\n".join(f"- {g}" for g in galat))
        else:
            daftar.append(inp.rapikan(data))
            st.session_state.terakhir = kode
            st.session_state.peringatan_terakhir = inp.peringatan(data)
            st.rerun()

    # ---- Daftar pemohon input ------------------------------------------
    tampilan.bagian(
        st, "Daftar", f"Pemohon input ({len(daftar)})",
        "Daftar ini hanya tersimpan selama halaman terbuka. Unduh CSV sebelum "
        "menutup atau me-refresh halaman, lalu unggah kembali bila diperlukan.",
    )

    if daftar:
        tabel = pd.DataFrame(daftar, columns=inp.KOLOM_INPUT)
        if df_keputusan is not None:
            tabel = tabel.merge(
                df_keputusan[["id_pemohon", "rank_TOPSIS", "status_keputusan"]],
                on="id_pemohon", how="left",
            ).rename(columns={"rank_TOPSIS": "peringkat", "status_keputusan": "status"})
        st.dataframe(tabel, use_container_width=True, hide_index=True)

        a, b = st.columns(2)
        with a:
            hapus = st.multiselect(
                "Pilih pemohon yang akan dihapus", [d["id_pemohon"] for d in daftar])
            if st.button("Hapus terpilih", disabled=not hapus):
                st.session_state.pemohon_baru = [
                    d for d in daftar if d["id_pemohon"] not in hapus]
                st.rerun()
        with b:
            st.download_button(
                "Unduh daftar pemohon input (CSV)",
                pd.DataFrame(daftar, columns=inp.KOLOM_INPUT)
                .to_csv(index=False).encode("utf-8"),
                file_name=f"pemohon_input_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv", use_container_width=True,
            )
            yakin = st.checkbox("Saya yakin menghapus semua pemohon input")
            if st.button("Hapus semua", disabled=not yakin):
                st.session_state.pemohon_baru = []
                st.session_state.pop("terakhir", None)
                st.rerun()

    # ---- Unggah CSV ----------------------------------------------------
    with st.expander("Tambah banyak pemohon sekaligus dari CSV"):
        st.markdown(
            "1. Unduh templat, 2. isi satu baris per pemohon (bisa di Excel), "
            "3. unggah kembali. Isian pilihan harus persis sama dengan daftar "
            "di bawah. Kolom `id_pemohon` boleh dikosongkan."
        )
        st.download_button(
            "Unduh templat CSV", inp.templat_csv(),
            file_name="templat_pemohon.csv", mime="text/csv",
        )
        st.dataframe(inp.petunjuk_pilihan(), use_container_width=True, hide_index=True)
        berkas = st.file_uploader("Unggah CSV pemohon", type="csv")
        if berkas is not None and st.button("Tambahkan dari CSV", type="primary"):
            sah, galat_csv = inp.baca_csv(berkas)
            terpakai = _id_terpakai(konteks_asal)
            for d in sah:
                if not d["id_pemohon"] or d["id_pemohon"] in terpakai:
                    d["id_pemohon"] = _kode_berikutnya(terpakai)
                terpakai.add(d["id_pemohon"])
                st.session_state.pemohon_baru.append(d)
            st.session_state.pesan_csv = (len(sah), galat_csv)
            st.rerun()


# --------------------------------------------------------------------------
# TAB: BOBOT
# --------------------------------------------------------------------------
def tab_bobot(hasil_ahp, bobot_pakai: np.ndarray) -> None:
    kiri, kanan = st.columns([1.1, 1])

    with kiri:
        tampilan.bagian(
            st, "Penilaian pengurus", "Matriks perbandingan berpasangan",
            "Gabungan tiga responden dengan rata-rata geometrik.",
        )
        st.dataframe(
            pd.DataFrame(
                hasil_ahp.matriks,
                index=cfg.CRITERIA_CODES, columns=cfg.CRITERIA_CODES,
            ).round(3),
            use_container_width=True,
        )
        warna = "normal" if hasil_ahp.konsisten else "inverse"
        st.metric(
            "Consistency Ratio", f"{hasil_ahp.cr:.4f}",
            f"ambang {cfg.CR_THRESHOLD}", delta_color=warna,
        )
        st.caption(
            f"lambda_max = {hasil_ahp.lambda_max:.4f} · CI = {hasil_ahp.ci:.4f}. "
            "CR di atas 0,10 berarti penilaian saling bertentangan dan "
            "wawancara harus diulang, bukan matriksnya yang disunting."
        )

    with kanan:
        tampilan.bagian(
            st, "Bobot", "AHP dibanding bobot yang dipakai",
            "Bobot dipakai mengikuti penggeser di sidebar.",
        )
        banding = pd.DataFrame({
            "Kriteria": [f"{k} {cfg.CRITERIA_NAMES[k]}" for k in cfg.CRITERIA_CODES],
            "Bobot AHP": hasil_ahp.bobot,
            "Bobot dipakai": bobot_pakai,
            "Selisih": bobot_pakai - hasil_ahp.bobot,
        })
        st.dataframe(
            banding.round(4), use_container_width=True, hide_index=True,
            column_config={
                "Bobot dipakai": st.column_config.ProgressColumn(
                    format="%.4f", min_value=0.0, max_value=0.5),
            },
        )
        st.bar_chart(
            banding.set_index("Kriteria")[["Bobot AHP", "Bobot dipakai"]],
            color=["#B8B1A2", tampilan.PALET[0]],
        )


# --------------------------------------------------------------------------
# TAB: SENSITIVITAS
# --------------------------------------------------------------------------
def tab_sensitivitas(konteks: dict, par: dict) -> None:
    tampilan.bagian(
        st, "Sensitivitas", "Ketahanan peringkat terhadap perubahan bobot",
        "Bobot satu kriteria digeser, bobot lain disesuaikan, lalu "
        "peringkat dihitung ulang.",
        penjelasan="Sumbu tegak adalah peringkat; garis yang menukik ke bawah "
                   "berarti peringkat memburuk. Garis yang saling berpotongan "
                   "menandakan pembalikan peringkat (rank reversal).",
    )

    kode = st.selectbox(
        "Kriteria yang divariasikan",
        cfg.CRITERIA_CODES,
        format_func=lambda k: f"{k} · {cfg.CRITERIA_NAMES[k]}",
    )
    n_teratas = st.slider("Batas 'n besar' yang dipantau", 5, 20, 10)
    idx = cfg.CRITERIA_CODES.index(kode)

    sens = sensitivity.analisis_sensitivitas_bobot(
        konteks["matriks"], par["bobot"], konteks["benefit"],
        konteks["id_alternatif"], idx,
    )

    teratas = (
        sens[sens["bobot"] == sens["bobot"].min()]
        .nsmallest(8, "peringkat")["id_pemohon"].tolist()
    )
    pivot = (
        sens[sens["id_pemohon"].isin(teratas)]
        .pivot(index="bobot", columns="id_pemohon", values="peringkat")
    )
    st.line_chart(
        pivot, height=340,
        color=tampilan.PALET[: len(pivot.columns)] or None,
    )

    kiri, kanan = st.columns(2)
    with kiri:
        st.markdown(f"**Titik balik komposisi {n_teratas} besar**")
        titik = sensitivity.titik_balik_peringkat(sens, n_teratas=n_teratas)
        if len(titik):
            st.dataframe(titik, use_container_width=True, hide_index=True)
        else:
            st.success(
                f"Komposisi {n_teratas} besar tidak berubah pada seluruh "
                "rentang bobot yang diuji. Hasil sangat stabil."
            )
    with kanan:
        st.markdown("**Stabilitas tiap pemohon**")
        st.dataframe(
            sensitivity.stabilitas_peringkat_teratas(sens, n_teratas=n_teratas).head(12),
            use_container_width=True, hide_index=True,
        )


# --------------------------------------------------------------------------
# TAB: WHAT-IF DAN GOAL SEEKING
# --------------------------------------------------------------------------
def tab_whatif(konteks: dict, par: dict, df_keputusan: pd.DataFrame) -> None:
    kiri, kanan = st.columns(2)

    with kiri:
        tampilan.bagian(
            st, "Simulasi", "What-if",
            "Ubah nilai satu kriteria pada satu pemohon, lalu lihat dampaknya "
            "pada peringkat.",
            penjelasan="Seluruh alternatif dihitung ulang karena TOPSIS "
                       "bersifat relatif terhadap solusi ideal.",
        )
        id_target = st.selectbox(
            "Pemohon", konteks["id_alternatif"], key="wi_id",
        )
        kode = st.selectbox(
            "Kriteria", cfg.CRITERIA_CODES, key="wi_krit",
            format_func=lambda k: f"{k} · {cfg.CRITERIA_NAMES[k]}",
        )
        idx_krit = cfg.CRITERIA_CODES.index(kode)
        idx_alt = konteks["id_alternatif"].index(id_target)
        nilai_kini = float(konteks["matriks"][idx_alt, idx_krit])

        kolom = konteks["matriks"][:, idx_krit]
        bawah, atas = float(kolom.min()), float(kolom.max())
        # Slider Streamlit gagal bila batas bawah sama dengan batas atas, yang
        # bisa terjadi bila hanya sedikit pemohon input dengan nilai kembar.
        if atas <= bawah:
            atas = bawah + max(abs(bawah), 1.0)
        nilai_baru = st.slider(
            f"Nilai baru {kode} (sekarang {nilai_kini:.4f})",
            bawah, atas, nilai_kini,
            key="wi_nilai",
        )

        hasil = sensitivity.what_if_pemohon(
            konteks["matriks"], par["bobot"], konteks["benefit"],
            konteks["id_alternatif"], id_target, {idx_krit: nilai_baru},
        )
        a, b, c = st.columns(3)
        a.metric("Peringkat sebelum", hasil["peringkat_sebelum"])
        b.metric(
            "Peringkat sesudah", hasil["peringkat_sesudah"],
            f"{hasil['delta_peringkat']:+d}",
        )
        c.metric("Alternatif lain bergeser", hasil["alternatif_lain_bergeser"])

    with kanan:
        tampilan.bagian(
            st, "Simulasi", "Goal seeking",
            "Tetapkan peringkat yang diinginkan, lalu sistem mencari nilai "
            "kriteria minimum yang dibutuhkan untuk mencapainya.",
        )
        id_gs = st.selectbox("Pemohon", konteks["id_alternatif"], key="gs_id")
        kode_gs = st.selectbox(
            "Kriteria yang boleh diubah", cfg.CRITERIA_CODES, key="gs_krit",
            format_func=lambda k: f"{k} · {cfg.CRITERIA_NAMES[k]}",
        )
        target = st.number_input(
            "Peringkat target", 1, len(konteks["id_alternatif"]), 5, key="gs_target",
        )

        if st.button("Cari nilai yang dibutuhkan", use_container_width=True):
            hasil_gs = sensitivity.goal_seeking(
                konteks["matriks"], par["bobot"], konteks["benefit"],
                konteks["id_alternatif"], id_gs,
                cfg.CRITERIA_CODES.index(kode_gs), int(target),
            )
            if hasil_gs["tercapai"]:
                st.success(
                    f"{kode_gs} harus bergeser dari "
                    f"{hasil_gs['nilai_awal']:.4f} ke "
                    f"{hasil_gs['nilai_dibutuhkan']:.4f} "
                    f"(selisih {hasil_gs['selisih']:+.4f}) agar mencapai "
                    f"peringkat {hasil_gs['peringkat_tercapai']}."
                )
            else:
                st.warning(hasil_gs["catatan"])


# --------------------------------------------------------------------------
# TAB: VALIDASI
# --------------------------------------------------------------------------
def tab_validasi(konteks: dict, df_keputusan: pd.DataFrame, par: dict) -> None:
    tampilan.bagian(
        st, "Validasi", "Perbandingan dengan kebijakan alternatif",
        "Seluruh kebijakan diuji pada plafon dan batas PD yang sama persis.",
        penjelasan="Kebijakan yang unggul pada kedua kolom sekaligus (tingkat "
                   "macet dan IKS penerima) adalah kebijakan yang benar-benar "
                   "lebih baik, bukan yang sekadar menukar satu tujuan dengan "
                   "tujuan lain.",
    )
    # Perbandingan kebijakan menghitung tingkat macet sebenarnya, sehingga
    # hanya pemohon contoh yang berlabel yang dapat diikutkan.
    df_label = df_keputusan[df_keputusan["target_macet"].notna()]
    n_tanpa_label = len(df_keputusan) - len(df_label)
    if len(df_label) < 2:
        st.info(
            "Perbandingan kebijakan membutuhkan pemohon contoh yang sudah "
            "diketahui lancar atau macet. Pilih sumber pemohon "
            f"'{SUMBER_GABUNG}' pada sidebar."
        )
    else:
        tabel = evaluasi.bandingkan_kebijakan(
            df_label, plafon=par["plafon"], pd_batas=par["pd_batas"],
        )
        st.dataframe(tabel, use_container_width=True)
        if n_tanpa_label:
            st.caption(
                f"{n_tanpa_label} pemohon input tidak diikutkan karena hasil "
                "angsurannya belum diketahui."
            )

    tampilan.bagian(
        st, "Model prediktif", "Kinerja pada data uji",
        "Data uji dipakai satu kali, setelah model dan ambang dipilih.",
    )
    hasil_model = konteks["hasil_model"]
    m = hasil_model.metrik_uji

    a, b, c, d = st.columns(4)
    a.metric("AUC-ROC (uji)", f"{m['auc']:.4f}")
    b.metric("Recall macet", f"{m['recall']:.4f}")
    c.metric("Brier score", f"{m['brier']:.4f}")
    d.metric(
        "Biaya", f"{m['biaya']:.0f}",
        f"{m['biaya'] - m['biaya_loloskan_semua']:.0f} vs tanpa model",
        delta_color="inverse",
    )

    kiri, kanan = st.columns(2)
    kiri.markdown("**Perbandingan kandidat model (CV pada data latih)**")
    kiri.dataframe(
        hasil_model.perbandingan_model.round(4),
        use_container_width=True, hide_index=True,
    )
    kanan.markdown("**Korelasi Spearman antar metode MCDM**")
    kanan.dataframe(konteks["korelasi_rank"], use_container_width=True)


# --------------------------------------------------------------------------
# TAB: RIWAYAT
# --------------------------------------------------------------------------
def tab_riwayat(df_keputusan: pd.DataFrame, par: dict, hasil_ahp) -> None:
    tampilan.bagian(
        st, "Riwayat", "Skenario keputusan tersimpan",
        "Skenario, bobot, dan hasilnya dapat ditelusuri kembali.",
        penjelasan="Penyimpanan riwayat inilah yang memisahkan SPK dari "
                   "kalkulator sekali pakai.",
    )

    catatan = st.text_input("Catatan skenario", "Skenario uji dari antarmuka")
    if st.button("Simpan skenario ini", type="primary"):
        with sqlite3.connect(cfg.DB_PATH) as conn:
            id_skenario = datetime.now().strftime("SKN-%Y%m%d-%H%M%S")
            conn.execute(
                "INSERT INTO skenario VALUES (?,?,?,?,?,?)",
                (
                    id_skenario, datetime.now().isoformat(timespec="seconds"),
                    float(par["plafon"]), float(hasil_ahp.cr),
                    json.dumps(dict(zip(
                        cfg.CRITERIA_CODES, par["bobot"].round(6).tolist()))),
                    catatan,
                ),
            )
            conn.executemany(
                "INSERT INTO keputusan VALUES (?,?,?,?,?,?,?,?)",
                [
                    (id_skenario, r["id_pemohon"], int(r["rank_TOPSIS"]),
                     float(r["skor"]), float(r["pd_model"]), int(r["iks_skor"]),
                     r["status_keputusan"], float(r["nominal_disetujui"]))
                    for _, r in df_keputusan.iterrows()
                ],
            )
        st.success(f"Tersimpan sebagai {id_skenario}")

    try:
        with sqlite3.connect(cfg.DB_PATH) as conn:
            skenario = pd.read_sql_query(
                "SELECT * FROM skenario ORDER BY waktu DESC LIMIT 25", conn
            )
        st.dataframe(skenario, use_container_width=True, hide_index=True)
    except Exception as e:
        st.info(f"Belum ada riwayat tersimpan. ({e})")

    st.download_button(
        "Unduh keputusan periode ini (CSV)",
        df_keputusan.to_csv(index=False).encode("utf-8"),
        file_name=f"keputusan_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------
# UTAMA
# --------------------------------------------------------------------------
def main() -> None:
    tampilan.pasang_gaya(st)

    konteks = muat_pipeline()
    st.session_state.setdefault("pemohon_baru", [])
    hasil_ahp = konteks["hasil_ahp"]
    par = gambar_sidebar(hasil_ahp.bobot)

    # Konteks aktif = kohort contoh dan/atau pemohon input, sesuai pilihan
    # sidebar. Seluruh tab analisis membaca konteks ini.
    aktif = susun_konteks_aktif(konteks, par["sumber"])
    cukup = len(aktif["id_alternatif"]) >= 2

    diajukan = float(aktif["df_periode"]["nilai_pengajuan_rp"].sum())
    tampilan.kop(
        st,
        atas="Koperasi Simpan Pinjam Sejahtera Mandiri · Surakarta",
        judul="Prioritas Penyaluran Kredit Mikro",
        sub="Sistem pendukung keputusan: risiko gagal bayar dari model "
            "prediktif, kebutuhan sosial dari kriteria BPS, bobot dari "
            "penilaian pengurus (AHP), diperingkat dengan "
            f"{par['metode']}.",
        fakta=[
            ("Plafon periode", rupiah(par["plafon"], singkat=True)),
            ("Pemohon", str(len(aktif["id_alternatif"]))),
            ("Total diajukan", rupiah(diajukan, singkat=True)),
            ("Permintaan / plafon",
             f"{diajukan / par['plafon']:.2f}x".replace(".", ",")),
            ("CR AHP", f"{hasil_ahp.cr:.3f}".replace(".", ",")),
        ],
    )

    df_keputusan = None
    if cukup:
        df_rank = hitung_ulang(aktif, par["bobot"], par["metode"])
        df_keputusan = pipeline.alokasikan_plafon(
            df_rank, plafon=par["plafon"], pd_batas=par["pd_batas"],
        )

        # Penjelasan peringkat dihitung ulang karena bobot pengguna mengubah
        # kontribusi tiap kriteria terhadap skor akhir.
        hasil_topsis = ranking.topsis(aktif["matriks"], par["bobot"], aktif["benefit"])
        peta_idx = {pid: i for i, pid in enumerate(aktif["id_alternatif"])}
        df_keputusan["total_alternatif"] = len(df_keputusan)
        df_keputusan["penjelasan"] = [
            pipeline.jelaskan_peringkat(
                r, hasil_topsis["matriks_terbobot"], peta_idx[r["id_pemohon"]])
            for _, r in df_keputusan.iterrows()
        ]

    if not hasil_ahp.konsisten:
        st.error(
            f"Matriks AHP tidak konsisten (CR = {hasil_ahp.cr:.4f}). "
            "Hasil pemeringkatan tidak layak dipakai."
        )

    def butuh_pemohon() -> bool:
        if not cukup:
            st.info(
                "Pemeringkatan membutuhkan minimal 2 pemohon. Tambahkan pemohon "
                f"di tab Input Pemohon, atau pilih '{SUMBER_GABUNG}' pada sidebar."
            )
        return cukup

    tabs = st.tabs([
        "Keputusan", "Input Pemohon", "Bobot & AHP", "Sensitivitas",
        "What-if & Goal Seeking", "Validasi", "Riwayat",
    ])
    with tabs[0]:
        if butuh_pemohon():
            tab_keputusan(df_keputusan, par)
    with tabs[1]:
        tab_input(konteks, df_keputusan)
    with tabs[2]:
        tab_bobot(hasil_ahp, par["bobot"])
    with tabs[3]:
        if butuh_pemohon():
            tab_sensitivitas(aktif, par)
    with tabs[4]:
        if butuh_pemohon():
            tab_whatif(aktif, par, df_keputusan)
    with tabs[5]:
        if butuh_pemohon():
            tab_validasi(aktif, df_keputusan, par)
    with tabs[6]:
        if butuh_pemohon():
            tab_riwayat(df_keputusan, par, hasil_ahp)

    tampilan.kaki(
        st,
        "Sistem ini bersifat decision support, bukan decision automation. "
        "Keputusan akhir tetap berada pada pengelola koperasi. "
        "Sumber data: UCI Statlog German Credit (id=144), lisensi CC BY 4.0.",
    )


if __name__ == "__main__":
    main()
