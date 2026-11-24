"""
Penilaian pemohon baru yang diisi petugas lewat formulir antarmuka.

Formulir memakai bahasa sehari-hari dan satuan Rupiah, sedangkan model
prediktif dilatih pada 20 atribut berkode Statlog German Credit (A11, A34,
dan seterusnya). Modul ini menjadi jembatan keduanya: menerjemahkan jawaban
formulir ke kode atribut, menghitung C1 sampai C6 dengan fungsi yang sama
persis dengan pipeline, lalu menambahkan informasi yang tidak ada di dataset
(pendapatan terlapor, jumlah anggota keluarga, jaminan, verifikasi lapangan).

Isian verifikasi lapangan dan penilaian karakter merupakan masukan responden
wawancara AHP (Ketua Pengurus menyebut "karakter peminjam", dua responden
menyesali penyaluran yang keterangan pemohonnya tidak sesuai fakta). Isian
tersebut dipakai sebagai aturan pendukung, TIDAK sebagai kriteria ketujuh,
karena menambah kriteria membatalkan bobot AHP hasil wawancara.

Logika dipisahkan dari app/app.py agar dapat diuji tanpa Streamlit.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import eksternal as ext
from . import etl, knowledge, lokalisasi, ml_model

# --------------------------------------------------------------------------
# KONVERSI AMBANG DEUTSCHE MARK KE RUPIAH
# --------------------------------------------------------------------------
# Kategori saldo rekening (A1) dan tabungan (A6) didefinisikan dalam DM.
# Ambangnya diterjemahkan dengan rasio median yang sama dengan pelokalan
# nilai pinjaman: median pinjaman historis 2.320 DM dipetakan ke median
# sebaran rujukan koperasi Rp3.350.000 (lihat docs/PERHITUNGAN_KRITERIA.md
# bagian C6). Dengan begitu satu "DM" bermakna sama di seluruh formulir.
MEDIAN_PINJAMAN_DM = 2_320
MEDIAN_PINJAMAN_RP = 3_350_000
RP_PER_DM = MEDIAN_PINJAMAN_RP / MEDIAN_PINJAMAN_DM


def _rp_dari_dm(dm: float) -> int:
    """Ubah ambang DM ke Rupiah, dibulatkan ke kelipatan Rp5.000."""
    return int(round(dm * RP_PER_DM / 5_000) * 5_000)


def _fmt(rp: float) -> str:
    return f"Rp{rp:,.0f}".replace(",", ".")


_RP_100 = _fmt(_rp_dari_dm(100))
_RP_200 = _fmt(_rp_dari_dm(200))
_RP_500 = _fmt(_rp_dari_dm(500))
_RP_1000 = _fmt(_rp_dari_dm(1000))

# --------------------------------------------------------------------------
# PILIHAN FORMULIR -> KODE ATRIBUT
# --------------------------------------------------------------------------
JENIS_KELAMIN = ["Laki-laki", "Perempuan"]
STATUS_NIKAH = ["Belum menikah", "Menikah", "Cerai / janda / duda"]

STATUS_RUMAH = {
    "Milik sendiri": "A152",
    "Sewa / kontrak": "A151",
    "Menumpang": "A153",
}
LAMA_DOMISILI = {
    "1 tahun atau kurang": 1,
    "2 tahun": 2,
    "3 tahun": 3,
    "4 tahun atau lebih": 4,
}
TELEPON = {"Ya": "A192", "Tidak": "A191"}

JENIS_PEKERJAAN = {
    "Tidak bekerja": "A171",
    "Buruh harian / pedagang kecil": "A172",
    "Karyawan / pekerja terampil": "A173",
    "Wiraswasta / pemilik usaha / tenaga ahli": "A174",
}
LAMA_BEKERJA = {
    "Tidak bekerja": "A71",
    "Kurang dari 1 tahun": "A72",
    "1 - 4 tahun": "A73",
    "4 - 7 tahun": "A74",
    "7 tahun atau lebih": "A75",
}
REKENING = {
    "Tidak punya rekening bank": "A14",
    "Saldo minus / ada tunggakan": "A11",
    f"Saldo di bawah {_RP_200}": "A12",
    f"Saldo {_RP_200} atau lebih": "A13",
}
TABUNGAN = {
    "Tidak punya / tidak diketahui": "A65",
    f"Kurang dari {_RP_100}": "A61",
    f"{_RP_100} - {_RP_500}": "A62",
    f"{_RP_500} - {_RP_1000}": "A63",
    f"{_RP_1000} atau lebih": "A64",
}
CICILAN_LAIN = {
    "Tidak ada": "A143",
    "Di bank / koperasi lain": "A141",
    "Kredit barang di toko": "A142",
}
# Label A34 sengaja mengikuti definisi sumber apa adanya. Skor C3 untuk kode
# ini tertinggi karena kalibrasi data latih (efek thin file), bukan karena
# "bermasalah" dinilai baik. Lihat docs/CATATAN_ITERASI.md iterasi 2.
RIWAYAT = {
    "Belum pernah meminjam / semua lunas": "A30",
    "Semua pinjaman di koperasi ini lunas": "A31",
    "Pinjaman berjalan, pembayaran lancar": "A32",
    "Pernah menunggak": "A33",
    "Akun kritis / punya pinjaman di lembaga lain": "A34",
}
TUJUAN = {
    "Modal usaha": "A49",
    "Pelatihan kerja": "A48",
    "Pendidikan": "A46",
    "Kendaraan bekas": "A41",
    "Perbaikan rumah": "A45",
    "Perabot / peralatan": "A42",
    "Kendaraan baru": "A40",
    "Alat rumah tangga": "A44",
    "Elektronik": "A43",
    "Lainnya": "A410",
}
PENJAMIN = {
    "Tidak ada": "A101",
    "Pengaju bersama": "A102",
    "Ada penjamin": "A103",
}
AGUNAN = {
    "Tanah / rumah bersertifikat": "A121",
    "Tabungan berjangka / asuransi": "A122",
    "Kendaraan (BPKB) / aset lain": "A123",
    "Tidak ada": "A124",
}
SUDAH_SURVEI = ["Belum", "Sudah"]
KARAKTER = ["Baik", "Cukup", "Kurang"]

# Atribut A20 tidak relevan bagi koperasi di Indonesia dan tidak ditanyakan.
# Diisi nilai mayoritas data latih (96,3 persen) agar model tetap menerima
# masukan yang dikenalnya.
PEKERJA_ASING_BAWAAN = "A201"

# Ambang rasio angsuran terhadap pendapatan untuk menurunkan A8 (skala 1-4).
# Arah mengikuti asumsi proyek, 1 = beban teringan, dan sejalan dengan data:
# tingkat macet naik dari 25 persen pada level 1 ke 33 persen pada level 4.
# Level 4 dimulai di batas kehati-hatian 30 persen (ext.BATAS_RASIO_ANGSURAN).
AMBANG_A8 = (0.10, 0.20, ext.BATAS_RASIO_ANGSURAN)

TENOR_MIN, TENOR_MAKS = 3, 60
USIA_MIN, USIA_MAKS = 18, 75
PINJAMAN_MIN = 100_000

# Urutan kolom CSV unggah/unduh. Sama dengan kunci kamus isian formulir.
KOLOM_INPUT = [
    "id_pemohon", "usia", "jenis_kelamin", "status_nikah", "anggota_rt",
    "tanggungan", "status_rumah", "lama_domisili", "telepon",
    "jenis_pekerjaan", "lama_bekerja", "pendapatan_rp", "rekening",
    "tabungan", "cicilan_lain", "pinjaman_aktif", "riwayat",
    "nilai_pinjaman_rp", "tenor_bulan", "tujuan", "penjamin", "agunan",
    "sudah_survei", "karakter", "catatan",
]
KOLOM_ANGKA = [
    "usia", "anggota_rt", "tanggungan", "pendapatan_rp", "pinjaman_aktif",
    "nilai_pinjaman_rp", "tenor_bulan",
]
PILIHAN_PER_KOLOM = {
    "jenis_kelamin": JENIS_KELAMIN,
    "status_nikah": STATUS_NIKAH,
    "status_rumah": list(STATUS_RUMAH),
    "lama_domisili": list(LAMA_DOMISILI),
    "telepon": list(TELEPON),
    "jenis_pekerjaan": list(JENIS_PEKERJAAN),
    "lama_bekerja": list(LAMA_BEKERJA),
    "rekening": list(REKENING),
    "tabungan": list(TABUNGAN),
    "cicilan_lain": list(CICILAN_LAIN),
    "riwayat": list(RIWAYAT),
    "tujuan": list(TUJUAN),
    "penjamin": list(PENJAMIN),
    "agunan": list(AGUNAN),
    "sudah_survei": SUDAH_SURVEI,
    "karakter": KARAKTER,
}


# --------------------------------------------------------------------------
# PEMETAAN
# --------------------------------------------------------------------------
def kode_status_personal(jenis_kelamin: str, status_nikah: str) -> str:
    """
    Gabungkan jenis kelamin dan status nikah menjadi kode A9.

    Kode A95 (perempuan lajang) tidak pernah muncul pada data latih. Encoder
    model memperlakukan kode yang tidak dikenal sama dengan kategori acuan
    (A91, pria cerai), sehingga seluruh pemohon perempuan dipetakan ke A92,
    kode perempuan satu-satunya yang dikenal model.
    """
    if jenis_kelamin == "Perempuan":
        return "A92"
    return {
        "Belum menikah": "A93",
        "Menikah": "A94",
        "Cerai / janda / duda": "A91",
    }[status_nikah]


def level_rasio_angsuran(rasio: float) -> int:
    """Turunkan A8 (1-4) dari rasio angsuran terhadap pendapatan sungguhan."""
    for level, batas in enumerate(AMBANG_A8, start=1):
        if rasio <= batas:
            return level
    return 4


def kode_tanggungan(tanggungan: int) -> int:
    """
    Turunkan A18 (1 atau 2) dari jumlah tanggungan.

    Atribut asli hanya punya dua tingkat, dan 84,5 persen data bernilai 1.
    Tingkat macet kedua kelompok sama (30 persen), sehingga pengaruhnya
    terhadap model kecil apa pun ambang yang dipilih.
    """
    return 1 if tanggungan <= 2 else 2


def petakan_rp_ke_historis(
    nilai_rp: np.ndarray, sebaran_historis: np.ndarray
) -> np.ndarray:
    """
    Kebalikan lokalisasi.petakan_nilai_pinjaman: Rupiah ke skala data latih.

    Persentil nilai pada sebaran rujukan Rupiah dicari, lalu diambil nilai
    pada persentil yang sama di sebaran historis. Model membutuhkan skala
    historis karena dilatih dengan satuan tersebut.
    """
    rujukan_rp = np.sort(lokalisasi.sebaran_rujukan_pinjaman())
    persentil = np.searchsorted(rujukan_rp, np.asarray(nilai_rp, dtype=float))
    persentil = np.clip(persentil / len(rujukan_rp), 0.001, 0.999)
    return np.round(np.quantile(np.asarray(sebaran_historis, dtype=float), persentil))


# --------------------------------------------------------------------------
# VALIDASI
# --------------------------------------------------------------------------
_POLA_RIBUAN = re.compile(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")


def _ke_angka(v) -> float | None:
    """
    Baca angka dari isian formulir atau sel CSV.

    Titik hanya dianggap pemisah ribuan bila polanya memang ribuan
    ("2.500.000"). Tanpa pemeriksaan ini, "2500000.0" hasil ekspor pandas
    akan terbaca dua puluh lima juta.
    """
    if v is None:
        return None
    if not isinstance(v, str):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    s = v.strip().replace("Rp", "").replace(" ", "")
    if _POLA_RIBUAN.match(s):
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def rapikan(data: dict) -> dict:
    """Seragamkan tipe isian, terutama untuk baris yang berasal dari CSV."""
    hasil = {k: data.get(k) for k in KOLOM_INPUT}
    for k in KOLOM_ANGKA:
        hasil[k] = _ke_angka(hasil.get(k))
    for k in KOLOM_INPUT:
        if k not in KOLOM_ANGKA:
            v = hasil.get(k)
            hasil[k] = "" if v is None or (isinstance(v, float) and np.isnan(v)) \
                else str(v).strip()
    return hasil


def validasi(data: dict) -> list[str]:
    """Kembalikan daftar kesalahan. Daftar kosong berarti isian sah."""
    d = rapikan(data)
    galat = []

    def ribuan(x: float) -> str:
        return f"{x:,.0f}".replace(",", ".")

    def angka(kolom, nama, minimum, maksimum=None):
        v = d[kolom]
        if v is None or np.isnan(v):
            galat.append(f"{nama} wajib diisi dengan angka.")
        elif v < minimum or (maksimum is not None and v > maksimum):
            batas = ribuan(minimum) + (f" sampai {ribuan(maksimum)}" if maksimum else " atau lebih")
            galat.append(f"{nama} harus {batas}, diisi {ribuan(v)}.")

    angka("usia", "Usia", USIA_MIN, USIA_MAKS)
    angka("anggota_rt", "Jumlah anggota keluarga", 1, 20)
    angka("tanggungan", "Jumlah tanggungan", 0, 20)
    angka("pendapatan_rp", "Pendapatan per bulan", 1)
    angka("pinjaman_aktif", "Jumlah pinjaman berjalan", 1, 4)
    angka("nilai_pinjaman_rp", "Nilai pinjaman", PINJAMAN_MIN)
    angka("tenor_bulan", "Jangka waktu", TENOR_MIN, TENOR_MAKS)

    for kolom, pilihan in PILIHAN_PER_KOLOM.items():
        if d[kolom] not in pilihan:
            galat.append(
                f"Isian '{kolom}' bernilai '{d[kolom]}', harus salah satu dari: "
                + "; ".join(pilihan) + "."
            )

    if not galat and d["tanggungan"] >= d["anggota_rt"] + 1:
        galat.append("Jumlah tanggungan tidak boleh melebihi jumlah anggota keluarga.")
    return galat


def peringatan(data: dict) -> list[str]:
    """Hal yang sah tetapi perlu diperhatikan petugas."""
    d = rapikan(data)
    catatan = []
    maks = lokalisasi.KELIPATAN_UMK_MAKS * ext.UMK_SURAKARTA.nilai
    if d["nilai_pinjaman_rp"] and d["nilai_pinjaman_rp"] > maks:
        catatan.append(
            f"Nilai pinjaman di atas {_fmt(maks)}, batas atas sebaran rujukan "
            "koperasi mikro. Model memperlakukannya sama dengan pinjaman "
            "terbesar di data latih."
        )
    if d["jenis_pekerjaan"] == "Tidak bekerja" and d["lama_bekerja"] != "Tidak bekerja":
        catatan.append("Jenis pekerjaan 'Tidak bekerja' tetapi lama bekerja diisi.")
    return catatan


# --------------------------------------------------------------------------
# PENILAIAN
# --------------------------------------------------------------------------
def nilai_pemohon(daftar_data: list[dict], konteks: dict) -> pd.DataFrame:
    """
    Hitung seluruh kolom yang dibutuhkan pemeringkatan untuk pemohon baru.

    `konteks` adalah keluaran pipeline.jalankan(): dipakai model terlatih,
    data latih (kalibrasi C3), dan sebaran pinjaman historis (pemetaan C6).
    Pemanggil wajib memvalidasi isian lebih dahulu.
    """
    baris = []
    for data in daftar_data:
        d = rapikan(data)
        baris.append({
            "id_pemohon": d["id_pemohon"],
            "status_rekening": REKENING[d["rekening"]],
            "durasi_bulan": int(d["tenor_bulan"]),
            "riwayat_kredit": RIWAYAT[d["riwayat"]],
            "tujuan_pinjaman": TUJUAN[d["tujuan"]],
            "tabungan": TABUNGAN[d["tabungan"]],
            "lama_bekerja": LAMA_BEKERJA[d["lama_bekerja"]],
            "status_personal": kode_status_personal(d["jenis_kelamin"], d["status_nikah"]),
            "penjamin": PENJAMIN[d["penjamin"]],
            "lama_domisili": LAMA_DOMISILI[d["lama_domisili"]],
            "kepemilikan_aset": AGUNAN[d["agunan"]],
            "usia": int(d["usia"]),
            "cicilan_lain": CICILAN_LAIN[d["cicilan_lain"]],
            "status_rumah": STATUS_RUMAH[d["status_rumah"]],
            "jumlah_kredit_aktif": int(d["pinjaman_aktif"]),
            "jenis_pekerjaan": JENIS_PEKERJAAN[d["jenis_pekerjaan"]],
            "jumlah_tanggungan": kode_tanggungan(int(d["tanggungan"])),
            "telepon": TELEPON[d["telepon"]],
            "pekerja_asing": PEKERJA_ASING_BAWAAN,
            "label_risiko": np.nan,
            "nilai_pengajuan_rp": float(d["nilai_pinjaman_rp"]),
            "pendapatan_rt_rp": float(d["pendapatan_rp"]),
            "anggota_rt": int(d["anggota_rt"]),
            "sumber": "Input",
            "sudah_survei": d["sudah_survei"],
            "karakter": d["karakter"],
            "catatan_petugas": d["catatan"],
            "perlu_verifikasi": d["sudah_survei"] != "Sudah",
        })
    df = pd.DataFrame(baris)

    # Beban angsuran dari angka Rupiah sungguhan, lalu A8 diturunkan darinya.
    beban = lokalisasi.hitung_beban_angsuran(
        df["nilai_pengajuan_rp"].to_numpy(),
        df["durasi_bulan"].to_numpy(),
        df["pendapatan_rt_rp"].to_numpy(),
    )
    for kolom, nilai in beban.items():
        df[kolom] = nilai
    df["rasio_angsuran"] = [level_rasio_angsuran(r) for r in df["rasio_angsuran_pendapatan"]]

    sebaran_historis = konteks["df"]["jumlah_kredit"].to_numpy()
    df["jumlah_kredit"] = petakan_rp_ke_historis(df["nilai_pengajuan_rp"], sebaran_historis)
    df["jumlah_kredit_historis"] = df["jumlah_kredit"]

    # C1 sampai C5 memakai fungsi yang sama persis dengan pipeline.
    df = knowledge.terapkan_aturan_iks(df)
    df = etl.tambah_label_terbaca(df)
    peta_riwayat = knowledge.ambil_peta_skor_riwayat(konteks["df_latih"])
    df = knowledge.terapkan_kriteria_turunan(df, peta_riwayat)
    df["target_macet"] = np.nan   # belum diketahui sampai masa angsuran berjalan
    df["pd_model"] = ml_model.prediksi_pd(konteks["hasil_model"].pipeline, df)

    df["pengeluaran_per_kapita_rp"] = np.round(df["pendapatan_rt_rp"] / df["anggota_rt"])
    df["status_kesejahteraan"] = [
        ext.klasifikasi_kesejahteraan(p, n)
        for p, n in zip(df["pendapatan_rt_rp"], df["anggota_rt"])
    ]
    df["rasio_garis_kemiskinan"] = (
        df["pengeluaran_per_kapita_rp"] / ext.GARIS_KEMISKINAN_JATENG.nilai
    ).round(3)
    return df


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def contoh_isian(id_pemohon: str = "BARU-001") -> dict:
    """Satu baris contoh, dipakai sebagai nilai bawaan dan templat CSV."""
    return {
        "id_pemohon": id_pemohon, "usia": 35, "jenis_kelamin": "Perempuan",
        "status_nikah": "Menikah", "anggota_rt": 4, "tanggungan": 2,
        "status_rumah": "Sewa / kontrak", "lama_domisili": "4 tahun atau lebih",
        "telepon": "Ya", "jenis_pekerjaan": "Buruh harian / pedagang kecil",
        "lama_bekerja": "1 - 4 tahun", "pendapatan_rp": 2_500_000,
        "rekening": "Tidak punya rekening bank",
        "tabungan": "Tidak punya / tidak diketahui", "cicilan_lain": "Tidak ada",
        "pinjaman_aktif": 1, "riwayat": "Pinjaman berjalan, pembayaran lancar",
        "nilai_pinjaman_rp": 3_000_000, "tenor_bulan": 12, "tujuan": "Modal usaha",
        "penjamin": "Tidak ada", "agunan": "Tidak ada", "sudah_survei": "Sudah",
        "karakter": "Baik", "catatan": "",
    }


def templat_csv() -> bytes:
    """Templat unggah berisi satu baris contoh dan daftar pilihan sah."""
    contoh = pd.DataFrame([contoh_isian()], columns=KOLOM_INPUT)
    return contoh.to_csv(index=False).encode("utf-8")


def petunjuk_pilihan() -> pd.DataFrame:
    """Tabel pilihan sah per kolom, ditampilkan di samping templat."""
    return pd.DataFrame([
        {"Kolom": k, "Pilihan yang sah": " | ".join(v)}
        for k, v in PILIHAN_PER_KOLOM.items()
    ])


def baca_csv(berkas) -> tuple[list[dict], list[str]]:
    """Baca CSV unggahan. Kembalikan baris sah dan pesan kesalahan per baris."""
    df = pd.read_csv(berkas, dtype=str, keep_default_na=False)
    hilang = [k for k in KOLOM_INPUT if k not in df.columns and k != "catatan"]
    if hilang:
        return [], [f"Kolom tidak ditemukan: {', '.join(hilang)}. Gunakan templat."]

    sah, galat = [], []
    for i, baris in enumerate(df.to_dict(orient="records"), start=2):
        masalah = validasi(baris)
        if masalah:
            galat.append(f"Baris {i}: " + " ".join(masalah))
        else:
            sah.append(rapikan(baris))
    return sah, galat
