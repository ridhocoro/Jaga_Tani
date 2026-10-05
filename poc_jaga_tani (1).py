"""
PoC Jaga Tani - uji kelayakan rule engine penyiraman (LKP P6, Section 2)

Yang diuji:
  Apakah logika rule-based (kelembaban tanah + prakiraan hujan -> rekomendasi siram)
  masih jalan baik kalau sensornya noise, kadang mati, dan prakiraan cuacanya meleset?

Data: SINTETIS (simulasi neraca air tanah sederhana). Belum ada data lapangan asli,
jadi hasil di sini hanya membuktikan logika & alur pipeline, BUKAN akurasi di lahan nyata.
Nilai ambang kelembaban di bawah juga masih nilai awal yang harus divalidasi penyuluh.

Cara jalanin:  python poc_jaga_tani.py   (butuh numpy & pandas; bisa juga dipecah per sel di Colab)
"""
import json
import numpy as np
import pandas as pd

# %% ---------------------------------------------------------------
# 1. RULE ENGINE (ini yang nanti jadi isi endpoint /rekomendasi)
# ------------------------------------------------------------------
AMBANG = {"cabai": 45, "tomat": 50, "sayur_daun": 55}  # % kelembaban minimum (nilai awal, perlu validasi)
SELISIH_KRITIS = 15      # kalau kelembaban < ambang - 15  -> SEGERA, abaikan prakiraan hujan
PROB_HUJAN_TUNDA = 0.70  # prakiraan hujan >= 70% -> tunda siram (kalau belum kritis)
JAM_MIN_ANTAR_SIRAM = 12 # baru disiram < 12 jam lalu -> tahan rekomendasi


def rekomendasi(tanaman, kelembaban_rata, prob_hujan, jam_sejak_siram, margin=0, prob_tunda=PROB_HUJAN_TUNDA):
    """Input sudah berupa fitur turunan (rata-rata 3 pembacaan sensor, bukan 1 titik)."""
    if kelembaban_rata is None or prob_hujan is None:
        return {"rekomendasi": "CEK_MANUAL", "tingkat": "Perlu Dicek",
                "alasan": "Data sensor/cuaca belum lengkap, mohon cek lahan langsung"}
    ambang = AMBANG[tanaman]
    if jam_sejak_siram < JAM_MIN_ANTAR_SIRAM:
        return {"rekomendasi": "TAHAN", "tingkat": "Aman",
                "alasan": f"Baru disiram {jam_sejak_siram} jam lalu"}
    if kelembaban_rata >= ambang + margin:   # margin = pengaman terhadap noise sensor
        return {"rekomendasi": "TIDAK_PERLU", "tingkat": "Aman",
                "alasan": f"Kelembaban {kelembaban_rata:.0f}% masih di atas ambang {ambang}%"}
    if ambang - kelembaban_rata >= SELISIH_KRITIS:
        return {"rekomendasi": "SIRAM", "tingkat": "Segera",
                "alasan": f"Kelembaban {kelembaban_rata:.0f}% jauh di bawah ambang {ambang}%"}
    if prob_hujan >= prob_tunda:
        return {"rekomendasi": "TUNDA_HUJAN", "tingkat": "Perlu Dicek",
                "alasan": f"Kelembaban {kelembaban_rata:.0f}% (ambang {ambang}%), tapi peluang hujan {prob_hujan:.0%}"}
    return {"rekomendasi": "SIRAM", "tingkat": "Perlu Dicek",
            "alasan": f"Kelembaban {kelembaban_rata:.0f}% di bawah ambang {ambang}%, peluang hujan {prob_hujan:.0%}"}


# %% ---------------------------------------------------------------
# 2. UNIT TEST KECIL (pastikan aturan sesuai yang kita tulis di SRS)
# ------------------------------------------------------------------
def _tes_aturan():
    assert rekomendasi("cabai", 60, 0.1, 48)["rekomendasi"] == "TIDAK_PERLU"
    assert rekomendasi("cabai", 40, 0.1, 48)["rekomendasi"] == "SIRAM"
    assert rekomendasi("cabai", 40, 0.1, 48)["tingkat"] == "Perlu Dicek"
    assert rekomendasi("cabai", 40, 0.9, 48)["rekomendasi"] == "TUNDA_HUJAN"
    assert rekomendasi("cabai", 25, 0.9, 48)["rekomendasi"] == "SIRAM"      # kritis, hujan diabaikan
    assert rekomendasi("cabai", 25, 0.9, 48)["tingkat"] == "Segera"
    assert rekomendasi("tomat", 40, 0.1, 3)["rekomendasi"] == "TAHAN"
    assert rekomendasi("tomat", None, 0.1, 48)["rekomendasi"] == "CEK_MANUAL"
    print("Unit test aturan: OK")


# %% ---------------------------------------------------------------
# 3. SIMULASI LAHAN (neraca air sederhana, 1 langkah = 1 hari)
# ------------------------------------------------------------------
GAIN_SIRAM = 25          # tambahan kelembaban (poin %) tiap kali disiram
HUJAN_EFEKTIF = 10       # hujan >= 10 poin dianggap cukup mengubah keputusan


def simulasi(kebijakan, n_lahan=400, hari=60, noise_sd=2.5, peluang_sensor_mati=0.03, seed=42,
             margin=0, prob_tunda=PROB_HUJAN_TUNDA, n_baca=3):
    rng = np.random.default_rng(seed)
    baris = []
    for lahan in range(n_lahan):
        musim = rng.choice(["hujan", "kemarau"])
        p_hujan = 0.45 if musim == "hujan" else 0.10
        et_rata = 2.2 if musim == "hujan" else 3.6
        tanaman = rng.choice(list(AMBANG))
        ambang = AMBANG[tanaman]
        theta = rng.uniform(55, 75)
        for h in range(hari):
            # kejadian hari ini (belum diketahui saat keputusan pagi)
            hujan = rng.uniform(6, 28) if rng.random() < p_hujan else 0.0
            hujan_efektif = hujan >= HUJAN_EFEKTIF

            # sensor: rata-rata n_baca pembacaan berisik; kadang mati
            sensor_mati = rng.random() < peluang_sensor_mati
            baca = None if sensor_mati else float(np.mean(theta + rng.normal(0, noise_sd, n_baca)))

            # prakiraan hujan (tidak sempurna)
            prob = rng.beta(5, 2) if hujan_efektif else rng.beta(1.5, 5)

            # kebutuhan sebenarnya (kebenaran dasar)
            perlu = (theta < ambang) and (not hujan_efektif)

            # keputusan
            if kebijakan == "jaga_tani":
                out = rekomendasi(tanaman, baca, prob, 48, margin, prob_tunda)
                siram = out["rekomendasi"] == "SIRAM"
                ada_keputusan = out["rekomendasi"] != "CEK_MANUAL"
            else:  # jadwal tetap tiap 2 hari. CATATAN: pembanding kasar, parameternya sembarang -> jangan dipakai klaim "hemat air"
                out = None
                siram = (h % 2 == 0)
                ada_keputusan = True

            baris.append(dict(lahan=lahan, hari=h, musim=musim, tanaman=tanaman,
                              theta=theta, ambang=ambang, perlu=perlu, siram=siram,
                              ada_keputusan=ada_keputusan,
                              di_bawah_ambang=theta < ambang,
                              boros=siram and not perlu,
                              # tambahan: apa yang "dilihat" dan "diputuskan" sistem
                              kelembaban_sensor=baca, hujan_mm=hujan, hujan_efektif=hujan_efektif,
                              prob_hujan=prob,
                              rekomendasi=(out["rekomendasi"] if out else "JADWAL_TETAP"),
                              tingkat=(out["tingkat"] if out else None)))

            # update kelembaban besok
            theta = theta - rng.normal(et_rata, 0.6) + hujan + (GAIN_SIRAM if siram else 0)
            theta = float(np.clip(theta, 5, 85))
    return pd.DataFrame(baris)


def ringkas(df):
    d = df[df.ada_keputusan]
    tp = int(((d.siram) & (d.perlu)).sum()); fp = int(((d.siram) & (~d.perlu)).sum())
    fn = int(((~d.siram) & (d.perlu)).sum()); tn = int(((~d.siram) & (~d.perlu)).sum())
    n_lahan = df.lahan.nunique()
    return {
        "akurasi": (tp + tn) / max(len(d), 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "cakupan_keputusan": len(d) / len(df),
        "siram_per_lahan": float(df.siram.sum() / n_lahan),
        "boros_per_lahan": float(df.boros.sum() / n_lahan),
        "hari_di_bawah_ambang_per_lahan": float(df.di_bawah_ambang.sum() / n_lahan),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# %% ---------------------------------------------------------------
# 4. JALANKAN
# ------------------------------------------------------------------
def f1(r):
    p, rc = r["precision"], r["recall"]
    return 2 * p * rc / (p + rc) if (p + rc) else 0.0


if __name__ == "__main__":
    _tes_aturan()
    hasil = {}

    # (a) konfigurasi awal (sesuai SRS), data uji seed 42
    df_awal = simulasi("jaga_tani", seed=42)
    hasil["awal"] = ringkas(df_awal)

    # simpan dataset sintetisnya (1 baris = 1 lahan pada 1 hari, 400 lahan x 60 hari)
    kolom = {
        "lahan": "lahan_id", "hari": "hari_ke", "musim": "musim", "tanaman": "tanaman", "ambang": "ambang_pct",
        "theta": "kelembaban_sebenarnya_pct", "kelembaban_sensor": "kelembaban_sensor_pct",
        "hujan_mm": "hujan_poin", "hujan_efektif": "hujan_efektif", "prob_hujan": "prob_hujan_prakiraan",
        "perlu": "perlu_siram_sebenarnya", "rekomendasi": "rekomendasi", "tingkat": "tingkat",
        "siram": "disiram", "ada_keputusan": "ada_keputusan", "di_bawah_ambang": "di_bawah_ambang",
        "boros": "siram_tidak_perlu",
    }
    export = df_awal[list(kolom)].rename(columns=kolom).round(2)
    export["lahan_id"] = "L-" + export["lahan_id"].astype(int).astype(str).str.zfill(3)
    export.to_csv("data_sintetis_jaga_tani.csv", index=False)
    print(f"Dataset sintetis disimpan: {len(export)} baris x {export.shape[1]} kolom")

    # (b) tuning 2 parameter pakai seed 1 (data "latih"), lalu diuji ulang di seed 42 (data "uji")
    grid = []
    for margin in [0, 2, 4, 6]:
        for pt in [0.6, 0.7, 0.8, 0.9]:
            r = ringkas(simulasi("jaga_tani", seed=1, margin=margin, prob_tunda=pt))
            grid.append(dict(margin=margin, prob_tunda=pt, f1=f1(r), precision=r["precision"], recall=r["recall"]))
    terbaik = max(grid, key=lambda g: g["f1"])
    hasil["grid_tuning"] = grid
    hasil["terbaik_tuning"] = terbaik
    hasil["setelah_tuning"] = ringkas(simulasi("jaga_tani", seed=42, margin=terbaik["margin"],
                                               prob_tunda=terbaik["prob_tunda"]))

    # (c) sensitivitas terhadap noise sensor & sensor mati (pakai konfigurasi setelah tuning)
    hasil["sensitivitas_noise"] = []
    for sd in [1.5, 2.5, 4.0, 6.0]:
        r = ringkas(simulasi("jaga_tani", seed=42, noise_sd=sd, margin=terbaik["margin"], prob_tunda=terbaik["prob_tunda"]))
        r["noise_sd"] = sd
        hasil["sensitivitas_noise"].append(r)
    hasil["sensitivitas_sensor_mati"] = []
    for pm in [0.0, 0.03, 0.10, 0.20]:
        r = ringkas(simulasi("jaga_tani", seed=42, peluang_sensor_mati=pm, margin=terbaik["margin"], prob_tunda=terbaik["prob_tunda"]))
        r["peluang_sensor_mati"] = pm
        hasil["sensitivitas_sensor_mati"].append(r)

    # (d) apakah merata-ratakan lebih banyak pembacaan sensor membantu? (noise 2.5)
    hasil["sensitivitas_n_baca"] = []
    for nb in [1, 3, 6, 12]:
        r = ringkas(simulasi("jaga_tani", seed=42, n_baca=nb))
        r["n_baca"] = nb
        hasil["sensitivitas_n_baca"].append(r)

    print(json.dumps(hasil, indent=2))
    with open("hasil_poc.json", "w") as f:
        json.dump(hasil, f, indent=2)
