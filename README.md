# Empirical-heavy forecaster (jump memory)

Forecaster untuk `polymarket-faa-attribution-01`: membaca satu case
(`makefun/polymarket-case/v1`) — harga Yes per menit tiap market sampai
checkpoint — dan menulis lima kuantil (5%, 25%, 50%, 75%, 95%) untuk tiap
menit dari 240 menit berikutnya. Python 3.13, standard library saja, tanpa
network, deterministik.

## Ide

Harga Yes Polymarket kebanyakan diam, lalu bergerak kecil, lalu sesekali
melompat besar. Random walk Gaussian dengan sigma tetap (baseline house)
salah di kedua ujung itu: terlalu lebar saat pasar tenang, terlalu sempit
saat pasar baru melompat. Forecaster ini memakai sejarah market itu sendiri:

- **Kuantil empiris.** Untuk tiap anchor horizon H = 1, 3, 6, 12, 30, 60,
  90, 180, 240 menit, dihitung kuantil empiris perubahan harga H-menit
  selama 2880 menit (48 jam) terakhir. Yang dipakai adalah *bentuk*
  sebarannya (selisih tiap kuantil terhadap median empiris), bukan
  mediannya.
- **Median = harga terakhir (persistence).** Median empiris return masa
  lalu ternyata pusat yang lebih buruk daripada harga terakhir (di proxy
  practice, pinball pooled 0,399 → 0,363 setelah median dipindah ke
  persistence). Sejarah membentuk lebar, bukan arah.
- **Campuran 75% empiris / 25% Gaussian.** Kuantil akhir:
  `q = x + 0,75 · w · (emp_q − emp_median) + 0,25 · z · sigma · sqrt(h)`,
  dengan `sigma` dari median pergerakan absolut 1-menit
  (`median|Δ| / 0,6745`), dibatasi [0,0013, 0,0055]. Antar anchor,
  kuantil empiris diinterpolasi linear dalam `sqrt(h)`.
- **Thin-anchor shrinkage.** Anchor yang sampelnya sedikit (market muda)
  bobot empirisnya dikecilkan dengan faktor `cnt / (cnt + 25)`, jadi
  anchor 13-sampel tidak mendominasi forecast.
- **Regime 10 jam.** Rasio sigma 10 jam terakhir terhadap sigma jendela
  48 jam mengskalakan seluruh lebar, dijepit ke [0,75, 1,55]. Pasar yang
  baru memanas dapat lebar lebih besar; pasar yang mendingin menyempit.
- **Jump memory.** Jika pergerakan 1 menit terakhir melebihi 6 sigma,
  kuantil luar (5% dan 95%) dilebarkan sampai 12%
  (`min(0,12, 0,03 · (rasio − 6))`). Market yang baru melompat lebih
  mungkin melompat lagi.

Semua kuantil dijepit ke [0, 1], dibulatkan 6 desimal, dan dipaksa
non-decreasing. Market yang terlalu muda untuk anchor mana pun memakai
Gaussian murni dengan sigma default 0,0025.

## Parameter (varian w3)

| Parameter | Nilai |
|---|---|
| Bobot empiris | 0,75 |
| Anchor | 1, 3, 6, 12, 30, 60, 90, 180, 240 menit |
| Jendela empiris | 2880 menit |
| Sigma (median-abs) | floor 0,0013, cap 0,0055 |
| Regime | jendela 10 jam, clamp [0,75, 1,55] |
| Jump | ambang 6 sigma, boost luar maks 12% |
| Thin-anchor shrinkage | `cnt / (cnt + 25)` |

## Cara menjalankan

```sh
# Satu case (mode argv):
python3 main.py <case>.in [<forecast>.out]
# Tanpa argv, main.py membaca $MAKEFUN_INPUT_DIR/*.in
# dan menulis $MAKEFUN_OUTPUT_DIR/*.out (default /input dan /output):
python3 main.py
# Contoh lokal:
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Output: `{"forecasts": [{"id": <market id>, "q": [[q05, q25, q50, q75, q95] x 240]}]}`,
satu baris JSON per market, ukuran di bawah 1 MiB per case.

## Validasi lokal (6 practice case)

Semua case valid (4–24 market sesuai case, 240 baris, 5 kuantil monoton
dalam [0, 1], 6 desimal), deterministik (byte-identik pada 2 run),
runtime 0,26–0,57 detik per case (batas 8 detik).

Kit tidak menyertakan file reference/answer, jadi skor pinball resmi
tidak bisa dihitung. Sebagai gantinya, skor proxy dihitung dengan memakai
harga 240 menit dari case berikutnya sebagai actual (p00–p04):
**w3 = 0,365502 (pooled) vs baseline house = 0,392518** — di bawah
baseline pada proxy ini. Baseline resmi practice 6 case adalah 0,363589;
proxy ini hanya 5 case dan bukan skor resmi.
