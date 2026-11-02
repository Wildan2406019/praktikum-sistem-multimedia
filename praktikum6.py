
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

os.makedirs("Hasil", exist_ok=True)

gambar_asli = Image.open("input.jpeg").convert("RGB")

for kualitas in [10, 50, 90]:
    gambar_asli.save(
        f"Hasil/jpeg_{kualitas}.jpeg",
        "JPEG",
        quality=kualitas
    )

    gambar_asli.save(
        f"Hasil/webp_{kualitas}.webp",
        "WEBP",
        quality=kualitas
    )

print("Enam gambar kompresi berhasil dibuat.")

gambar_asli = np.array(gambar_asli)

def hitung_mse(asli, kompresi):
    selisih = asli.astype(float) - kompresi.astype(float)
    return np.mean(selisih ** 2)

def hitung_psnr_manual(mse):
    if mse == 0:
        return float("inf")
    return 10 * np.log10((255 ** 2) / mse)

daftar_gambar = [
    f"Hasil/jpeg_{kualitas}.jpeg"
    for kualitas in [10, 50, 90]
] + [
    f"Hasil/webp_{kualitas}.webp"
    for kualitas in [10, 50, 90]
]

hasil = []

for nama_file in daftar_gambar:
    gambar_kompresi = np.array(
        Image.open(nama_file).convert("RGB")
    )

    mse = hitung_mse(gambar_asli, gambar_kompresi)
    psnr_manual = hitung_psnr_manual(mse)

    psnr_skimage = peak_signal_noise_ratio(
        gambar_asli,
        gambar_kompresi,
        data_range=255
    )

    ssim = structural_similarity(
        gambar_asli,
        gambar_kompresi,
        channel_axis=2,
        data_range=255
    )

    ukuran = os.path.getsize(nama_file) / 1024

    hasil.append({
        "nama": nama_file,
        "format": "JPEG" if "jpeg" in nama_file else "WebP",
        "kualitas": int(nama_file.split("_")[1].split(".")[0]),
        "mse": mse,
        "psnr_manual": psnr_manual,
        "psnr": psnr_skimage,
        "ssim": ssim,
        "ukuran": ukuran
    })

print("\nTABEL HASIL PENGUJIAN")
print("-" * 105)
print(
    f"{'Format':<8} {'Kualitas':<10} {'MSE':<12} "
    f"{'PSNR Manual':<14} {'PSNR Skimage':<14} "
    f"{'SSIM':<10} {'Ukuran KB':<12}"
)
print("-" * 105)

for d in hasil:
    print(
        f"{d['format']:<8} "
        f"{d['kualitas']:<10} "
        f"{d['mse']:<12.4f} "
        f"{d['psnr_manual']:<14.4f} "
        f"{d['psnr']:<14.4f} "
        f"{d['ssim']:<10.4f} "
        f"{d['ukuran']:<12.2f}"
    )

print("-" * 105)

with open("Hasil/hasil_pengujian.csv", "w", newline="") as file:
    kolom = [
        "Format", "Kualitas", "MSE", "PSNR Manual",
        "PSNR Skimage", "SSIM", "Ukuran (KB)"
    ]

    writer = csv.DictWriter(file, fieldnames=kolom)
    writer.writeheader()

    for d in hasil:
        writer.writerow({
            "Format": d["format"],
            "Kualitas": d["kualitas"],
            "MSE": round(d["mse"], 4),
            "PSNR Manual": round(d["psnr_manual"], 4),
            "PSNR Skimage": round(d["psnr"], 4),
            "SSIM": round(d["ssim"], 4),
            "Ukuran (KB)": round(d["ukuran"], 2)
        })

with open("Hasil/hasil_pengujian.txt", "w") as file:
    file.write("TABEL HASIL PENGUJIAN PSNR DAN SSIM\n\n")
    file.write(
        f"{'Format':<8} {'Kualitas':<10} {'MSE':<12} "
        f"{'PSNR Manual':<14} {'PSNR Skimage':<14} "
        f"{'SSIM':<10} {'Ukuran KB':<12}\n"
    )

    for d in hasil:
        file.write(
            f"{d['format']:<8} "
            f"{d['kualitas']:<10} "
            f"{d['mse']:<12.4f} "
            f"{d['psnr_manual']:<14.4f} "
            f"{d['psnr']:<14.4f} "
            f"{d['ssim']:<10.4f} "
            f"{d['ukuran']:<12.2f}\n"
        )

jpeg = [d for d in hasil if d["format"] == "JPEG"]
webp = [d for d in hasil if d["format"] == "WebP"]

plt.figure(figsize=(8, 5))

plt.plot(
    [d["ukuran"] for d in jpeg],
    [d["psnr"] for d in jpeg],
    marker="o",
    label="JPEG"
)

plt.plot(
    [d["ukuran"] for d in webp],
    [d["psnr"] for d in webp],
    marker="s",
    label="WebP"
)

plt.xlabel("Ukuran File (KB)")
plt.ylabel("PSNR (dB)")
plt.title("Perbandingan PSNR JPEG dan WebP")
plt.legend()
plt.grid(True)
plt.savefig("Hasil/grafik_psnr.png")
plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    [d["ukuran"] for d in jpeg],
    [d["ssim"] for d in jpeg],
    marker="o",
    label="JPEG"
)

plt.plot(
    [d["ukuran"] for d in webp],
    [d["ssim"] for d in webp],
    marker="s",
    label="WebP"
)

plt.xlabel("Ukuran File (KB)")
plt.ylabel("SSIM")
plt.title("Perbandingan SSIM JPEG dan WebP")
plt.legend()
plt.grid(True)
plt.savefig("Hasil/grafik_ssim.png")
plt.show()

print("\nSemua hasil berhasil disimpan di folder Hasil!")
