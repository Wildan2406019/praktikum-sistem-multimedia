from pathlib import Path
from collections import Counter
import csv
import json
import subprocess

import cv2
import numpy as np
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
HASIL = BASE / "hasil"
HASIL.mkdir(exist_ok=True)

# Kedua video dibandingkan pada durasi dan resolusi yang sama.
DURASI_ANALISIS = 10
UKURAN_ANALISIS = (640, 360)


def ffprobe(video, opsi):
    proses = subprocess.run(
        ["ffprobe", "-v", "error", *opsi, "-of", "json", str(video)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(proses.stdout)


def analisis(nama):
    video = BASE / f"video_{nama}.mp4"
    folder = HASIL / nama
    folder.mkdir(exist_ok=True)

    if not video.exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {video}")

    metadata = ffprobe(video, [
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,duration,nb_frames"
    ])

    data = ffprobe(video, [
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pict_type,key_frame,best_effort_timestamp_time"
    ])

    for filename, isi in [
        ("metadata.json", metadata),
        ("frames.json", data),
    ]:
        (folder / filename).write_text(
            json.dumps(isi, indent=2), encoding="utf-8"
        )

    frames = data["frames"]
    jumlah = Counter(f.get("pict_type", "?") for f in frames)
    stream = metadata["streams"][0]
    pembilang, penyebut = map(int, stream["r_frame_rate"].split("/"))
    fps = pembilang / penyebut

    # Indeks dimulai dari 0, sesuai urutan pembacaan OpenCV.
    keyframes = [
        i for i, f in enumerate(frames)
        if int(f.get("key_frame", 0)) == 1
    ]
    panjang_gop = np.diff(keyframes)

    with open(folder / "daftar_frame.csv", "w",
              newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Indeks", "Waktu_detik", "Tipe", "Keyframe"])
        for i, frame in enumerate(frames):
            writer.writerow([
                i,
                frame.get("best_effort_timestamp_time", ""),
                frame.get("pict_type", "?"),
                frame.get("key_frame", 0),
            ])

    plt.figure(figsize=(6, 4))
    batang = plt.bar(
        ["I", "P", "B"],
        [jumlah[t] for t in ["I", "P", "B"]],
        color=["tomato", "steelblue", "orange"],
    )
    for bar in batang:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(bar.get_height())),
            ha="center", va="bottom"
        )
    plt.title(f"Distribusi Frame Video {nama.capitalize()}")
    plt.xlabel("Tipe frame")
    plt.ylabel("Jumlah frame")
    plt.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(folder / "distribusi_frame.png", dpi=160)
    plt.close()

    # Simpan frame sebelum, tepat pada, dan setelah keyframe.
    # Ambil maksimal tiga keyframe awal.
    target_sampel = set()
    for k in keyframes[:3]:
        for indeks in [k - 1, k, k + 1]:
            if 0 <= indeks < len(frames):
                target_sampel.add(indeks)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV tidak dapat membuka {video}")

    waktu, selisih = [], []
    sebelumnya = None
    indeks = 0

    try:
        while True:
            berhasil, frame = cap.read()
            if not berhasil:
                break

            if indeks in target_sampel:
                info = frames[indeks]
                tipe = info.get("pict_type", "?")
                penanda = (
                    "keyframe" if info.get("key_frame") == 1
                    else "sekitar_keyframe"
                )
                cv2.imwrite(
                    str(folder / (
                        f"sampel_{indeks:04d}_{tipe}_{penanda}.jpg"
                    )),
                    frame
                )

            if indeks < len(frames):
                timestamp = frames[indeks].get(
                    "best_effort_timestamp_time"
                )
            else:
                timestamp = None

            t = float(timestamp) if timestamp is not None else indeks / fps
            awal = frames[0].get("best_effort_timestamp_time")
            if awal is not None:
                t -= float(awal)

            if t < DURASI_ANALISIS:
                kecil = cv2.resize(
                    frame, UKURAN_ANALISIS,
                    interpolation=cv2.INTER_AREA
                )
                abu = cv2.cvtColor(kecil, cv2.COLOR_BGR2GRAY)

                if sebelumnya is not None:
                    beda = cv2.absdiff(abu, sebelumnya)
                    waktu.append(t)
                    selisih.append(float(np.mean(beda)))

                sebelumnya = abu

            indeks += 1
    finally:
        cap.release()

    if not selisih:
        raise RuntimeError("Frame untuk analisis tidak mencukupi.")

    with open(folder / "frame_differencing.csv", "w",
              newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Waktu_detik", "Rata_rata_selisih"])
        writer.writerows(zip(waktu, selisih))

    rata = float(np.mean(selisih))
    gop_rata = (
        float(np.mean(panjang_gop)) if len(panjang_gop) else None
    )

    print(f"\nVIDEO {nama.upper()}")
    print(f"Codec          : {stream['codec_name']}")
    print(f"Resolusi       : {stream['width']} x {stream['height']}")
    print(f"FPS            : {fps:.3f}")
    print(f"Durasi         : {stream.get('duration', '-')} detik")
    print(f"Total frame    : {len(frames)}")
    print(f"I / P / B      : {jumlah['I']} / {jumlah['P']} / {jumlah['B']}")
    print(f"Indeks keyframe: {keyframes}")
    if gop_rata is not None:
        print(f"Rata-rata GOP  : {gop_rata:.2f} frame")
    else:
        print("Rata-rata GOP  : Tidak cukup keyframe")
    print(f"Rata-rata beda : {rata:.4f}")

    return {
        "nama": nama,
        "jumlah": jumlah,
        "waktu": waktu,
        "selisih": selisih,
        "rata": rata,
        "gop": gop_rata,
    }


def main():
    hasil = [analisis("statis"), analisis("dinamis")]

    # Dua panel dengan skala vertikal sama agar mudah dibandingkan.
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, sharey=True
    )

    for ax, data in zip(axes, hasil):
        ax.plot(data["waktu"], data["selisih"], linewidth=0.9)
        ax.axhline(
            data["rata"], color="red", linestyle="--",
            label=f"Rata-rata: {data['rata']:.4f}"
        )
        ax.set_title(f"Video {data['nama'].capitalize()}")
        ax.set_ylabel("Rata-rata selisih piksel")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Waktu (detik)")
    fig.suptitle("Perbandingan Frame Differencing — 10 Detik Awal")
    fig.tight_layout()
    fig.savefig(HASIL / "perbandingan_temporal.png", dpi=160)
    plt.close(fig)

    with open(HASIL / "ringkasan.csv", "w",
              newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Video", "Frame_I", "Frame_P", "Frame_B",
            "Rata_rata_GOP", "Rata_rata_selisih"
        ])
        for data in hasil:
            writer.writerow([
                data["nama"],
                data["jumlah"]["I"],
                data["jumlah"]["P"],
                data["jumlah"]["B"],
                data["gop"],
                data["rata"],
            ])

    print(f"\nSelesai. Hasil tersimpan di: {HASIL}")


if __name__ == "__main__":
    main()