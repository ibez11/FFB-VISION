#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JANJANG VISION - Penghitung Janjang Kelapa Sawit (TBS) berbasis AI
==================================================================
Satu file Python untuk mendeteksi & menghitung tandan buah segar (TBS)
kelapa sawit ("janjang") menggunakan YOLO (Ultralytics).

MODE
----
1) Foto          : python janjang_counter.py foto.jpg
2) Video / CCTV  : python janjang_counter.py --video cctv.mp4
3) Kamera        : python janjang_counter.py --cam
4) Training      : python janjang_counter.py --train dataset.yaml
5) Demo          : python janjang_counter.py --demo   (uji coba pipeline)

CONTOH LENGKAP
--------------
  python janjang_counter.py foto.jpg
  python janjang_counter.py foto.jpg --model model_janjang.pt --conf 0.4
  python janjang_counter.py --video cctv.mp4 --model model_janjang.pt --save
  python janjang_counter.py --cam --model model_janjang.pt
  python janjang_counter.py --train dataset.yaml --epochs 100

INSTALL (sekali saja)
---------------------
  pip install ultralytics opencv-python

CATATAN
-------
- Model bawaan (yolo11n.pt) dilatih di dataset COCO, jadi TIDAK mengenali
  janjang secara khusus. Untuk akurasi di lapangan: kumpulkan foto janjang,
  anotasi (LabelImg / Roboflow), latih dengan mode 4, lalu pakai --model.
- Jika model custom punya kelas bernama "janjang" / "tbs" / "ffb", program
  otomatis hanya menghitung kelas itu saja.
"""

import argparse
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- Cek dependensi -------------------------------------------------------
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "ERROR: library 'ultralytics' belum terinstall.\n"
        "Jalankan dulu:  pip install ultralytics opencv-python"
    )

try:
    import cv2
except ImportError:
    sys.exit(
        "ERROR: library 'opencv-python' belum terinstall.\n"
        "Jalankan dulu:  pip install ultralytics opencv-python"
    )

import numpy as np

# Nama kelas yang dianggap sebagai janjang (sesuaikan dengan model custom-mu)
JANJANG_NAMES = {"janjang", "tbs", "ffb", "bunch", "fruit_bunch", "fruit"}


# --- Fungsi inti -----------------------------------------------------------
def cari_kelas_janjang(names):
    """Cari id kelas 'janjang' pada model. Return None jika tidak ada."""
    for cid, nama in names.items():
        if str(nama).lower() in JANJANG_NAMES:
            return cid
    return None


def deteksi_frame(model, frame_bgr, conf, classes):
    """Deteksi objek pada satu frame (numpy BGR), kembalikan hitungan + gambar anotasi."""
    results = model.predict(source=frame_bgr, conf=conf, classes=classes, verbose=False)
    r = results[0]
    names = r.names
    cls_ids = r.boxes.cls.int().tolist() if r.boxes is not None else []

    per_kelas = {}
    for c in cls_ids:
        nama = names.get(c, str(c))
        per_kelas[nama] = per_kelas.get(nama, 0) + 1

    return per_kelas, r.plot()


def hitung_gambar(model, path_foto, conf, classes):
    """Hitung janjang pada satu file gambar, simpan hasil ber-anotasi."""
    img = cv2.imread(str(path_foto))
    if img is None:
        sys.exit(f"ERROR: tidak bisa membaca gambar: {path_foto}")
    per_kelas, annotated = deteksi_frame(model, img, conf, classes)
    path_hasil = path_foto.with_name(path_foto.stem + "_hasil" + path_foto.suffix)
    cv2.imwrite(str(path_hasil), annotated)
    return per_kelas, path_hasil


def cetak_ringkasan(per_kelas, path_hasil=None):
    """Cetak hasil hitungan ke layar."""
    total = sum(per_kelas.values())
    print("=" * 52)
    print("HASIL DETEKSI")
    print("=" * 52)
    if not per_kelas:
        print("  (tidak ada objek terdeteksi)")
    for nama, n in per_kelas.items():
        print(f"  {nama:<20}: {n}")
    print("-" * 52)
    print(f"  TOTAL            : {total}")
    if path_hasil:
        print(f"  Gambar hasil     : {path_hasil}")
    print("=" * 52)
    return total


# --- Mode deteksi ----------------------------------------------------------
def mode_foto(model, path_foto, conf, classes):
    per_kelas, path_hasil = hitung_gambar(model, path_foto, conf, classes)
    cetak_ringkasan(per_kelas, path_hasil)


def _mode_stream(model, cap, conf, classes, save, path_video=None):
    """Loop deteksi per frame dari video/kamera. Tekan 'q' untuk keluar."""
    if not cap.isOpened():
        sys.exit("ERROR: tidak bisa membuka sumber video/kamera.")

    writer = None
    out_path = None
    if save and path_video is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = path_video.with_name(path_video.stem + "_hasil.mp4")
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    print("Tekan 'q' untuk keluar.")
    total_akumulasi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        per_kelas, annotated = deteksi_frame(model, frame, conf, classes)
        total = sum(per_kelas.values())
        total_akumulasi += total

        overlay = annotated.copy()
        teks = f"Janjang: {total}   (akumulasi: {total_akumulasi})"
        cv2.putText(overlay, teks, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 0), 3)
        cv2.imshow("Janjang Vision - tekan 'q' untuk keluar", overlay)
        if writer is not None:
            writer.write(overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Selesai. Total janjang terdeteksi: {total_akumulasi}")
    if out_path:
        print(f"Video hasil disimpan: {out_path}")


def mode_video(model, path_video, conf, classes, save):
    cap = cv2.VideoCapture(str(path_video))
    _mode_stream(model, cap, conf, classes, save, path_video)


def mode_kamera(model, conf, classes):
    cap = cv2.VideoCapture(0)  # 0 = kamera pertama
    _mode_stream(model, cap, conf, classes, save=False)


# --- Mode training ---------------------------------------------------------
def mode_train(dataset_yaml, epochs, imgsz, batch, patience):
    """Latih model custom dengan dataset format YOLO (data.yaml)."""
    if not Path(dataset_yaml).exists():
        sys.exit(f"ERROR: file dataset tidak ditemukan: {dataset_yaml}")
    model = YOLO("yolo11n.pt")  # pretrained sebagai titik awal
    print(f"Mulai training {epochs} epoch pada dataset: {dataset_yaml}")
    print("Proses ini bisa lama (jam), tergantung GPU/CPU dan jumlah data.")
    model.train(data=dataset_yaml, epochs=epochs, imgsz=imgsz,
                batch=batch, patience=patience, name="janjang_train")
    best_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print("=" * 52)
    print("Training selesai!")
    print(f"Model terbaik: {best_path}")
    print(f"Pakai dengan:  python janjang_counter.py foto.jpg --model {best_path}")
    print("=" * 52)


# --- Mode demo -------------------------------------------------------------
def buat_gambar_demo(path_out="demo_janjang.jpg"):
    """Buat gambar uji coba sintetis berisi 'janjang' untuk mengetes pipeline."""
    rng = np.random.default_rng(42)
    h, w = 720, 960
    img = np.full((h, w, 3), (55, 130, 60), dtype=np.uint8)  # hijau kebun

    for _ in range(12):
        cx = int(rng.integers(90, w - 90))
        cy = int(rng.integers(90, h - 90))
        r = int(rng.integers(30, 50))
        # badan janjang: coklat gelap
        cv2.circle(img, (cx, cy), r, (38, 55, 85), -1)
        cv2.circle(img, (cx, cy), int(r * 0.6), (60, 80, 110), -1)
        # duri/spike kasar
        for _ in range(12):
            ang = rng.uniform(0, 2 * math.pi)
            x1 = int(cx + rng.uniform(-r * 0.5, r * 0.5))
            y1 = int(cy + rng.uniform(-r * 0.5, r * 0.5))
            x2 = int(x1 + 14 * math.cos(ang))
            y2 = int(y1 + 14 * math.sin(ang))
            cv2.line(img, (x1, y1), (x2, y2), (25, 40, 65), 3)

    cv2.imwrite(path_out, img)
    print(f"Gambar demo dibuat: {path_out}")
    return Path(path_out)


# --- Main ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Janjang Vision - hitung janjang kelapa sawit (TBS) dengan AI (YOLO).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("source", nargs="?", help="path foto (jpg/png) yang mau dihitung")
    p.add_argument("--model", default="yolo11n.pt",
                   help="model .pt (default yolo11n.pt, otomatis diunduh saat pertama dipakai)")
    p.add_argument("--conf", type=float, default=0.35,
                   help="ambang confidence 0-1 (default 0.35)")
    p.add_argument("--classes", help="filter kelas, contoh: 0,1 (opsional)")
    p.add_argument("--video", help="path video/CCTV untuk dihitung")
    p.add_argument("--cam", action="store_true", help="hitung real-time dari kamera")
    p.add_argument("--save", action="store_true", help="simpan video hasil (dengan --video)")
    p.add_argument("--train", help="path dataset.yaml untuk training model custom")
    p.add_argument("--epochs", type=int, default=100, help="jumlah epoch training")
    p.add_argument("--imgsz", type=int, default=640, help="ukuran gambar training")
    p.add_argument("--batch", type=int, default=-1,
                   help="batch size training (default: otomatis)")
    p.add_argument("--patience", type=int, default=20,
                   help="early stop jika tidak ada peningkatan (epoch)")
    p.add_argument("--demo", action="store_true",
                   help="buat gambar uji coba sintetis & jalankan deteksi")
    args = p.parse_args()

    # 1) Mode training
    if args.train:
        mode_train(args.train, args.epochs, args.imgsz, args.batch, args.patience)
        return

    # 2) Mode deteksi: muat model
    print(f"Memuat model: {args.model} ...")
    model = YOLO(args.model)
    names = model.names

    classes = None
    if args.classes:
        classes = [int(x) for x in args.classes.split(",")]
    else:
        cid = cari_kelas_janjang(names)
        if cid is not None:
            classes = [cid]
            print(f"Model mengenali kelas janjang ('{names[cid]}') - hanya kelas ini yang dihitung.")
        else:
            print("Model tidak punya kelas janjang spesifik - menghitung semua objek terdeteksi.")
            print("Tips: latih model sendiri dengan --train, atau filter dengan --classes 0,1")

    # 3) Demo
    if args.demo:
        path_demo = buat_gambar_demo()
        per_kelas, path_hasil = hitung_gambar(model, path_demo, args.conf, classes)
        cetak_ringkasan(per_kelas, path_hasil)
        return

    # 4) Video / kamera / foto
    if args.video:
        mode_video(model, Path(args.video), args.conf, classes, args.save)
    elif args.cam:
        mode_kamera(model, args.conf, classes)
    elif args.source:
        mode_foto(model, Path(args.source), args.conf, classes)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
