# RAG Quantitative Research Engine for IDX

Project ini bertujuan untuk membangun mesin riset kuantitatif berbasis **Retrieval-Augmented Generation (RAG)** yang dikhususkan untuk menganalisis laporan keuangan emiten di Bursa Efek Indonesia (IDX). Sistem ini dirancang untuk mengatasi hambatan ekstraksi manual data finansial yang memakan waktu dan rentan bias.

## 🚀 Progres Pengembangan

### 1. Inisialisasi Project & Akuisisi Data
Pada tahap awal, fokus utama adalah membangun fondasi pengembangan dan pengumpulan dataset mentah:
*   **Lingkungan Pengembangan:** Menyiapkan repository GitHub untuk kontrol versi dan **Google Colab** sebagai platform komputasi berbasis cloud menggunakan Python.
*   **Pengumpulan Dataset:** Kami telah mengumpulkan **50 dokumen Laporan Keuangan (Financial Statements)** secara manual dari sumber resmi [Bursa Efek Indonesia (IDX)](https://www.idx.co.id/id/perusahaan-tercatat/laporan-keuangan-dan-tahunan).
*   **Penyimpanan Data:** Seluruh dokumen PDF disimpan dalam folder terstruktur `dataset_idx` dengan standar penamaan file yang konsisten untuk memudahkan identitas emiten (Ticker).

### 2. Pipeline Ekstraksi & Preprocessing Data
Langkah kedua adalah mengubah dokumen PDF yang tidak terstruktur menjadi potongan informasi yang siap diproses oleh Kecerdasan Buatan (AI):
*   **Pemindaian PDF Otomatis:** Membangun sistem pembaca PDF menggunakan library `pypdf` untuk mengekstraksi teks mentah dari seluruh halaman dokumen laporan keuangan.
*   **Automated Metadata Tagging:** Mengembangkan logika pemrograman untuk mengidentifikasi kode saham (Ticker) dari setiap file secara otomatis, memastikan data tidak tertukar antar perusahaan.
*   **Strategi Hierarchical Chunking:** Menggunakan `RecursiveCharacterTextSplitter` dari framework **LangChain**. Laporan keuangan dipecah menjadi **9.744 potongan teks (chunks)** berukuran 1000 karakter dengan *overlap* 200 karakter. Hal ini bertujuan agar konteks angka-angka finansial tetap terjaga dan akurat.
*   **Audit Kualitas:** Melakukan verifikasi akhir untuk memastikan data hasil ekstraksi bersih dan memiliki struktur yang tepat sebelum masuk ke tahap penyimpanan vektor.

## 📊 Status Dataset Saat Ini
| Indikator | Jumlah |
|---|---|
| Total Dokumen PDF | 50 File |
| Total Emiten Unik | Teridentifikasi via Metadata |
| Total Potongan Teks (Chunks) | 9.744 Chunks |
| **Status** | **Siap untuk Tahap Embedding & Vector DB** |

## 🛠️ Tech Stack
*   **Bahasa Pemrograman:** Python
*   **Framework AI:** LangChain
*   **PDF Processing:** PyPDF
*   **Data Management:** Pandas
*   **Version Control:** GitHub

---

### Cara Menjalankan Pipeline
1. Clone repository ini:
   ```bash
   git clone https://github.com/Naufallm/RAG-Quantitative-Research-Engine-for-IDX.git
   ```
2. Instal library yang diperlukan:
   ```bash
   pip install langchain pypdf langchain-text-splitters
   ```
3. Buka notebook di Google Colab dan jalankan setiap section untuk memproses file di folder `dataset_idx`.

---
*Note: Dokumentasi ini akan terus diperbarui seiring dengan perkembangan integrasi model LLM dan Vector Database.*
