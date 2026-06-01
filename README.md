# Dokumentasi Proyek: RAG Quantitative Research Engine for IDX

## 1. Deskripsi Proyek
Proyek ini bertujuan untuk membangun sistem asisten riset kuantitatif berbasis kecerdasan buatan yang mengimplementasikan metode **Retrieval-Augmented Generation (RAG)**. Sistem dirancang khusus untuk memproses, mengekstraksi, dan menganalisis data dari laporan keuangan emiten yang terdaftar di Bursa Efek Indonesia (IDX). Fokus utama proyek adalah efisiensi ekstraksi data finansial yang sebelumnya dilakukan secara manual dan rentan terhadap bias informasi.

## 2. Fitur Sistem
Sistem ini menyediakan kapabilitas analisis mendalam dengan fitur-fitur sebagai berikut:
* **Analisis Data Terstruktur**: Mengekstraksi angka-angka dari tabel neraca, laba rugi, dan arus kas.
* **Filter Metadata Eksklusif**: Memastikan pencarian informasi hanya dilakukan pada dokumen emiten yang dipilih (Ticker-based filtering).
* **Mitigasi Halusinasi**: AI diinstruksikan untuk menolak memberikan jawaban apabila data pendukung tidak ditemukan dalam dokumen asli.
* **Atribusi Sumber**: Menyediakan referensi file PDF asli untuk setiap jawaban guna kebutuhan audit dan verifikasi data.
* **Antarmuka Responsif**: Menyediakan dashboard interaktif dengan rekomendasi pertanyaan analisis fundamental.

## 3. Spesifikasi Teknis
Berikut adalah ringkasan teknologi yang digunakan dalam pengembangan sistem:

| Komponen | Teknologi | Deskripsi |
| :--- | :--- | :--- |
| **Language Model** | Llama-3.3-70b-versatile | Model bahasa melalui Groq Cloud LPU untuk inferensi cepat. |
| **Vector Database** | ChromaDB | Penyimpanan vektor permanen dengan metrik Cosine Similarity. |
| **Embedding Model** | Multilingual-MiniLM-L12-v2 | Model pendukung bahasa Indonesia dan Inggris. |
| **Framework** | LangChain | Integrasi alur kerja RAG dan manajemen prompt. |
| **User Interface** | Streamlit | Framework aplikasi web untuk interaksi pengguna. |
| **PDF Engine** | PyPDF | Library ekstraksi teks dari dokumen PDF tidak terstruktur. |

## 4. Alur Pengembangan Proyek
Proyek ini diselesaikan melalui empat tahapan utama yang terstruktur sebagai berikut:

### Tahap 1: Inisialisasi & Akuisisi Data
* Penyiapan repositori GitHub dan lingkungan pengembangan berbasis cloud.
* Pengumpulan 50 laporan keuangan resmi (Financial Statements) dari website Bursa Efek Indonesia.
* Standardisasi penamaan file untuk memudahkan ekstraksi metadata.

### Tahap 2: Preprocessing & Ekstraksi Teks
* Implementasi PDF parsing untuk mengubah data tidak terstruktur menjadi format teks mentah.
* Proses **Hierarchical Chunking** menggunakan *Recursive Character Text Splitter*.
* Hasil akhir tahap ini adalah 9.744 potongan teks (chunks) dengan ukuran 1.000 karakter per bagian.

### Tahap 3: Indeksasi Vektor (Embedding)
* Transformasi data teks menjadi representasi vektor numerik.
* Konfigurasi database vektor secara permanen (Persistent Storage).
* Implementasi sistem *score thresholding* untuk membedakan tingkat relevansi dokumen.

### Tahap 4: Integrasi LLM & Deployment
* Penghubungan database ke model Llama-3.3 melalui Groq API.
* Penerapan *Prompt Engineering* untuk menetapkan persona AI sebagai Senior Financial Analyst.
* Deployment aplikasi web secara publik melalui Streamlit Cloud.

## 5. Panduan Penggunaan Sistem

Untuk menggunakan aplikasi **IDX AI Research Engine**, pengguna dapat mengikuti langkah-langkah berikut:

1. **Akses Aplikasi**: Buka tautan resmi di [https://idx-ai.streamlit.app/](https://idx-ai.streamlit.app/).
2. **Pilih Emiten**: Pada panel sidebar di sebelah kiri, pilih Kode Saham (Ticker) yang ingin dianalisis (misalnya: RISE, BBCA, atau IBFN).
3. **Pilih/Input Pertanyaan**:
   * Pengguna dapat mengklik tombol rekomendasi pertanyaan yang tersedia (Laba Bersih, Total Aset, Liabilitas, dll).
   * Pengguna dapat mengetik pertanyaan bebas pada kolom teks yang disediakan.
4. **Proses Analisis**: Klik tombol **Jalankan Analisis** dan tunggu sistem melakukan pencarian data.
5. **Verifikasi**: Tinjau jawaban yang diberikan oleh AI dan periksa bagian **Lihat Sumber Referensi** untuk melihat dokumen asli yang digunakan sebagai rujukan.

## 6. Anggota Tim Pengembang

Proyek ini dikembangkan melalui kolaborasi dua anggota tim dengan peran sebagai berikut:

| Nama Pengembang | Peran Utama | Link Profil |
| :--- | :--- | :--- |
| **Naufallm** | Backend Architect, RAG Pipeline & Deployment | [GitHub Naufallm](https://github.com/Naufallm) |
| **Syahrialfaturr** | UI/UX Designer, Frontend Developer & QA | [GitHub Syahrialfaturr](https://github.com/syahrialfaturr) |

---

**Catatan Akhir**: Seluruh data yang digunakan dalam sistem ini bersumber dari laporan keuangan publik yang diterbitkan oleh perusahaan tercatat di Bursa Efek Indonesia. Penggunaan sistem ini ditujukan untuk tujuan edukasi dan riset kuantitatif.
