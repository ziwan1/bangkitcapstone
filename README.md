# New Renewable Energy Placement Recommendation System for Solar Panels Based on the Location of East Java District Using Geospatial Analysis, Statistical Approaches, and Deep Learning

Tim kami mengumpulkan dataset dalam kurun waktu 1 Januari 2010 hingga 10 April 2023. Dataset yang dikumpulkan terdiri dari data geografis cuaca kecamatan berupa kumpulan temperatur, kelembaban, tekanan udara, curah hujan, tutupan awan, radiasi matahari, tingkat penyinaran matahari langsung, dan kecepatan angin. Dataset ini kami dapatkan dari website open meteo. Dataset ini mencatat per jamnya perubahan dari data-data di atas dalam kurun waktu 1 Januari 2010 hingga 10 April 2023

Dataset ada di https://intip.in/datasetcuaca

Dataset pembuatan model ada di https://intip.in/datasetv1

Dataset tersebut kami dapatkan dengan scraping API openmeteo menggunakan script yang terdapat pada file apiscraper.ipynb

Pada directory github ini akan terdapat beberapa model hasil training dari dataset yang kami kumpulkan 

Jika ingin melakukan training, maka lakukan langkah berikut

1. Extract file capstone.zip
2. Tambahkan CSV yang terdapat pada dataset/tambahkan dataset sendiri
3. Ubah destinasi dalam file ts_forecasting.py agar sesuai dengan directory anda
4. Gunakan command conda env create -f environment.yml sesuai dengan file .yml yang terdapat pada github ini (disarankan diganti nama environmentnya)
5. Aktifkan environment hasil import dengan command conda activate <environment_name>
6. Buka command prompt dan cd ke directory hasil unzip
7. Jalankan command python
8. Tunggu hingga training selesai

