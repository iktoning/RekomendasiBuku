import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from keras.models import load_model

# Memuat model
model = tf.keras.models.load_model('model.keras')

Dataset_buku = './data_sets'

@st.cache_data
#def prepare_data

#def book_recomend

def get_user_data(user_id, book_data):
    
    Dataset_buku = './data_sets'

    ratings_data = pd.read_csv(Dataset_buku + '/ratings.csv')

    df = ratings_data
    id_buku = df['book_id'].unique().tolist()

    book_to_book_encoded = {i: x for i, x in enumerate(id_buku)}

    user_ratings = ratings_data[ratings_data['user_id'] == user_id]
    book_ids_read_by_user = user_ratings['book_id'].values

    book_not_read = book_data[~book_data['id_buku'].isin(book_ids_read_by_user)]['id_buku']
    book_not_read = list(
        set(book_not_read)
        .intersection(set(book_to_book_encoded.keys()))
    )

    book_not_read = [[book_to_book_encoded.get(x)] for x in book_not_read]

    return user_ratings, book_not_read

def show_user_recommendations(user_ratings, book_not_read, book_data):
    
    Dataset_buku = './data_sets'

    ratings_data = pd.read_csv(Dataset_buku + '/ratings.csv')

    df = ratings_data
    id_reader = df['user_id'].unique().tolist()
    id_buku = df['book_id'].unique().tolist()
    
    user_to_user_encoded = {x: i for i, x in enumerate(id_reader)}
    book_encoded_to_book = {i: x for i, x in enumerate(id_buku)}

    user_encoder = user_to_user_encoded.get(user_ratings['user_id'].values[0])
    user_book_array = np.hstack(
        ([[user_encoder]] * len(book_not_read), book_not_read)
    )

    rating_buku = model.predict(user_book_array).flatten()

    top_ratings_indices = rating_buku.argsort()[-10:][::-1]
    recommended_book_ids = [
        book_encoded_to_book.get(book_not_read[x][0]) for x in top_ratings_indices
    ]

    st.write('Menampilkan Rekomendasi Buku untuk Pembaca dengan User ID:', user_ratings['user_id'].values[0])
    st.write('===' * 15)
    st.write('Daftar Rekomendasi Buku dengan rating tinggi dari pembaca')
    st.write('----' * 15)

    top_book_user = (
        user_ratings.sort_values(
            by='rating',
            ascending=False
        )
        .head(5)
        .book_id.values
    )

    book_df_rows = book_data[book_data['id_buku'].isin(top_book_user)]
    for row in book_df_rows.itertuples():
        st.write(row.penulis, ':', row.judul_buku)

    st.write('----' * 15)
    st.write('Daftar Top 10 Buku yang Direkomendasikan')
    st.write('----' * 15)

    recommended_book = book_data[book_data['id_buku'].isin(recommended_book_ids)]
    for row in recommended_book.itertuples():
        st.write(row.penulis, ':', row.judul_buku)


def main():
    st.title('Sistem Rekomendasi Buku')

    book_data, cosine_sim_df = prepare_data()

    judul_buku_input = st.text_input('Masukkan judul buku:')
    rekomendasi_jumlah = st.slider('Jumlah rekomendasi:', 1, 10, 5)

    if st.button('Rekomendasikan'):
        rekomendasi = book_recommendations(judul_buku_input, cosine_sim_df, book_data, k=rekomendasi_jumlah)
        st.write('Rekomendasi Buku:')
        st.dataframe(rekomendasi)


    user_id_input = st.text_input('Masukkan user ID:')
    
    if st.button('Tampilkan Rekomendasi User'):
        # df = pd.read_csv(Dataset_buku + '/ratings.csv')
        user_ratings, book_not_read = get_user_data(int(user_id_input), book_data)
        show_user_recommendations(user_ratings, book_not_read, book_data)


# Memanggil fungsi show_data saat aplikasi Streamlit dijalankan
if __name__ == '__main__':
    main()
