import streamlit as st
from home import show_home
from tentang import show_tentang

# Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih halaman", ["Home", "Tentang"])

# Menampilkan halaman yang dipilih
if page == "Home":
    show_home()
elif page == "Tentang":
    show_tentang()
