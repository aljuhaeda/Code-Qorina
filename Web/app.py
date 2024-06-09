import streamlit as st
from home import home_page
from about import about_page

# Main Function
def main():
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Home", "Tentang"])

    if page == "Home":
        home_page()
    elif page == "Tentang":
        about_page()

if __name__ == "__main__":
    main()
