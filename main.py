import streamlit as st
import os

from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(layout="wide")

sections = True

nav = get_nav_from_toml(
    ".streamlit/pages_sections.toml" if sections else ".streamlit/pages.toml"
)


print(os.listdir('.streamlit/'))


pg = st.navigation(nav)

add_page_title(pg)

pg.run()

