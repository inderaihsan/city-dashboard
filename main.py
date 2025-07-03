import streamlit as st
import os

from st_pages import add_page_title, get_nav_from_toml, hide_pages

st.set_page_config(layout="wide", 
                   initial_sidebar_state="expanded"
                )

sections = True 

st.logo(
    'assets/man.png'
)

nav = get_nav_from_toml(
    ".streamlit/pages_sections.toml"
)


# print(os.listdir('.streamlit/'))


pg = st.navigation(nav)
# hide_pages(
#     [
#         # "Automated Valuation Model",
#     ],
# )
add_page_title(pg)

pg.run()

