import streamlit.components.v1 as components

# Read the HTML file
with open("data/Rerf.html", "r", encoding="utf-8") as f:
    html_data = f.read()

# Display it inside Streamlit
components.html(html_data, height=800, scrolling=True)