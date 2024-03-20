import streamlit as st
from plotly_figures import get_fig_nn
architecture=(2, 8, 8, 8, 1, 8, 8, 8, 2)
st.plotly_chart(get_fig_nn(architecture))
