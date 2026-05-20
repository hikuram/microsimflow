from __future__ import annotations

import streamlit as st

try:
    from app.streamlit_builder import render_builder
    from app.streamlit_importer import render_importer
except ModuleNotFoundError:
    from streamlit_builder import render_builder
    from streamlit_importer import render_importer


def main() -> None:
    st.set_page_config(
        page_title="microsimflow",
        page_icon=":material/hive:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.markdown("### :material/apps: Workspace")
        with st.container(border=True):
            mode = st.segmented_control(
                "Application mode",
                [
                    ":material/construction: Builder",
                    ":material/image_search: Importer",
                ],
                default=":material/construction: Builder",
                key="microsimflow_app_mode",
            )

    if mode == ":material/image_search: Importer":
        render_importer()
    else:
        render_builder()


if __name__ == "__main__":
    main()
