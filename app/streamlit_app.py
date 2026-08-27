"""Streamlit entry point for the congested-fixture-intelligence dashboard.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations


# DESIGN: keep this file thin — it should only wire config + cached loaders to
# the page components, so switching to a different UI framework later is
# a mechanical change rather than a rewrite.
def main() -> None:
    """Render the top-level dashboard.

    Args:
        None.

    Returns:
        None.

    Raises:
        NotImplementedError: Placeholder until the pipeline lands.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
