"""Headless UI smoke test using Streamlit's AppTest harness.

Run: python tests/test_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest  # noqa: E402


def _by_key(elements, key):
    for el in elements:
        if getattr(el, "key", None) == key:
            return el
    return None


def _goto(at: AppTest, token: str) -> None:
    """Switch the sidebar page radio to the option containing ``token``.

    Matching on a substring (not the full emoji-prefixed label) keeps the test
    robust to cosmetic label changes.
    """
    radio = at.sidebar.radio[0]
    for option in radio.options:
        if token in option:
            radio.set_value(option).run()
            return
    raise AssertionError(f"page not found on radio: {token}")


def test_app():
    root = Path(__file__).resolve().parent.parent
    at = AppTest.from_file(str(root / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, f"initial run failed: {at.exception}"

    # ---- Finder: apply a Toyota filter, raise a weight, expect a result table
    _goto(at, "Finder")
    assert not at.exception, at.exception
    _by_key(at.multiselect, "f_makes").set_value(["Toyota"]).run()
    assert not at.exception, at.exception
    w = _by_key(at.slider, "w_power")
    if w is not None:
        w.set_value(5).run()
        assert not at.exception, at.exception
    # a scored table should be present
    assert len(at.dataframe) >= 1, "Finder should render a results dataframe"

    # ---- Similar: pick a car, expect neighbour comparison table
    _goto(at, "Similar Models")
    assert not at.exception, at.exception
    _by_key(at.selectbox, "s_make").set_value("Chevrolet").run()
    assert not at.exception, at.exception
    _by_key(at.selectbox, "s_model").set_value("Corvette").run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1, "Similar should render a comparison dataframe"

    # ---- Evolution: default make/model renders (plotly figures are built
    # inside the run; any error would surface as at.exception)
    _goto(at, "Evolution")
    assert not at.exception, at.exception

    # ---- Rankings: switch metric, expect table
    _goto(at, "Rankings")
    assert not at.exception, at.exception
    _by_key(at.selectbox, "r_metric").set_value("economy").run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1, "Rankings should render a table"

    # ---- Market: switch metric
    _goto(at, "Market")
    assert not at.exception, at.exception
    _by_key(at.selectbox, "m_metric").set_value("curb_weight_kg").run()
    assert not at.exception, at.exception

    print("OK — all 5 pages render and interact without errors")


if __name__ == "__main__":
    test_app()