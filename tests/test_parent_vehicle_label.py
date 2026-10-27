from __future__ import annotations

from backend.app.agent.tools.spending.search import _parent_vehicle_label


def test_orders_name_their_parent_vehicle_and_their_own_id():
    label = _parent_vehicle_label("DOC003", "CONT_AWD_DOC003_1323_DOCYA132315BU0033_1323")
    assert "DOCYA132315BU0033" in label and "own ID is DOC003" in label


def test_standalone_contracts_and_assistance_get_no_label():
    assert _parent_vehicle_label("80GSFC18C0120", "CONT_AWD_80GSFC18C0120_8000_-NONE-_-NONE-") == ""
    assert _parent_vehicle_label("NNX16AO69A", "ASST_NON_NNX16AO69A_8000") == ""
