from backend.app.retrieval.pipeline.ingest_glossary import (
    entry_to_chunk,
    extract_related_slugs,
)


class TestExtractRelatedSlugs:
    def test_none_returns_empty(self):
        assert extract_related_slugs(None) == []

    def test_no_glossary_links_returns_empty(self):
        # An external link (e.g. an FPDS help page), not a cross-reference.
        assert extract_related_slugs("Learn more from the [FPDS IDC page](https://www.fpds.gov/help/x.htm)") == []

    def test_single_glossary_link(self):
        assert extract_related_slugs("See also:\n\n- [Federal Account](?glossary=federal-account)") == [
            "federal-account"
        ]

    def test_multiple_glossary_links_mixed_with_external(self):
        resources = (
            "See also:\n\n"
            "- [Account Balance (File A)](?glossary=account-balance-file-a)\n"
            "- [Awards Data (File D)](?glossary=awards-data-file-d)\n\n"
            "For more information, see our [Data Dictionary](https://www.usaspending.gov/data-dictionary)"
        )
        assert extract_related_slugs(resources) == ["account-balance-file-a", "awards-data-file-d"]


class TestEntryToChunk:
    def test_basic_entry(self):
        entry = {
            "term": "Transaction",
            "slug": "transaction",
            "data_act_term": None,
            "plain": "A transaction can be the initial contract, grant, loan, or insurance award.",
            "official": None,
            "resources": None,
        }
        chunk = entry_to_chunk(entry)
        assert chunk.id == "Glossary_transaction"
        assert chunk.source == "USASpending Glossary"
        assert chunk.term == "Transaction"
        assert chunk.slug == "transaction"
        assert chunk.related_slugs == []
        assert chunk.page_start == 0
        assert chunk.page_end == 0
        assert chunk.paragraph_index == 0
        assert chunk.text == "Transaction\n\nA transaction can be the initial contract, grant, loan, or insurance award."

    def test_official_appended_when_it_differs_from_plain(self):
        entry = {
            "term": "Treasury Account Symbol (TAS)",
            "slug": "treasury-account-symbol-tas",
            "data_act_term": "Treasury Account Symbol",
            "plain": "Treasury and OMB assign a code to each appropriation.",
            "official": "The account identification codes assigned by the Department of the Treasury.",
            "resources": "See also:\n\n- [Federal Account](?glossary=federal-account)",
        }
        chunk = entry_to_chunk(entry)
        assert "Official definition: The account identification codes" in chunk.text
        assert chunk.related_slugs == ["federal-account"]

    def test_official_not_duplicated_when_identical_to_plain(self):
        entry = {
            "term": "Ultimate Parent Legal Entity Name",
            "slug": "ultimate-parent-legal-entity-name",
            "data_act_term": "Ultimate Parent Legal Entity Name",
            "plain": "The name of the ultimate parent of the awardee or recipient.",
            "official": "The name of the ultimate parent of the awardee or recipient.",
            "resources": None,
        }
        chunk = entry_to_chunk(entry)
        assert chunk.text.count("The name of the ultimate parent") == 1
        assert "Official definition" not in chunk.text

    def test_data_act_term_included_in_heading_when_it_differs_from_term(self):
        entry = {
            "term": "Treasury Account Symbol (TAS)",
            "slug": "treasury-account-symbol-tas",
            "data_act_term": "Treasury Account Symbol",
            "plain": "Treasury and OMB assign a code to each appropriation.",
            "official": None,
            "resources": None,
        }
        chunk = entry_to_chunk(entry)
        assert chunk.text.startswith("Treasury Account Symbol (TAS) (DATA Act term: Treasury Account Symbol)")

    def test_data_act_term_omitted_from_heading_when_same_as_term(self):
        entry = {
            "term": "Ultimate Parent Legal Entity Name",
            "slug": "ultimate-parent-legal-entity-name",
            "data_act_term": "Ultimate Parent Legal Entity Name",
            "plain": "The name of the ultimate parent of the awardee or recipient.",
            "official": None,
            "resources": None,
        }
        chunk = entry_to_chunk(entry)
        assert chunk.text.startswith("Ultimate Parent Legal Entity Name\n\n")
        assert "DATA Act term" not in chunk.text
