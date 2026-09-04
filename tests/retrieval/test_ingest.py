from backend.app.retrieval.pipeline.ingest import (
    chunk_unit,
    clean_page_text,
    detect_repeated_lines,
    question_split,
)


class TestDetectRepeatedLines:
    def test_line_repeated_across_pages_is_detected(self):
        pages = ["Header\nContent A", "Header\nContent B", "Header\nContent C"]
        repeated = detect_repeated_lines(pages, min_count=3)
        assert "Header" in repeated

    def test_line_below_min_count_is_not_detected(self):
        pages = ["Header\nContent A", "Header\nContent B", "Content C only"]
        repeated = detect_repeated_lines(pages, min_count=3)
        assert "Header" not in repeated

    def test_numeric_only_lines_are_excluded_even_if_repeated(self):
        # page numbers repeat across every page but must never be flagged as
        # a "repeated header/footer" candidate the same way text is
        pages = ["1\nContent", "2\nContent", "3\nContent"]
        repeated = detect_repeated_lines(pages, min_count=3)
        assert "1" not in repeated
        assert "2" not in repeated


class TestCleanPageText:
    def test_strips_page_n_of_m_footer_regardless_of_number(self):
        # this was a real bug: exact-repeat detection alone missed these
        # because the page number makes each footer line unique
        text = "Some content\nANALYST'S GUIDE TO FEDERAL SPENDING DATA\nPAGE 3 OF 20"
        cleaned = clean_page_text(text, repeated_lines=set())
        assert "PAGE 3 OF 20" not in cleaned
        assert "Some content" in cleaned

    def test_strips_bare_page_n_footer(self):
        text = "Some content\nPage 7"
        cleaned = clean_page_text(text, repeated_lines=set())
        assert "Page 7" not in cleaned

    def test_strips_pure_numeric_lines(self):
        text = "Some content\n42"
        cleaned = clean_page_text(text, repeated_lines=set())
        assert "42" not in cleaned

    def test_strips_lines_in_repeated_set(self):
        text = "Some content\nRUNNING HEADER"
        cleaned = clean_page_text(text, repeated_lines={"RUNNING HEADER"})
        assert "RUNNING HEADER" not in cleaned

    def test_strips_separator_lines(self):
        text = "Some content\n----------\nMore content"
        cleaned = clean_page_text(text, repeated_lines=set())
        assert "----------" not in cleaned

    def test_preserves_normal_content(self):
        text = "This is a normal sentence about federal spending."
        cleaned = clean_page_text(text, repeated_lines=set())
        assert cleaned == text


class TestQuestionSplit:
    def test_single_question_with_answer(self):
        text = "‘What is a prime award?’\nA prime award is an agreement."
        units = question_split(text)
        assert len(units) == 1
        assert "What is a prime award?" in units[0]
        assert "A prime award is an agreement." in units[0]

    def test_leading_section_header_kept_as_own_unit(self):
        text = "AWARD SPENDING\n‘What is a prime award?’\nAnswer text."
        units = question_split(text)
        assert units[0] == "AWARD SPENDING"
        assert len(units) == 2

    def test_multiple_subquestions_under_one_quote_stay_together(self):
        # the guide's real pattern: several "What is X?" fragments share a
        # single opening quote with no quote characters in between, and
        # should end up as ONE unit along with their shared answer
        text = (
            "‘What is a prime award? What is a sub-award?’\n"
            "A prime award is an agreement. A sub-award is different."
        )
        units = question_split(text)
        assert len(units) == 1
        assert "What is a sub-award?" in units[0]

    def test_malformed_closing_quote_does_not_create_a_false_split(self):
        # real bug found in the source PDF: some closing quotes are
        # mis-rendered using the *opening* quote glyph. A closing-quote-pair
        # matcher broke on this; the current start-marker approach must not
        # treat "‘ <lowercase / non-question-word text>" as a new question.
        text = (
            "‘How do the face value and subsidy cost of loans impact "
            "the value of award spending? ‘\n"
            "Positive loan subsidy costs are included in obligations."
        )
        units = question_split(text)
        assert len(units) == 1
        assert "Positive loan subsidy costs" in units[0]

    def test_inline_quoted_word_is_not_treated_as_a_question(self):
        # e.g. "...also called ‘base’ contract..." must not split
        text = (
            "‘What is a prime award transaction?’\n"
            "It can be the initial (also called ‘base’) contract."
        )
        units = question_split(text)
        assert len(units) == 1

    def test_no_questions_returns_single_unit(self):
        text = "Just a plain paragraph with no questions in it."
        units = question_split(text)
        assert units == [text]

    def test_empty_text_returns_empty_list(self):
        assert question_split("") == []
        assert question_split("   ") == []


class TestChunkUnit:
    def test_short_unit_passes_through_unchanged(self):
        unit = "A short question and answer."
        assert chunk_unit(unit, max_chars=1000) == [unit]

    def test_long_unit_splits_into_multiple_chunks(self):
        sentence = "This is one sentence about federal spending data. "
        unit = sentence * 50  # well over max_chars
        chunks = chunk_unit(unit, max_chars=200, overlap=50)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 300  # max_chars + reasonable overlap slack

    def test_chunks_preserve_all_sentences(self):
        sentence = "Sentence number {}. "
        unit = "".join(sentence.format(i) for i in range(20))
        chunks = chunk_unit(unit, max_chars=100, overlap=20)
        combined = " ".join(chunks)
        for i in range(20):
            assert f"Sentence number {i}." in combined
