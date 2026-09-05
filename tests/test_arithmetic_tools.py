from backend.app.agent.arithmetic_tools import (
    average,
    delta,
    percentage_of,
    rank_values,
    ratio,
    sum_values,
)


class TestSumValues:
    def test_basic_sum(self):
        assert sum_values.func([100.0, 250.5, 3.25]) == "$353.75"

    def test_non_currency(self):
        assert sum_values.func([3, 5, 7], as_currency=False) == "15.00"

    def test_empty_list_sums_to_zero(self):
        # sum([]) == 0 mathematically - unlike average, there's no
        # "undefined" case for a sum, so this deliberately isn't an error.
        assert sum_values.func([]) == "$0.00"


class TestAverage:
    def test_basic_average(self):
        assert average.func([100.0, 200.0]) == "$150.00"

    def test_non_currency(self):
        assert average.func([3, 5, 7], as_currency=False) == "5.00"

    def test_empty_list_returns_error_not_exception(self):
        # Unlike sum, an average of zero items is undefined - must not
        # raise ZeroDivisionError or silently return 0.
        assert average.func([]) == "Cannot compute an average of an empty list of values."


class TestPercentageOf:
    def test_basic_percentage(self):
        assert percentage_of.func(25.0, 200.0) == "12.5%"

    def test_whole_zero_returns_error_not_exception(self):
        assert percentage_of.func(5.0, 0.0) == "Cannot compute a percentage of a whole that is zero."


class TestDelta:
    def test_increase(self):
        assert delta.func(100.0, 150.0) == "$+50.00 (+50.0%), an increase"

    def test_decrease(self):
        assert delta.func(150.0, 100.0) == "$-50.00 (-33.3%), a decrease"

    def test_no_change(self):
        assert delta.func(100.0, 100.0) == "$+0.00 (+0.0%), no change"

    def test_before_zero_percentage_is_undefined_not_a_crash(self):
        result = delta.func(0.0, 50.0)
        assert result == "$+50.00 (undefined (started from zero)), an increase"

    def test_non_currency(self):
        assert delta.func(3, 7, as_currency=False) == "+4.00 (+133.3%), an increase"


class TestRatio:
    def test_basic_ratio_is_labeled_and_directional(self):
        result = ratio.func(650_000_000.0, "Alaska", 210_000_000.0, "Alabama")
        assert result == "Alaska is 3.1x Alabama"

    def test_b_zero_returns_error_not_exception(self):
        assert ratio.func(5.0, "X", 0.0, "Y") == "Cannot compute a ratio: Y is zero."


class TestRankValues:
    def test_descending_default(self):
        result = rank_values.func({"NSF": 12.4, "NASA": 30.1, "EPA": 20.0})
        assert result == "1. NASA: 30.10\n2. EPA: 20.00\n3. NSF: 12.40"

    def test_ascending(self):
        result = rank_values.func({"NSF": 12.4, "NASA": 30.1, "EPA": 20.0}, descending=False)
        assert result == "1. NSF: 12.40\n2. EPA: 20.00\n3. NASA: 30.10"

    def test_ties_broken_by_insertion_order(self):
        # Documented behavior: a stable sort, so among equal values the
        # first-inserted one ranks higher (in both descending and
        # ascending order) - Python's sorted() guarantees this even with
        # reverse=True.
        result = rank_values.func({"NASA": 30.1, "EPA": 30.1, "NSF": 12.4})
        assert result == "1. NASA: 30.10\n2. EPA: 30.10\n3. NSF: 12.40"

    def test_single_item(self):
        assert rank_values.func({"only": 5.0}) == "1. only: 5.00"

    def test_empty_dict_returns_message_not_exception(self):
        assert rank_values.func({}) == "No values to rank."


class TestDisambiguationDocstrings:
    # percentage_of, ratio, and delta are conceptually adjacent enough that
    # a model could reach for the wrong one - each docstring must name the
    # other two so that disambiguation can't silently drift out of the code
    # as these tools evolve.

    def test_percentage_of_names_ratio_and_delta(self):
        doc = percentage_of.func.__doc__
        assert "ratio" in doc
        assert "delta" in doc

    def test_ratio_names_percentage_of_and_delta(self):
        doc = ratio.func.__doc__
        assert "percentage_of" in doc
        assert "delta" in doc

    def test_delta_names_percentage_of_and_ratio(self):
        doc = delta.func.__doc__
        assert "percentage_of" in doc
        assert "ratio" in doc


class TestNoNetworkCalls:
    def test_none_of_the_six_tools_touch_the_network(self, monkeypatch):
        # These are pure arithmetic functions - unlike every other tool in
        # this project, they must never make an API call. Denying
        # socket.socket outright and calling all six with valid input
        # proves that, rather than just trusting that nobody added one.
        def deny_socket(*args, **kwargs):
            raise AssertionError("arithmetic tools must not perform network I/O")

        monkeypatch.setattr("socket.socket", deny_socket)

        sum_values.func([1.0, 2.0])
        average.func([1.0, 2.0])
        percentage_of.func(1.0, 2.0)
        delta.func(1.0, 2.0)
        ratio.func(1.0, "a", 2.0, "b")
        rank_values.func({"a": 1.0, "b": 2.0})
