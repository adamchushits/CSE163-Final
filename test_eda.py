"""
Adam Chu
CSE 163 AC
Test the returned strings from the EDA functions
"""

import pandas as pd
from Data_Analysis import analyze_data, cleaned_data_summary, set_stellar_type


def test_some_missing_values():
    """
    test the case where there are some missing values in the DataFrame.
    """
    data = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [None, 20, None, 40],
        "C": [None, None, None, 4],
    })
    expected = (
        "Number of columns: 3\n"
        "Number of rows: 4\n"
        "Number of missing values: 5\n"
        "Mean missing values percentage: 0.4166666666666667\n"
        "Missing values row: 3\n"
        "Missing values column: 2"
    )
    assert (analyze_data(data) == expected)


def test_no_missing_values():
    """
    test the case where there are no missing values in the DataFrame.
    """
    data = pd.DataFrame({"A": [0, 1, 2], "B": [3, 0, 5]})
    expected = (
        "Number of columns: 2\n"
        "Number of rows: 3\n"
        "Number of missing values: 0\n"
        "Mean missing values percentage: 0.0\n"
        "Missing values row: 0\n"
        "Missing values column: 0"
    )
    assert (analyze_data(data) == expected)


def test_all_missing_values():
    """
    Every cell, row, and column should be counted as missing.
    """
    data = pd.DataFrame({"A": [None, None, None],
                            "B": [None, None, None]})
    expected = (
        "Number of columns: 2\n"
        "Number of rows: 3\n"
        "Number of missing values: 6\n"
        "Mean missing values percentage: 1.0\n"
        "Missing values row: 3\n"
        "Missing values column: 2"
    )
    assert (analyze_data(data) == expected)



def check_summary(data, values):
    """
    Format known statistics to compare with the returned summary.
    """
    expected_stats = pd.Series(
        values,
        index=["count", "mean", "std", "min", "25%", "50%",
                "75%", "max"],
        name=data.name,
        dtype=float,
    )
    expected = f"{data.name} Description:\n {expected_stats}"
    assert (cleaned_data_summary(data) == expected)


def test_statistics():
    """
    test cleaned data summary with a simple series.
    """
    data = pd.Series([2, 4, 6], name="orbital_period")
    check_summary(data, [3, 4, 2, 2, 3, 4, 5, 6])


def test_constant_values():
    """
    test a constant series
    """
    data = pd.Series([7, 7, 7, 7], name="ratio")
    check_summary(data, [4, 7, 0, 7, 7, 7, 7, 7])


def test_set_stellar_type():
    """
    Test stellar types and temperature boundaries.
    """
    temperatures = [
        3000, 4500, 5500, 6500, 8500, 15000, 40000,
        3899.9, 3900, 5299.9, 5300, 5999.9, 6000,
        7299.9, 7300, 9999.9, 10000, 32999.9, 33000
    ]
    expected = [
        "M", "K", "G", "F", "A", "B", "O",
        "M", "K", "K", "G", "G", "F",
        "F", "A", "A", "B", "B", "O"
    ]
    for i in range(len(temperatures)):
        data = pd.Series({"st_teff": temperatures[i]})
        result = set_stellar_type(data)
        assert result == expected[i]


def main():
    test_some_missing_values()
    test_no_missing_values()
    test_all_missing_values()
    test_statistics()
    test_constant_values()
    test_set_stellar_type()
    print("All tests passed!")


if __name__ == '__main__':
    main()
