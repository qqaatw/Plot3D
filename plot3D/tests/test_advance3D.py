import pytest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .. import main


# Initialize essentials
ax = Axes3D(plt.figure())
X1, Y1 = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))  # 3D

SKIP_TEST = "pass"

test_cases = [
    # basic
    ("sin(x)+cos(y)",
    ["sin(", "x", ")", "+", "cos(", "y", ")"],
    (np.sin(X1)+np.cos(Y1))),
    # website example
    ("-3sin(x)+2+8cos(-y)^2+ln(5)",
    SKIP_TEST,
    (-3 * np.sin(X1) + 2 + 8 * np.cos(-Y1) ** 2 + np.log(5))
    ),
    # log
    ("ln(e ^ 2) * log10(100) + log2(8)",
    SKIP_TEST,
    (np.log(np.e ** 2) * np.log10(100) + np.log2(8))
    ),
    # formula with single operand
    ("(x)",
    ["(", "x", ")"],
    (X1)),
    # formula with spaces
    ("x * y",
    ["x", "*", "y"],
    (X1 * Y1)),
    # formula with negative number
    ("(-x)^2+(-y^2)^2",
    ['(', '(', 0.0, '-', 1.0, ')', '*', 'x', ')', '^', 2.0, '+', '(', '(', 0.0, '-', 1.0, ')', '*', 'y', '^', 2.0, ')', '^', 2.0],
    ((-X1) ** 2 + (-Y1 ** 2) ** 2)),
    # formula with omitted multipler
    ("2pixex",
    [2.0, "*",  "pi", "*", "x", "*", "e", "*", "x"],
    (2.0 * np.pi * X1 * np.e * X1)),
    # formula with floating point
    ("2.5 / 3.5",
    [2.5, "/", 3.5],
    np.array(2.5/3.5)),
    # formulas with invalid bracket
    ("sin(x))",
    None,
    (None)),
    ("(((x+y))",
    None,
    (None)),
    # formula with invalid op
    ("x++y",
    ["x", "+", "+", "y"],
    (None))
]

class TestAdvance3D:
    def test_init(self):
        adv3d = main.Advance3D()
        # Test duplicate elements of operators
        assert len(adv3d._operators) == len(set(adv3d._operators))
        # Test whether operator_weights' elements in operator list. 
        for key in adv3d._operator_weights.keys():
            if key !='\0':
                assert key in adv3d._operators
    
    @pytest.mark.parametrize("test_input, expected_1, expected_2", test_cases)
    def test_parse_formula(self, test_input, expected_1, expected_2):
        if expected_1 == SKIP_TEST:
            pytest.skip("Encounter SKIP_TEST, skip.")
        adv3d = main.Advance3D()
        result = adv3d.parse_formula(test_input, None, None, debug=True)
        assert result == expected_1

    @pytest.mark.parametrize("test_input, expected_1, expected_2", test_cases)
    def test_formula_compute(self, test_input, expected_1, expected_2):
        if expected_2 == SKIP_TEST:
            pytest.skip("Encounter SKIP_TEST, skip.")
        adv3d = main.Advance3D()
        result = adv3d.parse_formula(test_input, X1, Y1, debug=False)
        if expected_2 is None:
            assert result is None
        else:
            assert result.all() == expected_2.all()
    #    main.Advance3D.
