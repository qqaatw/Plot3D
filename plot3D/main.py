import re
import uuid
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')

class Advance3D:
    """Parsing 3D formula.
    """

    def __init__(self, *args, **kwargs):
        self._constants = {'x' : None, 'y': None, 'pi': np.pi, 'e' : np.e}

        self._bracket_operators = ['sin(', 'cos(', 'tan(', 'sqrt(', 'log2(', 'log10(', 'ln(', '(']
        self._non_bracket_operators = ['+', '-', '*', '/', '^', '%', ')']
        
        self._operands = ['\d+\.\d+', '^\d+'] + list(self._constants.keys())
        self._operators = self._bracket_operators + self._non_bracket_operators
        self._operator_weights = {
            'sqrt(': 4, 'cos(': 4, 'sin(': 4, 'tan(': 4, 'log2(': 4, 'log10(': 4, 'ln(': 4,
            '(': 4, '^': 4 , '*': 3, '/': 3, '%': 3, '+': 2, '-': 2, ')': 1, '\0' : 0}

    def parse_formula(self, formula, x, y, debug=False):
        """Parse a formula from string.

        Parameters
        ----------
        formula : str
            Formula string.
        x : numpy.ndarray
            Grid of axis y array.
        y : numpy.ndarray
            Grid of axis x array.

        Returns
        -------
        if debug:
            list
                Return formula_list for debugging.
        otherwise:
            numpy.ndarray or None
                If no errors occur, return computed array, otherwise None.
        """

        self._constants['x'], self._constants['y'] = x, y

        operands_string = "|".join(map(lambda x: '^' + x, self._operands))
        escaped_operators_string = "|".join(map(lambda x: '^' + re.escape(x), self._operators))

        formula_list = []
        # Handle negative number
        formula = re.sub("^\-", "(0-1)*", formula.replace(" ", "").lower())
        formula = re.sub("\(\-", "((0-1)*", formula)

        if not self._formula_check(formula):
            return None
        while formula != "":
            # OP parsing
            match = re.match(operands_string + "|" + escaped_operators_string, formula)
            if match:
                op = match.group()
                try:
                    converted_op = float(op)
                except:
                    converted_op = op
                
                if len(formula_list) != 0:
                    # Handle omitted multipler
                    if re.match(operands_string, str(formula_list[-1])) and \
                        converted_op in (list(self._constants.keys()) + self._bracket_operators):
                        formula_list.append('*')

                formula_list.append(converted_op)
                formula = formula[len(op):]
            # Invalid OP
            else:
                return None
        
        if debug:
            return formula_list
        else:
            return self._formula_compute(formula_list)

    def _formula_check(self, formula):
        """Check whether formula is valid.

        Parameters
        ----------
        formula : str
            Formula string.

        Returns
        -------
        bool
            Return True if formula is valid, otherwise False. 
        """

        left_bracket = len(re.findall('\(', formula))
        right_bracket = len(re.findall('\)', formula))
        return left_bracket == right_bracket

    def _formula_compute(self, segmented_formula):
        """Compute formula by stack parsing.

        Parameters
        ----------
        segmented_formula : list
            Segmented formula.

        Returns
        -------
        numpy.ndarray
            Computed array.
        """

        stack = []
        watchdog = 0
        for op in segmented_formula:
            if type(op) is float:
                stack.append(op)
            elif op in self._constants:
                stack.append(self._constants[op])
            else:
                stack.append(op)
                stack = self._refresh_stack(stack)
        stack.append('\0')
        
        # If no errors, the stack will remain the result and '\0' symbol,
        # otherwise continue refreshing the stack until the watchdog counter exceeds.
        while len(stack) != 2:
            if watchdog > 20:
                return None
            watchdog += 1
            stack = self._refresh_stack(stack)
        return np.array(stack[0])

    def _compute_op(self, operand_1, operator, operand_2=None):
        """Compute a single operand-operator pair.

        Parameters
        ----------
        operand_1 : float or numpy.ndarray
            First operand.
        operator : str
            Operator.
        operand_2 : float or numpy.ndarray, optional
            Second operand, by default None.

        Returns
        -------
        float or numpy.ndarray
            Computed result.    
        """

        if operand_2 is None:
            if operator == '(':
                return operand_1
            if operator == 'sin(':
                return np.sin(operand_1)
            if operator == 'cos(':
                return np.cos(operand_1)
            if operator == 'tan(':
                return np.tan(operand_1)
            if operator == 'sqrt(':
                return np.sqrt(operand_1)
            if operator == 'log2(':
                return np.log2(operand_1)
            if operator == 'log10(':
                return np.log10(operand_1)
            if operator == 'ln(':
                return np.log(operand_1)
        else:
            if operator == '+':
                return operand_1 + operand_2
            if operator == '-':
                return operand_1 - operand_2
            if operator == '*':
                return operand_1 * operand_2
            if operator == '/':
                return operand_1 / operand_2
            if operator == '%':
                return operand_1 % operand_2
            if operator == '^':
                return operand_1 ** operand_2
        
    def _refresh_stack(self, stack):
        """Refresh stack, according to the conditions.

        Parameters
        ----------
        stack : list
            Working stack.

        Returns
        -------
        list
            Working stack.
        """

        while len(stack) > 3\
         and (isinstance(stack[-2], float) or type(stack[-2]) is np.ndarray)\
         and (isinstance(stack[-4], float) or type(stack[-4]) is np.ndarray)\
         and self._operator_weights[stack[-1]] <= self._operator_weights[stack[-3]]: # normal condition
            currentOP = stack.pop()
            number2 = stack.pop()
            operator = stack.pop()
            number1 = stack.pop()
            stack.append(self._compute_op(operand_1=number1, operator=operator, operand_2=number2))
            stack.append(currentOP)

        if len(stack) > 2 and stack[-1] == ')' and type(stack[-3]) is str: # bracket condition
            if self._operator_weights[stack[-3]] != 4:  # not (single number)
                stack = self._refresh_stack(stack)
            elif self._operator_weights[stack[-3]] == 4:  # (single number)
                stack.pop()
                number1 = stack.pop()
                operator = stack.pop()
                stack.append(self._compute_op(operand_1=number1, operator=operator))
        return stack


def generate_plot(dimention3D, dimention2D, advance3D):
    """Generate plot by three methods.

    Parameters
    ----------
    dimention3D : list of dicts
        List of 3D dict to compute.
    dimention2D : list of dicts
        List of 2D dict to compute.
    advance3D : list of strs
        List of formula string to compute.

    Returns
    -------
    str
        Plot's uuid, which is saved in ./images folder. 
    """

    ax = Axes3D(plt.figure())
    adv3d = Advance3D()

    # 3D
    X1 = np.arange(-5, 5, 0.25)  
    Y1 = np.arange(-5, 5, 0.25)  
    X1, Y1 = np.meshgrid(X1, Y1)
    
    # 2D
    X2 = np.arange(-5, 5, 0.25)
    
    for i in dimention3D:
        try:
            Z1 = float(i["A"])*(X1 ** float(i["Xexp"])) + float(i["B"])*(Y1 ** float(i["Yexp"])) + float(i["C"])
            ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, cmap='rainbow')
        except:
            logging.info('Invalid Dimension 3D, ignore it.')
        
    for i in dimention2D:
        try:
            Y2 = float(i["A"])*(X2 ** float(i["Xexp"])) + float(i["C"])
            Z2 = float(i["Z"])
            ax.plot(X2, Y2, Z2)
        except:
            logging.info('Invalid Dimension 2D, ignore it.')
    for i in advance3D:
        try:
            Z1 = adv3d.parse_formula(i, X1, Y1)
            ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, cmap='rainbow')
        except:
            logging.info('Invalid Advance 3D, ignore it.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    uid = uuid.uuid1()
 
    plt.savefig('./plot3D/images/'+str(uid)+'.png', bbox_inches='tight', format="png", dpi=300)
    return uid


if __name__ == "__main__":
    ax = Axes3D(plt.figure())
    X1 = np.arange(-5, 5, 0.25)  # 3D
    Y1 = np.arange(-5, 5, 0.25)  # 3D
    X1, Y1 = np.meshgrid(X1, Y1)  # 3D
    
    adv3d = Advance3D()
    generate_plot([], [], ["ln(e ^ 2) * log10(100) + log2(8)"]) # ["2x^2+2y^2"]
