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
        self._operands = ['\d+\.\d+', '^\d+', 'x', 'y']
        self._operators = ['+', '-', '*', '/', '^', '%', 'sin(', 'cos(', 'tan(', 'sqrt(', '(', ')']
        self._operator_weights = {'sqrt(': 4, 'cos(': 4, 'sin(': 4, 'tan(': 4, '(': 4, '^': 4 , '*': 3,
                    '/': 3, '%': 3, '+': 2, '-': 2, 'x': 1, 'y': 1, ')': 0, '\0' : 0}

    def parse_formula(self, formula, x, y):
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
        numpy.ndarray or None
            If no errors occur, return computed array, otherwise None.
        """

        operands_string = "|".join(map(lambda x: '^' + x, self._operands))
        escaped_operators_string = "|".join(map(lambda x: '^' + re.escape(x), self._operators))

        formula_list = []
        formula = re.sub(" ", "", formula)
        if not self._formula_check(formula):
            return None
        while formula != "":
            # OP parsing
            if re.match(operands_string + "|" + escaped_operators_string, formula):
                op = re.match(operands_string + "|" + escaped_operators_string, formula).group()
                try:
                    converted_op = float(op)
                except:
                    converted_op = op
                if len(formula_list) != 0 and \
                    type(formula_list[-1]) is float and \
                    (converted_op in ['x', 'y', '(']):
                    formula_list.append('*')
                formula_list.append(converted_op)

                formula = formula[len(op):]
            # Invalid OP
            else:
                return None
        return self._formula_compute(formula_list, x, y)

    def _formula_check(self, formula):
        left_bracket = len(re.findall('\(', formula))
        right_bracket = len(re.findall('\)', formula))
        return left_bracket == right_bracket

    def _formula_compute(self, segmented_formula, x, y):
        """Compute formula by stack parsing.

        Parameters
        ----------
        segmented_formula : list
            Segmented formula.
        x : numpy.ndarray
            Grid of axis y array.
        y : numpy.ndarray
            Grid of axis x array.

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
            elif op == 'x':
                stack.append(x)
            elif op == 'y':
                stack.append(y)
            else:
                stack.append(op)
                stack = self._refresh_stack(stack)
        stack.append('\0')
        
        # If no errors, the stack will remain the result and '\0' symbol,
        # otherwise continue refreshing the stack until watchdog counter exceed.
        while len(stack) != 2:
            if watchdog > 20:
                return None
            watchdog += 1
            stack = self._refresh_stack(stack)
        return stack[0]

    def _compute_op(self, **kwargs):
        """Compute a single operand-operator pair.

        Returns
        -------
        float or numpy.ndarray
            Computed result.    
        """

        if 'number2' in kwargs:
            if kwargs['operator'] == '+':
                return kwargs['number1'] + kwargs['number2']
            if kwargs['operator'] == '-':
                return kwargs['number1'] - kwargs['number2']
            if kwargs['operator'] == '*':
                return kwargs['number1'] * kwargs['number2']
            if kwargs['operator'] == '/':
                return kwargs['number1'] / kwargs['number2']
            if kwargs['operator'] == '%':
                return kwargs['number1'] % kwargs['number2']
            if kwargs['operator'] == '^':
                return kwargs['number1'] ** kwargs['number2']
        else:
            if kwargs['operator'] == '(':
                return kwargs['number1']
            if kwargs['operator'] == 'sin(':
                return np.sin(kwargs['number1'])
            if kwargs['operator'] == 'cos(':
                return np.cos(kwargs['number1'])
            if kwargs['operator'] == 'tan(':
                return np.tan(kwargs['number1'])
            if kwargs['operator'] == 'sqrt(':
                return np.sqrt(kwargs['number1'])
        
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
         and (type(stack[-2]) is float or type(stack[-2]) is np.ndarray)\
         and (type(stack[-4]) is float or type(stack[-4]) is np.ndarray)\
         and self._operator_weights[stack[-1]] <= self._operator_weights[stack[-3]]: # normal condition
            currentOP = stack.pop()
            number2 = stack.pop()
            operator = stack.pop()
            number1 = stack.pop()
            stack.append(self._compute_op(number1=number1, number2=number2, operator=operator))
            stack.append(currentOP)

        if len(stack) > 2 and stack[-1] == ')' and type(stack[-3]) is str: # bracket condition
            if self._operator_weights[stack[-3]] != 4:  # not (x)
                stack = self._refresh_stack(stack)
            elif self._operator_weights[stack[-3]] == 4:
                stack.pop()
                number1 = stack.pop()
                operator = stack.pop()
                stack.append(self._compute_op(number1=number1, operator=operator))
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
            logging.warning('Invalid Dimension 3D, ignore it.')
        
    for i in dimention2D:
        try:
            Y2 = float(i["A"])*(X2 ** float(i["Xexp"])) + float(i["C"])
            Z2 = float(i["Z"])
            ax.plot(X2, Y2, Z2)
        except:
            logging.warning('Invalid Dimension 2D, ignore it.')
    for i in advance3D:
        Z1 = adv3d.parse_formula(i.lower(), X1, Y1)
        if type(Z1) is np.ndarray:
            ax.plot_surface(X1, Y1, Z1, rstride=1, cstride=1, cmap='rainbow')
        else:
            logging.warning('Invalid Advance 3D, ignore it.')

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
    generate_plot([], [], ["2x^3+2y^^3"])
