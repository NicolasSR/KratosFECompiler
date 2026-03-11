import re

import sympy as sp
from sympy.codegen.rewriting import create_expand_pow_optimization

# We expand terms up to 8th power for optimization purposes
expand_opt = create_expand_pow_optimization(8)

# Output functions
def _Indentation(indentation_level):
    """Returns the indentation string."""
    return "    " * indentation_level

def _CodeGen(language, value):
    return  {
        "c"     : sp.ccode,
        "python": sp.pycode
    }[language](value)

def _VariableDeclaration(language, variable_name, variable_expression):
    """"
    Returns the variable declaration, without indentation nor suffix.

    The expression must have been turned into code already.
    """
    return  {
        "c"     : "const double {name} = {expr}",
        "python": "{name} = {expr}"
    }[language].format(name=variable_name, expr=variable_expression)

def _Suffix(language):
    """Returns the endline suffix."""
    return  {
        "c"     : ";\n",
        "python": "\n"
    }[language]

def _ReplaceIndices(language, expression):
    """Replaces array access with underscored variable.

    For vectors:  `variable_3`   becomes `variable[3]`
    For matrices: `variable_3_7` becomes `variable[3,7]`
    For lists of matrices: `variable_0_3_7` becomes `variable[0][3,7]`

    The accessor for matrices is chosen according to the language (`[]` vs. `()`)
    """
    #Lists of matrices
    pattern = r"_(\d+)_(\d+)_(\d+)"
    replacement = r"[\1][\2,\3]" if language == 'python' else r"[\1](\2,\3)"
    expression = re.sub(pattern, replacement, expression)

    #Matrices
    pattern = r"_(\d+)_(\d+)"
    replacement = r"[\1,\2]" if language == 'python' else r"(\1,\2)"
    expression = re.sub(pattern, replacement, expression)

    # Vectors
    pattern = r"_(\d+)"
    replacement = r"[\1]"
    expression = re.sub(pattern, replacement, expression)

    return expression

def OutputScalar(scalar_expression, name, language, indentation_level=0, replace_indices=True, assignment_op="="):
    """
    This function generates code to assign to a (pre-declared) scalar

    Keyword arguments:
    - scalar_expression -- A scalar
    - name -- The name of the variables
    - language -- The language of output
    - indentation_level -- The number of tabulations considered
    - replace_indices -- Set to `True` to replace matrix[i,j] with matrix_i_j (And similarly for vectors)
    - assignment_op -- The assignment operation
    """
    prefix = _Indentation(indentation_level)
    suffix = _Suffix(language)

    fmt = prefix + "{var}{op}{expr}" + suffix

    expression = _CodeGen(language, scalar_expression)
    outstring = fmt.format(var=name, op=assignment_op, expr=expression)

    if replace_indices:
        outstring = _ReplaceIndices(language, outstring)

    return outstring

def OutputVector(vector_expression, name, language, indentation_level=0, replace_indices=True, assignment_op="="):
    """
    This function generates code to fill a (pre-declared) vector.

    Keyword arguments:
    - rhs -- The RHS vector
    - name -- The name of the variables
    - language -- The language of output
    - indentation_level -- The number of tabulations considered
    - replace_indices -- Set to `True` to replace matrix_i_j with matrix[i,j] (And similarly for vectors)
    - assignment_op -- The assignment operation
    """
    prefix = _Indentation(indentation_level)
    suffix = _Suffix(language)
    fmt = prefix + "{var}[{i}]{op}{expr}" + suffix

    outstring = str("")
    for i in range(vector_expression.shape[0]):
        expression = _CodeGen(language, vector_expression[i,0])
        outstring += fmt.format(var=name, i=i, op=assignment_op, expr=expression)

    if replace_indices:
        outstring = _ReplaceIndices(language, outstring)

    return outstring


def OutputMatrix(matrix_expression, name, language, indentation_level=0, replace_indices=True, assignment_op="="):
    """
    This function generates code to fill a (pre-declared) matrix.

    Keyword arguments:
    - matrix_expression -- The matrix
    - name -- The name of the variables
    - language -- The language of output
    - indentation_level -- The number of tabulations considered
    - replace_indices -- Set to `True` to replace `matrix_i_j` with `matrix[i,j]` (And similarly for vectors)
    - assignment_op -- The assignment operation
    """
    prefix = _Indentation(indentation_level)
    suffix = _Suffix(language)

    fmt = prefix \
          + ("{var}[{i},{j}]{op}{expr}" if language == "python" else "{var}({i},{j}){op}{expr}") \
          + suffix

    outstring = str("")
    for i in range(matrix_expression.shape[0]):
        for j in range(matrix_expression.shape[1]):
            expression = _CodeGen(language, matrix_expression[i,j])
            outstring += fmt.format(var=name, i=i, j=j, op=assignment_op, expr=expression)

    if replace_indices:
        outstring = _ReplaceIndices(language, outstring)

    return outstring

def OutputSymbolicVariable(expression, language, replace_indices=True):
    """
    This function generates code from an expression..

    Keyword arguments:
    - expression -- The expression to generate code from
    - language -- The language of output
    - indentation_level -- The number of tabulations considered
    - max_index -- The maximum index
    - replace_indices -- Set to `True` to replace matrix[i,j] with matrix_i_j (And similarly for vectors)
    """
    outstring = _CodeGen(language, expression) + _Suffix(language)

    if replace_indices:
        outstring = _ReplaceIndices(language, outstring)

    return outstring

def OutputSymbolicVariableDeclaration(expression, name, language, indentation_level=0, replace_indices=True):
    """
    This function generates code to declare and assign an expression, such as:

    ```C++
        const double variable = expression;

    ```

    Keyword arguments:
    - expression -- The variable to define symbolic
    - language -- The language of output
    - name -- The name of the variables
    - indentation_level -- The number of tabulations considered
    - max_index -- DEPRECATED The maximum index
    - replace_indices -- Set to `True` to replace matrix[i,j] with matrix_i_j (And similarly for vectors)
    """
    prefix = _Indentation(indentation_level)
    value = _CodeGen(language, expression)
    expr = _VariableDeclaration(language, name, value)
    suffix = _Suffix(language)

    outstring = prefix + expr + suffix

    if replace_indices:
        outstring = _ReplaceIndices(language, outstring)

    return outstring

def _AuxiliaryOutputCollectionFactors(A, name, language, indentation_level, optimizations, replace_indices, assignment_op, output_func):
    """
    This method collects the constants of the replacement for matrices, vectors and scalars.

    Keyword arguments:
    - A -- The  factors
    - name -- The name of the constant
    - language -- The language of replacement
    - indentation_level -- The depth of the indentation (4 spaces per level)
    - optimizations -- The level of optimizations
    - replace_indices -- Set to `True` to replace matrix[i,j] with matrix_i_j (And similarly for vectors)
    - assignment_op -- The assignment operation
    - output_func -- The output function. Must have the same signature as OutputMatrix and OutputVector
    """
    symbol_name = "c" + name
    A_factors, A_collected = sp.cse(A, sp.numbered_symbols(symbol_name), optimizations)
    A = A_collected[0] #overwrite A with the one with the collected components
    A = expand_opt(A)

    Acoefficient_str = str("")
    for factor in A_factors:
        varname = str(factor[0])
        value = factor[1]
        value = expand_opt(value)
        Acoefficient_str += OutputSymbolicVariableDeclaration(value, varname, language, indentation_level, replace_indices)

    A_out = Acoefficient_str + output_func(A, name, language, indentation_level, replace_indices, assignment_op)
    return A_out


def OutputMatrix_CollectingFactors(A, name, language, indentation_level=0, max_index=None, optimizations='basic', replace_indices=True, assignment_op="="):
    """
    This method collects the constants of the replacement for matrices.

    Keyword arguments:
    - A -- The  factors
    - name -- The name of the constant
    - language -- The language of replacement
    - indentation_level -- The depth of the indentation (4 spaces per level)
    - max_index -- DEPRECATED The max number of indexes
    - optimizations -- The level of optimizations
    - replace_indices -- Set to `True` to replace matrix[i,j] with matrix_i_j (And similarly for vectors)
    - assignment_op -- The assignment operation
    """
    if max_index is not None:
        print("Warning: max_index parameter is deprecated in OutputMatrix_CollectingFactors")

    return _AuxiliaryOutputCollectionFactors(A, name, language, indentation_level, optimizations, replace_indices, assignment_op, OutputMatrix)


def OutputVector_CollectingFactors(A, name, language, indentation_level=0, max_index=None, optimizations='basic', replace_indices=True, assignment_op="="):
    """
    This method collects the constants of the replacement for vectors.

    Keyword arguments:
    - A -- The  factors
    - name -- The name of the constant
    - language -- The language of replacement
    - indentation_level -- The depth of the indentation (4 spaces per level)
    - max_index -- DEPRECATED The max number of indexes
    - optimizations -- The level of optimizations
    - replace_indices -- Set to `True` to replace matrix[i,j] with matrix_i_j (And similarly for vectors)
    - assignment_op -- The assignment operation
    """
    if max_index is not None:
        print("Warning: max_index parameter is deprecated in OutputVector_CollectingFactors")

    return _AuxiliaryOutputCollectionFactors(A, name, language, indentation_level, optimizations, replace_indices, assignment_op, OutputVector)


def OutputScalar_CollectingFactors(A, name, language, indentation_level=0, optimizations='basic', replace_indices=True, assignment_op="="):
    """
    This method collects the constants of the replacement for vectors.

    Keyword arguments:
    - A -- The  factors
    - name -- The name of the constant
    - language -- The language of replacement
    - indentation_level -- The depth of the indentation (4 spaces per level)
    - optimizations -- The level of optimizations
    - replace_indices -- Set to `True` to replace matrix[i,j] with matrix_i_j (And similarly for vectors)
    - assignment_op -- The assignment operation
    """
    return _AuxiliaryOutputCollectionFactors(A, name, language, indentation_level, optimizations, replace_indices, assignment_op, OutputScalar)