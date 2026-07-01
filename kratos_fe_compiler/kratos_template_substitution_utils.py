
def _FindAndRemoveIndentation(input_text, target_substring):
    end = input_text.find(target_substring)
    begin = end
    leading_spaces = 0
    while begin > 0:
        if input_text[begin] == '\n':
            break
        if input_text[begin] != ' ':
            end = begin - 1
        begin -= 1

    leading_spaces = end - begin

    if leading_spaces % 4 == 0:
        input_text = input_text[:begin+1] + input_text[end+1:]
        return int(leading_spaces / 4)

    raise ValueError(
        "Inconsistent indentation in source file! Substitution {} found {} leading spaces, which is not a multiple of four.".format(
        target_substring, leading_spaces
    ))
    
def _AddIndentationToSubstitution(substitution, n_indents):
    # This function adds indentation to the RHS and LHS code
    # to match the indentation of the target substring
    # It ignores any indentation included in the original code
    lines = substitution.split('\n')
    indented_lines = [' ' * (n_indents * 4) + line.strip() for line in lines]
    return '\n'.join(indented_lines)

def PerformSubstitutionOnTemplate( input_text, substitution_pattern, substitution_content_file_path):
    # Reading and filling the template file
    
    # Reading and filling the template file
    with open(substitution_content_file_path, "r") as substitution_content_file:
        substitution_content = substitution_content_file.read()

    indents = _FindAndRemoveIndentation(input_text, substitution_pattern)
    substitution_content = _AddIndentationToSubstitution(substitution_content, indents)
    return input_text.replace(substitution_pattern, substitution_content)