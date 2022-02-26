# Script to perform operations on the list of predictions

# Function to find precedence
# of operators.
def precedence(op):
    if op == '+' or op == '-':
        return 1
    if op == '*' or op == '/':
        return 2
    return 0


# Function to perform arithmetic
# operations.
def applyOp(a, b, op):
    if op == '+': return a + b
    if op == '-': return a - b
    if op == '*': return a * b
    if op == '/': return a // b


# Function that returns value of
# expression after evaluation.
def calculate(tokens):
    # stack to store integer values.
    values = []

    # stack to store operators.
    ops = []
    i = 0
    try:
        while i < len(tokens):

            if tokens[i] in ['<', '>', '!=']:
                return (float('nan'), "Invalid Syntax: Can't have more than one comparision")

            elif tokens[i].isdigit():
                val = 0

                while (i < len(tokens) and tokens[i].isdigit()):
                    val = (val * 10) + int(tokens[i])
                    i += 1

                values.append(val)

                i -= 1
            else:
                while (len(ops) != 0 and
                       precedence(ops[-1]) >= precedence(tokens[i])):
                    val2 = values.pop()
                    val1 = values.pop()
                    op = ops.pop()

                    values.append(applyOp(val1, val2, op))

                ops.append(tokens[i])

            i += 1

        while len(ops) != 0:
            val2 = values.pop()
            val1 = values.pop()
            op = ops.pop()

            values.append(applyOp(val1, val2, op))
    except Exception as e:
        print(e)
        return (float('nan'), "Syntax Error")

    return (values[-1], "")


def evaluate(tokens):
    if not (tokens[0].isdigit() and tokens[-1].isdigit()):
        return (float('nan'), "Invalid Syntax")

    # Check for relational operations
    is_relational = False
    for i, token in enumerate(tokens):
        if token in ['<', '>', '!=']:
            if not is_relational:
                is_relational = True
                result1, error1 = calculate(tokens[:i])
                result2, error2 = calculate(tokens[i + 1:])
                if result1 and result2 and error1 == "" and error2 == "":
                    if token == '<':
                        return ("True" if result1 < result2 else "False", error1 + error2)
                    elif token == '>':
                        return ("True" if result1 > result2 else "False", error1 + error2)
                    elif token == '!=':
                        return ("True" if result1 != result2 else "False", error1 + error2)
                else:
                    return (float('nan'), "Invalid Syntax")
            else:
                return (float('nan'), "Invalid Syntax: found more than one relational operator.")

    if not is_relational:
        result, error = calculate(tokens)
        return (result, error)
