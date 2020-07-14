import numpy as np
import math


def do_operations(results):
    # print(results)
    if len(results) == 1:
        # print("final result: ", results[0])
        return (results[0], "")

    for result in results:
        if result in ['l', 'g', 'n']:
            # print("Invalid Syntax: Can't have more than one comparision")
            return (float('nan'), "Invalid Syntax: Can't have more than one comparision")

    for i, result in enumerate(results):
        if result == 't':
            try:
                results[i+1] = results[i-1]*results[i+1]
                results.remove(results[i])
                results.remove(results[i-1])
                return do_operations(results)
            except:
                # print("Syntax Error")
                return (float('nan'), "Syntax Error")
        if result == '+':
            try:
                results[i+1] = results[i-1]+results[i+1]
                results.remove(results[i])
                results.remove(results[i-1])
                return do_operations(results)
            except:
                # print("Syntax Error")
                return (float('nan'), "Syntax Error")
        if result == '-':
            try:
                results[i+1] = results[i-1]-results[i+1]
                results.remove(results[i])
                results.remove(results[i-1])
                return do_operations(results)
            except:
                # print("Syntax Error")
                return (float('nan'), "Syntax Error")
