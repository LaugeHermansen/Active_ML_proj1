def tqdm(iterable, layer = 0, freq = 0.1, width = 50, n_child_layers = 5):
    """
    inputs:
    iterable:       you know what it is
    layer:          the line nr on which the progress bar will print - 0-indexed
    freq:           the frequency with which the progress bar updates
    width:          width of progress bar
    n_child_layers: how many nested loops inside this one - only works in layer = 0

    output: generator object that returns the elements in iterable, while printing the progress bar
    """
    n = len(iterable)
    lines_temp = ["|"]*width
    spaces_temp = [" "]*width
    for i, element in enumerate(iterable):
        if int(n*freq) == 0 or (i + 1) % int(n*freq) == 0 or i == n-1 or i == 0:
            p = (i+1)/n
            p_string = f'{p:.0%}'
            lines = lines_temp[:int(p*width)]
            spaces = spaces_temp[:(width - int(p*width))]
            output = ["["] + lines + spaces + ["]  "] + [" "*(4-len(p_string))] + [p_string]
            print("\n"*layer,"\r","".join(output),"\033[F"*layer, sep = '',end = '')
        yield element
    
    if layer == 0: print("\n"*n_child_layers)

if __name__ == "__main__":
    import time
    for i in tqdm(range(8)):
        for j in tqdm(range(90),1):
            time.sleep(0.001)
