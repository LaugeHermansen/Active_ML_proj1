from datetime import datetime


def tqdm(*iterable, layer = 0, freq = 0.1, width = 50, n_child_layers = 5):
    """
    Parameters
    ---------
    iterable:                    iterable - if an int or a tuple of ints is provided it uses range(*iterable) as iterable.
    layer (deprecated):          the line nr on which the progress bar will print - 0-indexed
                                 deprecated - I found a way to workaround it
    freq:                        the frequency with which the progress bar updates
    width:                       width of progress bar
    n_child_layers (deprecated): how many nested loops inside this one - only works in layer = 0

    Return
    -------
    generator object that returns the elements in iterable, while printing the progress bar
    """

    global global_layer
    try: global_layer == 0
    except: global_layer = 0

    global global_max_layer
    try: global_max_layer == 0
    except: global_max_layer = 0

    global_layer += 1
    global_max_layer = max(global_layer, global_max_layer)


    if type(iterable[0]) == int: iterable = range(*iterable)

    n = len(iterable)
    lines_temp = ["|"]*width
    spaces_temp = [" "]*width

    start_time = datetime.now()

    if global_max_layer != 1: print("\n",end = '')
    for i, element in enumerate(iterable):
        if int(n*freq) == 0 or (i + 1) % int(n*freq) == 0 or i == n-1 or i == 0:
            now = datetime.now()
            dif_time = now - start_time
            estimated_stop_time = "None" if i ==0 else (start_time + dif_time/i*n).time()

            p = (i+1)/n
            p_string = f'{p:.0%}'
            lines = lines_temp[:int(p*width)]
            spaces = spaces_temp[:(width - int(p*width))]
            output = ["["] + lines + spaces + ["]  "] + [" "*(4-len(p_string))] + [p_string]
            print("\r","".join(output), " - est. stop: ", estimated_stop_time, sep = '',end = '')

        yield element
    print("\033[F", end = '')
    
    global_layer -= 1

    if global_layer == 0:
        print("\n"*(global_max_layer), '\nDone', sep = '')
        global_max_layer = 0
    

if __name__ == "__main__":
    import time
    for i in tqdm(range(5)):
        for j in tqdm(range(13),1):
            time.sleep(0.01)
    
    # print('test')


    # a = [(i,j) for j in tqdm(10000) for i in tqdm(10,1)]
    # a = (1,5)
    # for i in range(*tuple(a)): print(i)