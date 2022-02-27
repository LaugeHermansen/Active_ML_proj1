from datetime import datetime

def tqdm(iterable, layer = 0, freq = 0.1, width = 50, n_child_layers = 5):
    """
    Parameters
    ---------
    iterable:       iterable - if an int is provided it uses range(int) as iterable
    layer:          the line nr on which the progress bar will print - 0-indexed
    freq:           the frequency with which the progress bar updates
    width:          width of progress bar
    n_child_layers: how many nested loops inside this one - only works in layer = 0

    Return
    -------
    generator object that returns the elements in iterable, while printing the progress bar
    """
    if type(iterable) == int: iterable = range(iterable)
    n = len(iterable)
    lines_temp = ["|"]*width
    spaces_temp = [" "]*width

    start_time = datetime.now()

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
            print("\n"*layer,"\r","".join(output), " - est. stop: ", estimated_stop_time,"\033[F"*layer, sep = '',end = '')

        yield element
    
    if layer == 0: print("\n"*n_child_layers)

if __name__ == "__main__":
    # import time
    # for i in tqdm(range(8)):
    #     for j in tqdm(range(90),1):
    #         time.sleep(0.03)


    a = [(i,j) for j in tqdm(10000) for i in tqdm(10,1)]