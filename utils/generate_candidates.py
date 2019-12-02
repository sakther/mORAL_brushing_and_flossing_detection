import numpy as np

def do_smooth_output(T, Y):
    Yn = [v for v in list(Y)]

    for i in range(3, len(Y)-3):
        val = np.mean([Y[j] for j in range(i-3, i+3)])
        if val>=0.5:
            Yn[i] = 1
        else:
            Yn[i]=0

    return Yn

def generate_brushing_candidates(AGMO):
    '''

    :param AGMO:
    :return: tuple of [start_index, end_index]
    '''

    T = [v[0] for v in AGMO]
    Ay = [v[2] for v in AGMO]

    D=np.array(Ay)

    D[D>0.1] = 1
    D[D<=0.1] = 0

    D = do_smooth_output(T, D)
    intervals = []
    i=0
    while i<len(D):
        if D[i] == 1:
            j = i
            while(j<len(D) and D[j] == 1):
                j+=1
            intervals.append([i, j-1])
            i = j
        else:
            i+=1

    return intervals


def generate_flossing_candidates(AGMO):
    '''

    :param AGMO:
    :return: tuple of [start_index, end_index]
    '''

    T = [v[0] for v in AGMO]
    Ay = [v[2] for v in AGMO]

    D=np.array(Ay)

    D[D>0.25] = 1
    D[D<=0.25] = 0

    D = do_smooth_output(T, D)
    intervals = []
    i=0
    while i<len(D):
        if D[i] == 1:
            j = i
            while(j<len(D) and D[j] == 1):
                j+=1
            intervals.append([i, j-1])
            i = j
        else:
            i+=1

    return intervals

