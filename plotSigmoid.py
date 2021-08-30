import numpy as np
from matplotlib import pyplot as plt

def zFunc(t, A, logValue=None):
    if logValue is None:
        logValue = np.log(2 + np.sqrt(3))

    return 1 / (1 + np.exp(-A * logValue * t)), A * logValue 

def main():    
    dfms = [1,0.555,0.333,0.222]
    vals = []
    legend = []
    vd = 0
    for d in dfms:
        xV = []
        vals.append([])
        i = -15
        while (i <= 15):
            xV.append(i)
            vf, vd = zFunc(i,d)            
            vals[-1].append(vf)
            i += 0.1

        print(vd)
        legend.append("A = %f" % (vd))
        plt.plot(xV,vals[-1])
    
    histV = np.histogram(vf, bins=8, range=[0, 1])[1]

    for h in histV[1:-1]:
        plt.hlines(h,-15,15,colors='c')

    plt.legend(legend,loc="lower right")
    plt.xlabel("Depth Difference - DD")
    plt.ylabel("f(DD)")
    plt.title("Depth encoding functions")
    plt.show()

if __name__ == '__main__':
    main()