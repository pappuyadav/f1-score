import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

recall_path=r'path of your recall text file'
precision_path=r'path of your precision text file'


def pr_plot(recall_path,precision_path):
    data = pd.read_csv(recall_path, sep=',', header=None)
    data1= pd.read_csv(precision_path, sep=',', header=None)

    for i, col in enumerate(data.columns):
        data.iloc[:, i] = data.iloc[:, i].str.replace("'", '')
        recall=data.loc[0]
    recall=pd.to_numeric(recall)


    for j, col1 in enumerate(data1.columns):
        data1.iloc[:, j] = data1.iloc[:, j].str.replace("'", '')
        precision=data1.loc[0]
    precision=pd.to_numeric(precision)
    
    def interpolate_gaps(values, limit=None):
        values = np.asarray(values)
        i = np.arange(values.size)
        valid = np.isfinite(values)
        filled = np.interp(i, i[valid], values[valid])

        if limit is not None:
            invalid = ~valid
            for n in range(1, limit+1):
                invalid[:-n] &= invalid[n:]
            filled[invalid] = np.nan

        return filled
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f1=[]
        for r1 in range(len(recall)):
            r11=recall[r1]
            p11=precision[r1]
            f11=2*(r11*p11)/(r11+p11)
            f1.append(f11)
        f1_filled = interpolate_gaps(f1, limit=6)
        f1_max=np.max(f1_filled)
        max_index=np.argmax(f1_filled)
        print(f1_max,max_index)
        
        
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.20, top=0.9, wspace=0.35, hspace=0.60)
    plt.suptitle("PR Plot for Volunteer Cotton")

    plt.subplot(2,2,1)
    plt.plot(recall, precision,label='Logistic',color='r')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig('precision-recall.png',dpi=1200)

    
    plt.subplot(2,2,2)
    plt.plot(recall,label='Logistic',color='b')
    plt.xlabel('Steps')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.savefig('recall-steps.png',dpi=1200)

    
    plt.subplot(2,2,3)
    plt.plot(precision,label='Logistic',color='g')
    plt.xlabel('Steps')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig('precision-steps.png',dpi=1200)
    
    
    plt.subplot(2,2,4)
    plt.plot(f1_filled,label='Logistic',color='r')
    plt.xlabel('Steps')
    plt.ylabel('f1_score')
    text= "f1-max={:.3f}".format(f1_max)
    plt.annotate(text, xy=(max_index, f1_max), xytext=(0.92,f1_max+0.08))
    plt.grid(True)
    plt.savefig('f1_score-steps.png',dpi=1200)
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
pr_plot(recall_path,precision_path)

    
