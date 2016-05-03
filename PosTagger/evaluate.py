import numpy as np

def confusion(true_assignments, predicted_assignments, xs, class_labels, unknown, ignored_classes):
    N = len(class_labels)
    contingency_all = np.zeros((N, N))
    contingency_known = np.zeros((N, N))

    ys, y_hats, xs = sum(true_assignments, []), sum(predicted_assignments, []), sum(xs, [])
    for y, yhat, x in zip(ys, y_hats, xs):
        if not(y in ignored_classes or yhat in ignored_classes):
            if x != unknown:
                contingency_known[y, yhat] += 1.
            contingency_all[y, yhat] += 1.

    accuracy_known = np.trace(contingency_known) / np.sum(contingency_known)
    accuracy_all = np.trace(contingency_all) / np.sum(contingency_all)

    return contingency_all, accuracy_all, contingency_known, accuracy_known

def latex_table(mat, labels):
    N = len(labels)
    print"\\begin{tabular}{c|",
    for i in range(N): print "c",
    print"}"
    print"&",

    labellist = []
    for i in range(1, N-1):
        labellist.append(labels[i])

    rot_labellist = ["\\rot{"+label+"}" for label in labellist]
    print "&".join(rot_labellist),"\\\\"
    print "\\hline\\\\"
    for i, label in enumerate(labellist):
        print label,"&",
        row = mat[i+1, 1:N-1]
        l = row.tolist()
        strings = [str(int(el)) for el in l]
        elements = "&".join(strings)
        print elements,"\\\\"
    print"\\end{tabular}"