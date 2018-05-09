import scipy.stats
import statsmodels.stats.multitest


def run(to_compare, names):

    '''
    :param to_compare: array-like 2 dimensions and with first dimension equal to 2
    :return:
    '''

    ps = []
    us = []

    for x in to_compare:
        u, p = scipy.stats.mannwhitneyu(x[0], x[1])
        ps.append(p)
        us.append(u)

    valid, p_corr, alpha_c_sidak, alpha_c_bonf = \
        statsmodels.stats.multitest.multipletests(pvals=ps, alpha=0.01, method="b")

    for p, u, p_c, v, name in zip(ps, us, p_corr, valid, names):
        print(
            f"[{name}] "
            f"Mann-Whitney rank test: u {u}, p {p:.3f}, p corr {p_c:.3f}, significant: {v}")
        print()

    # table = \
    #     r"\begin{table}[htbp]" + "\n" + \
    #     r"\begin{center}" + "\n" + \
    #     r"\begin{tabular}{llllllll}" + "\n" + \
    #     r"Measure & Variable & Constant & $u$ & $p$ (before corr.) " \
    #     r"& $p$ (after corr.) & Sign. at 1\% threshold \\" + "\n" + \
    #     r"\hline \\" + "\n"
    #
    # for p, u, p_c, v, dic in zip(ps, us, p_corr, valid, to_compare):
    #     p = "{:.3f}".format(p) if p >= 0.001 else "$<$ 0.001"
    #     p_c = "{:.3f}".format(p_c) if p_c >= 0.001 else "$<$ 0.001"
    #     v = "yes" if v else "no"
    #     table += r"{} & ${}$ & ${}$ & {} & {} & {} & {} \\" \
    #                  .format(dic["measure"], dic["var"], dic["constant"], u, p, p_c, v) \
    #              + "\n"
    #
    # table += \
    #     r"\end{tabular}" + "\n" + \
    #     r"\end{center}" + "\n" + \
    #     r"\caption{Significance tests for comparison using Mann-Withney's u. " \
    #     r"Bonferroni corrections are applied.}" + "\n" + \
    #     r"\label{table:significance_tests}" + "\n" + \
    #     r"\end{table}"
    #
    # print("*** Latex-formated table ***")
    # print(table)
