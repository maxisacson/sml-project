#!/usr/bin/env python3

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import scipy as sp
from scipy.stats import pearsonr

pd.set_option('display.max_columns', 500)


def mkdir(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        pass


def getData(datafile):
    ds = pd.read_csv(datafile)
    return ds


def main(argv):
    datafiles = [
                 "../test/mg5pythia8_hp200.root.test3.csv",
                 "../test/mg5pythia8_hp300.root.test.csv",
                 "../test/mg5pythia8_hp400.root.test.csv",
                ]

    frames = []
    for df in datafiles:
        frames.append(getData(df))
    data = pd.concat(frames)

    predictors = sp.array([
        "et(met)", "phi(met)",
        #"nbjet", "njet",
        "pt(reco tau1)", "eta(reco tau1)", "phi(reco tau1)", "m(reco tau1)",
        "pt(reco bjet1)", "eta(reco bjet1)", "phi(reco bjet1)", "m(reco bjet1)",
        "pt(reco jet1)", "eta(reco jet1)", "phi(reco jet1)", "m(reco jet1)",
        "pt(reco jet2)", "eta(reco jet2)", "phi(reco jet2)", "m(reco jet2)",
    ], dtype=str)

    target = "pt(mc nuH)"

    plotdir = "correlations"
    mkdir(plotdir)

    nplots = len(predictors)
    w = sp.ceil(sp.sqrt(nplots))
    h = sp.floor(sp.sqrt(nplots))
    
    nplots_total = nplots*len(frames)
    iplot = 0
    
    for j, frame in enumerate(frames):
        x = sp.array(frame[target], dtype=float)
        xmin = x.min()
        xmax = x.max()
        xedges = sp.linspace(xmin, xmax, 100)

        masspoint = datafiles[j].split('_')[1].split('.')[0].replace('hp', '')

        for i, pred in enumerate(predictors):
            y = sp.array(frame[pred], dtype=float)
            ymin = y.min()
            ymax = y.max()
            yedges = sp.linspace(ymin, ymax, 100)

            r, p = pearsonr(x, y)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            Z, xedges, yedges = sp.histogram2d(x, y, bins=100)
            X, Y = sp.meshgrid(xedges, yedges, indexing='ij')
            ax.pcolormesh(X, Y, Z, norm=LogNorm(vmin=1e-4, vmax=Z.max()))
            ax.set_xlabel(target)
            ax.set_ylabel(pred)
            ax.set_title("H+ {} GeV, rho = {:.2f}%, p = {:.2f}%".format(masspoint, 100*r, 100*p))
            ax.autoscale_view(tight=True)
            fig.tight_layout()
            plotname = "hp{}_{}".format(masspoint, pred.replace('(', '_').replace(')', ''))
            plt.savefig("{}/{}.png".format(plotdir, plotname))
            plt.savefig("{}/{}.pdf".format(plotdir, plotname))
            plt.close(fig)

            iplot += 1
            print("--- {} / {}".format(iplot, nplots_total))
   

if __name__ == "__main__":
    sys.exit(main(sys.argv))
