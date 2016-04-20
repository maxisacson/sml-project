#!/usr/bin/env python

import os, sys
import ROOT

ROOT.gROOT.SetBatch(1)

def mkdir(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        pass


def set_style():
    ROOT.gROOT.LoadMacro("atlasstyle-00-03-05/AtlasStyle.C")
    ROOT.SetAtlasStyle()
    ROOT.gStyle.SetLegendBorderSize(0)


def get_hmax(*args):
    hmax = -float('inf')
    for arg in args:
        try:
            tmpmax = arg.GetMaximum()
            if tmpmax > hmax:
                hmax = tmpmax
        except AttributeError:
            pass
    return hmax


def roodelete(o):
    o.IsA().Destructor(o)


def normalize(h):
    try:
        h.Sumw2()
        h.Scale(1./h.Integral())
    except AttributeError:
        pass


def rebin(h, n=2):
    try:
        h.Rebin(2)
    except AttributeError:
        pass


#def rectify_resolution(h, bins=(50, -100, 100), reverse=False):
#    h_new = ROOT.TH1D(h.GetName() + "_fixed", "", bins[0], bins[1], bins[2])
#    h_new.Sumw2()
#    
#    binrange = range(1,h.GetNbinsX()+1)
#    if reverse:
#        binrange = [x for x in reversed(binrange)]
#
#    for ibin,nbin in enumerate(binrange):
#        c = h.GetXaxis().GetBinCenter(nbin)
#        count = h.GetXaxis().GetBinContent(nbin)


def make_plot(files, ptype="pt", pdir="plots"):
    f_indep = ROOT.TFile.Open(files["independents"], "READ")
    f_gp = ROOT.TFile(files["gp"], "READ")
    f_nn = ROOT.TFile(files["nn"], "READ")
    f_rr = ROOT.TFile(files["rr1"], "READ")
    f_rr2 = ROOT.TFile(files["rr2"], "READ")
    f_rr3 = ROOT.TFile(files["rr3"], "READ")
    f_svr = ROOT.TFile(files["svr"], "READ")

    met_var = None
    truth_var = None
    gp_var = None
    nn_var = None
    rr_var = None
    xtitle = "Arbitraty units"
    ytitle = "Arbitraty units"

    if ptype == "pt":
        met_var = "met"
        truth_var = "pt_truth"
        gp_var = "pt_prediction"
        nn_var = "pt_prediction"
        rr_var = "pt_BayesRidgeReg"
        svr_var = "pt_svr_rbf"
        xtitle = "E_{T} [GeV]"
    if ptype == "mt":
        met_var = "mt_met"
        truth_var = "mt_truth"
        gp_var = "mt_prediction"
        nn_var = "mt_prediction"
        rr_var = "mt_BayesRidgeReg"
        svr_var = "mt_svr_rbf"
        xtitle = "M_{T} [GeV]"
    if ptype == "resolution":
        met_var = "res_met"
        gp_var = "res_prediction"
        nn_var = "res_prediction"
        rr_var = "res_BayesRidgeReg"
        svr_var = "res_svr_rbf"
        xtitle = "Resolution [%]"
    if ptype == "profile":
        met_var = "profile_met"
        gp_var = "profile_prediction"
        nn_var = "profile_prediction"
        rr_var = "profile_pred_BayesRidgeReg"
        svr_var = "profile_pred_svr_rbf"
        xtitle = "nu_{H} p_{T} [GeV]"
        ytitle = "Response"
   
    h_truth = None
    if truth_var:
        h_truth = f_indep.Get(truth_var)
    h_met = f_indep.Get(met_var)
    h_gp = f_gp.Get(gp_var)
    h_nn = f_nn.Get(nn_var)
    h_rr = f_rr.Get(rr_var)
    h_rr2 = f_rr2.Get(rr_var)
    h_rr3 = f_rr3.Get(rr_var)
    h_rr.Add(h_rr2)
    h_rr.Add(h_rr3)
    h_svr = f_svr.Get(svr_var)

    if ptype != "profile":
        map(normalize, [h_truth, h_met, h_gp, h_nn, h_rr, h_svr])
  
    map(rebin, [h_truth, h_met, h_gp, h_nn, h_rr, h_svr])

    if h_truth:
        h_truth.SetLineColor(ROOT.kBlack)
    h_met.SetLineColor(ROOT.kOrange+1)
    h_gp.SetLineColor(ROOT.kRed)
    h_nn.SetLineColor(ROOT.kMagenta+3)
    h_rr.SetLineColor(ROOT.kGreen+1)
    h_svr.SetLineColor(ROOT.kBlue)

    h_gp.SetMarkerStyle(ROOT.kFullCircle)
    h_gp.SetMarkerColor(ROOT.kRed)

    h_nn.SetMarkerStyle(ROOT.kFullTriangleDown)
    h_nn.SetMarkerColor(ROOT.kMagenta+3)

    h_rr.SetMarkerStyle(ROOT.kFullSquare)
    h_rr.SetMarkerColor(ROOT.kGreen+1)

    h_svr.SetMarkerStyle(ROOT.kFullTriangleUp)
    h_svr.SetMarkerColor(ROOT.kBlue)

    leg = ROOT.TLegend(.65, .65, .85, .85)
    if h_truth:
        leg.AddEntry(h_truth, "Target", "l")
    leg.AddEntry(h_met, "MET", "l")
    leg.AddEntry(h_gp, "GP", "lep")
    leg.AddEntry(h_nn, "NN", "lep")
    leg.AddEntry(h_rr, "BRR", "lep")
    leg.AddEntry(h_svr, "SVR", "lep")

    hmax = get_hmax(h_truth, h_met, h_gp, h_nn, h_rr, h_svr)
    if h_truth:
        first = h_truth
    else:
        first = h_met

    if ptype == "profile":
        first.GetYaxis().SetRangeUser(0, 3)
    else:
        first.GetYaxis().SetRangeUser(0, hmax*1.4)
    first.GetYaxis().SetTitle(ytitle)
    first.GetXaxis().SetTitle(xtitle)

    c = ROOT.TCanvas("c", "c", 800, 600)
    if h_truth:
        h_truth.Draw("samehist")
    h_met.Draw("samehist")
    h_gp.Draw("same")
    h_nn.Draw("same")
    h_rr.Draw("same")
    h_svr.Draw("same")
    if ptype == "profile":
        line = ROOT.TLine(0, 1, 500, 1)
        line.SetLineWidth(2)
        line.SetLineColor(ROOT.kBlack)
        line.SetLineStyle(ROOT.kDashed)
        line.Draw("same")
    leg.Draw()

    mkdir(pdir)
    c.Print(pdir + "/" + ptype + ".png")
    c.Print(pdir + "/" + ptype + ".pdf")


def main(argv):
    files = {
            "independents":"independents.root",
            "gp":"gp_regression.root",
            "nn":"../Mikael_project/25-10.sig-sig.min_max.pt(mc nuH).200000p.500e.all.root",
            "rr1":"/home/max/Dropbox/sml/project/sml-project/Camila_Project/Root/results/200GeV_Bayes.root",
            "rr2":"/home/max/Dropbox/sml/project/sml-project/Camila_Project/Root/results/300GeV_Bayes.root",
            "rr3":"/home/max/Dropbox/sml/project/sml-project/Camila_Project/Root/results/400GeV_Bayes.root",
            "svr":"/home/max/Dropbox/sml/project/sml-project/Henrik_Project/svr_rbf_C100.0_g0.09_e0.05_d3_i1000.root",
            }

    plot_types = [
            "pt",
            "resolution",
            "mt",
            "profile",
            ]

    set_style()
    for ptype in plot_types:
        make_plot(files=files, ptype=ptype)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
