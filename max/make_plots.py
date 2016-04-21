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


def make_plot(files, ptype="pt", pdir="plots", postfix=""):
    f_indep = ROOT.TFile.Open(files["independents"], "READ")
    f_gp = ROOT.TFile(files["gp"], "READ")
    f_nn = ROOT.TFile(files["nn"], "READ")
    f_rr = ROOT.TFile(files["rr"], "READ")
    #f_rr2 = ROOT.TFile(files["rr2"], "READ")
    #f_rr3 = ROOT.TFile(files["rr3"], "READ")
    f_svr = ROOT.TFile(files["svr"], "READ")
    f_rf = ROOT.TFile(files["rf"], "READ")

    met_var = None
    truth_var = None
    gp_var = None
    nn_var = None
    rr_var = None
    rf_var = None
    xtitle = "Arbitraty units"
    ytitle = "Arbitraty units"

    if ptype == "pt":
        met_var = "met"
        truth_var = "pt_truth"
        gp_var = "pt_prediction"
        nn_var = "pt_prediction"
        rr_var = "pt_BayesRidgeReg"
        svr_var = "pt_svr_rbf"
        rf_var = "pt_random_forest"
        xtitle = "E_{T} [GeV]"
    if ptype == "mt":
        met_var = "mt_met"
        truth_var = "mt_truth"
        gp_var = "mt_prediction"
        nn_var = "mt_prediction"
        rr_var = "mt_BayesRidgeReg"
        svr_var = "mt_svr_rbf"
        rf_var = "mt_random_forest"
        xtitle = "M_{T} [GeV]"
    if ptype == "resolution":
        met_var = "res_met"
        gp_var = "res_prediction"
        nn_var = "res_prediction"
        rr_var = "res_BayesRidgeReg"
        svr_var = "res_svr_rbf"
        rf_var = "res_random_forest"
        xtitle = "Resolution [%]"
    if ptype == "profile":
        met_var = "profile_met"
        gp_var = "profile_prediction"
        nn_var = "profile_prediction"
        rr_var = "profile_pred_BayesRidgeReg"
        svr_var = "profile_pred_svr_rbf"
        rf_var = "profile_pred_random_forest"
        xtitle = "nu_{H} p_{T} [GeV]"
        ytitle = "Response"
   
    h_truth = None
    if truth_var:
        h_truth = f_indep.Get(truth_var)
    h_met = f_indep.Get(met_var)
    h_gp = f_gp.Get(gp_var)
    h_nn = f_nn.Get(nn_var)
    h_rr = f_rr.Get(rr_var)
    #h_rr2 = f_rr2.Get(rr_var)
    #h_rr3 = f_rr3.Get(rr_var)
    #h_rr.Add(h_rr2)
    #h_rr.Add(h_rr3)
    h_svr = f_svr.Get(svr_var)
    h_rf = f_rf.Get(rf_var)

    if ptype != "profile":
        map(normalize, [h_truth, h_met, h_gp, h_nn, h_rr, h_svr, h_rf])
  
    map(rebin, [h_truth, h_met, h_gp, h_nn, h_rr, h_svr, h_rf])

    if h_truth:
        h_truth.SetLineColor(ROOT.kBlack)
    h_met.SetLineColor(ROOT.kOrange+1)
    h_gp.SetLineColor(ROOT.kRed)
    h_nn.SetLineColor(ROOT.kMagenta+3)
    h_rr.SetLineColor(ROOT.kGreen+1)
    h_svr.SetLineColor(ROOT.kCyan+1)
    h_rf.SetLineColor(ROOT.kBlue)

    h_gp.SetMarkerStyle(ROOT.kFullCircle)
    h_gp.SetMarkerColor(ROOT.kRed)

    h_nn.SetMarkerStyle(ROOT.kFullTriangleDown)
    h_nn.SetMarkerColor(ROOT.kMagenta+3)

    h_rr.SetMarkerStyle(ROOT.kFullSquare)
    h_rr.SetMarkerColor(ROOT.kGreen+1)

    h_svr.SetMarkerStyle(ROOT.kFullTriangleUp)
    h_svr.SetMarkerColor(ROOT.kCyan+1)

    h_rf.SetMarkerStyle(ROOT.kFullDiamond)
    h_rf.SetMarkerColor(ROOT.kBlue)

    leg = ROOT.TLegend(.65, .65, .85, .85)
    leg.SetFillStyle(0)
    if h_truth:
        leg.AddEntry(h_truth, "Target", "l")
    leg.AddEntry(h_met, "MET", "l")
    leg.AddEntry(h_gp, "GP", "lep")
    leg.AddEntry(h_nn, "NN", "lep")
    leg.AddEntry(h_rr, "BRR", "lep")
    leg.AddEntry(h_svr, "SVR", "lep")
    leg.AddEntry(h_rf, "RF", "lep")

    lex = ROOT.TLatex()
    lex.SetNDC()

    hmax = get_hmax(h_truth, h_met, h_gp, h_nn, h_rr, h_svr, h_rf)
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
    h_rf.Draw("same")

    if ptype == "profile":
        line = ROOT.TLine(0, 1, 500, 1)
        line.SetLineWidth(2)
        line.SetLineColor(ROOT.kBlack)
        line.SetLineStyle(ROOT.kDashed)
        line.Draw("same")
    leg.Draw()
    if postfix == "_comb":
        lex.DrawLatex(0.65, 0.86, "All")
    else:
        mass = postfix.replace("_", "")
        lex.DrawLatex(0.65, 0.86, "H+ " + mass + " GeV")

    mkdir(pdir)
    c.Print(pdir + "/" + ptype + postfix + ".png")
    c.Print(pdir + "/" + ptype + postfix + ".pdf")


def main(argv):
    files_all = {
            "independents":"independents.root",
            "gp":"gp_regression.root",
            "nn":"../Mikael_project/25-10.sig-sig.min_max.pt(mc nuH).200000p.500e.all.root",
            "rr1":"/home/max/Dropbox/sml/project/sml-project/Camila_Project/Root/results/200GeV_Bayes.root",
            "rr2":"/home/max/Dropbox/sml/project/sml-project/Camila_Project/Root/results/300GeV_Bayes.root",
            "rr3":"/home/max/Dropbox/sml/project/sml-project/Camila_Project/Root/results/400GeV_Bayes.root",
            "svr":"/home/max/Dropbox/sml/project/sml-project/Henrik_Project/svr_rbf_C100.0_g0.09_e0.05_d3_i1000.root",
            }

    files_200 = {
            "independents":"independents_200.root",
            "gp":"gp_regression_200.root",
            "nn":"../Mikael_project/neural_net_separate_200GeV.root",
            "rr":"/home/max/Dropbox/sml/project/sml-project/Camila_Project/Root/results/200GeV_Bayes.root",
            "svr":"/home/max/Dropbox/sml/project/sml-project/Henrik_Project/svr_rbf_s200_C100.0_g0.09_e0.05_d3_i1000.root",
            "rf":"/home/max/Dropbox/sml/project/sml-project/Henrik_Project/random_forest_s200_d100_d15.root"
            }

    files_300 = {
            "independents":"independents_300.root",
            "gp":"gp_regression_300.root",
            "nn":"../Mikael_project/neural_net_separate_300GeV.root",
            "rr":"/home/max/Dropbox/sml/project/sml-project/Camila_Project/Root/results/300GeV_Bayes.root",
            "svr":"/home/max/Dropbox/sml/project/sml-project/Henrik_Project/svr_rbf_s300_C100.0_g0.09_e0.05_d3_i1000.root",
            "rf":"/home/max/Dropbox/sml/project/sml-project/Henrik_Project/random_forest_s300_d100_d15.root"
            }

    files_400 = {
            "independents":"independents_400.root",
            "gp":"gp_regression_400.root",
            "nn":"../Mikael_project/neural_net_separate_400GeV.root",
            "rr":"/home/max/Dropbox/sml/project/sml-project/Camila_Project/Root/results/400GeV_Bayes.root",
            "svr":"/home/max/Dropbox/sml/project/sml-project/Henrik_Project/svr_rbf_s400_C100.0_g0.09_e0.05_d3_i1000.root",
            "rf":"/home/max/Dropbox/sml/project/sml-project/Henrik_Project/random_forest_s400_d100_d15.root"
            }

    files = {"comb":files_all, "200":files_200, "300":files_300, "400":files_400}

    try:
        file_set = argv[1]
    except KeyError:
        file_set = "all"

    print("--- Using '{}'".format(file_set))
    plot_types = [
            "pt",
            "resolution",
            "mt",
            "profile",
            ]

    set_style()
    for ptype in plot_types:
        if file_set == "all":
            for fs in ["200", "300", "400"]:
                make_plot(files=files[fs], ptype=ptype, postfix="_"+fs)
        else:
            make_plot(files=files[file_set], ptype=ptype, postfix="_"+file_set)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
