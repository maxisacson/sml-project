#!/usr/bin/env python

import os
import sys
import traceback
import ROOT
import pandas as pd
import scipy as sp


def getData(datafile):
    ds = pd.read_csv(datafile)
    return ds


def mt(et1, et2, phi1, phi2):
    return sp.sqrt(2*et1*et2*(1 - sp.cos(phi1 - phi2))) 


def create_root_file(**kwargs):
    met = kwargs["met"]
    met_phi = kwargs["met_phi"]
    target = kwargs["target"]
    target_phi = kwargs["target_phi"]
    tau_pt = kwargs["tau_pt"]
    tau_phi = kwargs["tau_phi"]

    response_met = met/target
    res_met = -1*(met - target)*100/target
    mt_met = mt(et1=met, et2=tau_pt, phi1=met_phi, phi2=tau_phi)
    mt_truth = mt(et1=target, et2=tau_pt, phi1=target_phi, phi2=tau_phi)

    f = ROOT.TFile("independents.root", "RECREATE")
    h_met = ROOT.TH1D("met", "", 100, 0, 500)
    h_pt_truth = ROOT.TH1D("pt_truth", "", 100, 0, 500)
    h_res_met = ROOT.TH1D("res_met", "", 100, -100, 100)
    h_mt_truth = ROOT.TH1D("mt_truth", "", 100, 0, 500)
    h_mt_met = ROOT.TH1D("mt_met", "", 100, 0, 500)
    p_profile_met = ROOT.TProfile("profile_met", "", 
            100, 0, 500, response_met.min(), response_met.max())

    map(h_met.Fill, met)
    map(h_pt_truth.Fill, target)
    map(h_res_met.Fill, res_met)
    map(h_mt_truth.Fill, mt_truth)
    map(h_mt_met.Fill, mt_met)
    map(p_profile_met.Fill, target, response_met)

    f.Write()
    f.Close()


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

    ds = data.sample(frac=1., random_state=123)

    t = sp.array(ds["pt(mc nuH)"], dtype=float)
    true_phi = sp.array(ds["phi(mc nuH)"], dtype=float)
    met = sp.array(ds["et(met)"], dtype=float)
    met_phi = sp.array(ds["phi(met)"], dtype=float)
    tau_pt = sp.array(ds["pt(reco tau1)"], dtype=float)
    tau_phi = sp.array(ds["phi(reco tau1)"], dtype=float)

    create_root_file(met=met, target=t, met_phi=met_phi, target_phi=true_phi,
        tau_pt=tau_pt, tau_phi=tau_phi)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
