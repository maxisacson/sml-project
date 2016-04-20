#define CreateHistos_cxx
#include "CreateHistos.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include "TProfile.h"

void CreateHistos::Loop()
{
    TH1D* pt_neural_net = new TH1D("pt_BayesRidgeReg","",100,0,500);
    
    TH1* mt_neural_net = new TH1D("mt_BayesRidgeReg","",100,0,500);
    
    TH1* res_neural_net = new TH1D("res_BayesRidgeReg","",100,-100,100);
    
    //TH1* profile_pred;
    
    profile_pred = new TProfile("profile_pred_BayesRidgeReg","",100,0,500);
    
    
    
    if (fChain == 0) return;
    
    Long64_t nentries = fChain->GetEntriesFast();
    
    Long64_t nbytes = 0, nb = 0;
    for (Long64_t jentry=0; jentry<nentries;jentry++) {
        Long64_t ientry = LoadTree(jentry);
        if (ientry < 0) break;
        nb = fChain->GetEntry(jentry);   nbytes += nb;
        // if (Cut(ientry) < 0) continue;
        
        double ratio = predicted_pTnu/pt_mc_nuH;
        
        pt_neural_net->Fill(predicted_pTnu);
        res_neural_net->Fill(resolution_pred_pT);
        
        profile_pred->Fill(pt_mc_nuH,ratio);
        
        double mT= sqrt(2 * predicted_pTnu *pt_reco_tau1* (1 - TMath::Cos(phi_reco_tau1 - phi_met)));
        
        mt_neural_net->Fill(mT);
        
        
        
    }
    
    TFile* output = new TFile("results/400GeV_Bayes.root","RECREATE");
    pt_neural_net->Write();
    res_neural_net->Write();
    profile_pred->Write();
    mt_neural_net->Write();
    output->Close();
}
