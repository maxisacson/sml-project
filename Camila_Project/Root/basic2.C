/// \file
/// \ingroup tutorial_tree
///
/// Create can ntuple reading data from an ascii file.
/// This macro is a variant of basic.C
/// \macro_image
/// \macro_code
/// \author Rene Brun

void basic2() {
   TString dir = gSystem->UnixPathName(__FILE__);
   dir.ReplaceAll("basic2.C","");
   dir.ReplaceAll("/./","/");

   TFile *f = new TFile("output_KNeighbours_400GeV.root","RECREATE");
   TTree *T = new TTree("ntuple","data from ascii file");
   Long64_t nlines = T->ReadFile(Form("%soutput_KNeighbours_400GeV.csv",dir.Data()),"#:eventNumber:pt_mc_h:eta_mc_h_:phi_mc_h_:e_mc_h_:pt_mc_tau_:eta_mc_tau_:phi_mc_tau_:e_mc_tau_:pt_mc_nuH:eta_mc_nuH_:phi_mc_nuH_:e_mc_nuH_:pt_mc_nuTau:eta_mc_nuTau_:phi_mc_nuTau_:e_mc_nuTau_:et_met:phi_met:ntau:nbjet:njet:pt_reco_tau1:eta_reco_tau1_:phi_reco_tau1:m_reco_tau1_:pt_reco_bjet1_:eta_reco_bjet1_:phi_reco_bjet1_:m_reco_bjet1_:pt_reco_bjet2_:eta_reco_bjet2_:phi_reco_bjet2_:m_reco_bjet2_:pt_reco_bjet3_:eta_reco_bjet3_:phi_reco_bjet3_:m_reco_bjet3_:pt_reco_bjet4_:eta_reco_bjet4_:phi_reco_bjet4_:m_reco_bjet4_:pt_reco_jet1_:eta_reco_jet1_:phi_reco_jet1_:m_reco_jet1_:pt_reco_jet2_:eta_reco_jet2_:phi_reco_jet2_:m_reco_jet2_:pt_reco_jet3_:eta_reco_jet3_:phi_reco_jet3_:m_reco_jet3_:pt_reco_jet4_:eta_reco_jet4_:phi_reco_jet4_:m_reco_jet4_:unnamed:truth_met:predicted_pTnu:resolution_pred_pT:resolution_default_pT");
   printf(" found %lld points\n",nlines);
   T->Write();
}
