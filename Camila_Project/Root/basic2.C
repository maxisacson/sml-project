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
   Long64_t nlines = T->ReadFile(Form("%soutput_KNeighbours_400GeV.csv",dir.Data()),"#:eventNumber:pt(mc h+):eta(mc h+):phi(mc h+):e(mc h+):pt(mc tau):eta(mc tau):phi(mc tau):e(mc tau):pt(mc nuH):eta(mc nuH):phi(mc nuH):e(mc nuH):pt(mc nuTau):eta(mc nuTau):phi(mc nuTau):e(mc nuTau):et(met):phi(met):ntau:nbjet:njet:pt(reco tau1):eta(reco tau1):phi(reco tau1):m(reco tau1):pt(reco bjet1):eta(reco bjet1):phi(reco bjet1):m(reco bjet1):pt(reco bjet2):eta(reco bjet2):phi(reco bjet2):m(reco bjet2):pt(reco bjet3):eta(reco bjet3):phi(reco bjet3):m(reco bjet3):pt(reco bjet4):eta(reco bjet4):phi(reco bjet4):m(reco bjet4):pt(reco jet1):eta(reco jet1):phi(reco jet1):m(reco jet1):pt(reco jet2):eta(reco jet2):phi(reco jet2):m(reco jet2):pt(reco jet3):eta(reco jet3):phi(reco jet3):m(reco jet3):pt(reco jet4):eta(reco jet4):phi(reco jet4):m(reco jet4):unnamed:truth_met:predicted_pTnu:resolution_pred_pT:resolution_default_pT");
   printf(" found %lld points\n",nlines);
   T->Write();
}
