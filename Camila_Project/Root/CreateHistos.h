//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Wed Apr 20 09:33:53 2016 by ROOT version 6.07/03
// from TTree ntuple/data from ascii file
// found on file: output_BayesRidge_200GeV.root
//////////////////////////////////////////////////////////

#ifndef CreateHistos_h
#define CreateHistos_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

class CreateHistos {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Float_t         #;
   Float_t         eventNumber;
   Float_t         pt(mc h+);
   Float_t         eta(mc h+);
   Float_t         phi(mc h+);
   Float_t         e(mc h+);
   Float_t         pt(mc tau);
   Float_t         eta(mc tau);
   Float_t         phi(mc tau);
   Float_t         e(mc tau);
   Float_t         pt(mc nuH);
   Float_t         eta(mc nuH);
   Float_t         phi(mc nuH);
   Float_t         e(mc nuH);
   Float_t         pt(mc nuTau);
   Float_t         eta(mc nuTau);
   Float_t         phi(mc nuTau);
   Float_t         e(mc nuTau);
   Float_t         et(met);
   Float_t         phi(met);
   Float_t         ntau;
   Float_t         nbjet;
   Float_t         njet;
   Float_t         pt(reco tau1);
   Float_t         eta(reco tau1);
   Float_t         phi(reco tau1);
   Float_t         m(reco tau1);
   Float_t         pt(reco bjet1);
   Float_t         eta(reco bjet1);
   Float_t         phi(reco bjet1);
   Float_t         m(reco bjet1);
   Float_t         pt(reco bjet2);
   Float_t         eta(reco bjet2);
   Float_t         phi(reco bjet2);
   Float_t         m(reco bjet2);
   Float_t         pt(reco bjet3);
   Float_t         eta(reco bjet3);
   Float_t         phi(reco bjet3);
   Float_t         m(reco bjet3);
   Float_t         pt(reco bjet4);
   Float_t         eta(reco bjet4);
   Float_t         phi(reco bjet4);
   Float_t         m(reco bjet4);
   Float_t         pt(reco jet1);
   Float_t         eta(reco jet1);
   Float_t         phi(reco jet1);
   Float_t         m(reco jet1);
   Float_t         pt(reco jet2);
   Float_t         eta(reco jet2);
   Float_t         phi(reco jet2);
   Float_t         m(reco jet2);
   Float_t         pt(reco jet3);
   Float_t         eta(reco jet3);
   Float_t         phi(reco jet3);
   Float_t         m(reco jet3);
   Float_t         pt(reco jet4);
   Float_t         eta(reco jet4);
   Float_t         phi(reco jet4);
   Float_t         m(reco jet4);
   Float_t         unnamed;
   Float_t         truth_met;
   Float_t         predicted_pTnu;
   Float_t         resolution_pred_pT;
   Float_t         resolution_default_pT;

   // List of branches
   TBranch        *b_#;   //!
   TBranch        *b_eventNumber;   //!
   TBranch        *b_pt(mc h+);   //!
   TBranch        *b_eta(mc h+);   //!
   TBranch        *b_phi(mc h+);   //!
   TBranch        *b_e(mc h+);   //!
   TBranch        *b_pt(mc tau);   //!
   TBranch        *b_eta(mc tau);   //!
   TBranch        *b_phi(mc tau);   //!
   TBranch        *b_e(mc tau);   //!
   TBranch        *b_pt(mc nuH);   //!
   TBranch        *b_eta(mc nuH);   //!
   TBranch        *b_phi(mc nuH);   //!
   TBranch        *b_e(mc nuH);   //!
   TBranch        *b_pt(mc nuTau);   //!
   TBranch        *b_eta(mc nuTau);   //!
   TBranch        *b_phi(mc nuTau);   //!
   TBranch        *b_e(mc nuTau);   //!
   TBranch        *b_et(met);   //!
   TBranch        *b_phi(met);   //!
   TBranch        *b_ntau;   //!
   TBranch        *b_nbjet;   //!
   TBranch        *b_njet;   //!
   TBranch        *b_pt(reco tau1);   //!
   TBranch        *b_eta(reco tau1);   //!
   TBranch        *b_phi(reco tau1);   //!
   TBranch        *b_m(reco tau1);   //!
   TBranch        *b_pt(reco bjet1);   //!
   TBranch        *b_eta(reco bjet1);   //!
   TBranch        *b_phi(reco bjet1);   //!
   TBranch        *b_m(reco bjet1);   //!
   TBranch        *b_pt(reco bjet2);   //!
   TBranch        *b_eta(reco bjet2);   //!
   TBranch        *b_phi(reco bjet2);   //!
   TBranch        *b_m(reco bjet2);   //!
   TBranch        *b_pt(reco bjet3);   //!
   TBranch        *b_eta(reco bjet3);   //!
   TBranch        *b_phi(reco bjet3);   //!
   TBranch        *b_m(reco bjet3);   //!
   TBranch        *b_pt(reco bjet4);   //!
   TBranch        *b_eta(reco bjet4);   //!
   TBranch        *b_phi(reco bjet4);   //!
   TBranch        *b_m(reco bjet4);   //!
   TBranch        *b_pt(reco jet1);   //!
   TBranch        *b_eta(reco jet1);   //!
   TBranch        *b_phi(reco jet1);   //!
   TBranch        *b_m(reco jet1);   //!
   TBranch        *b_pt(reco jet2);   //!
   TBranch        *b_eta(reco jet2);   //!
   TBranch        *b_phi(reco jet2);   //!
   TBranch        *b_m(reco jet2);   //!
   TBranch        *b_pt(reco jet3);   //!
   TBranch        *b_eta(reco jet3);   //!
   TBranch        *b_phi(reco jet3);   //!
   TBranch        *b_m(reco jet3);   //!
   TBranch        *b_pt(reco jet4);   //!
   TBranch        *b_eta(reco jet4);   //!
   TBranch        *b_phi(reco jet4);   //!
   TBranch        *b_m(reco jet4);   //!
   TBranch        *b_unnamed;   //!
   TBranch        *b_truth_met;   //!
   TBranch        *b_predicted_pTnu;   //!
   TBranch        *b_resolution_pred_pT;   //!
   TBranch        *b_resolution_default_pT;   //!

   CreateHistos(TTree *tree=0);
   virtual ~CreateHistos();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef CreateHistos_cxx
CreateHistos::CreateHistos(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("output_BayesRidge_200GeV.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("output_BayesRidge_200GeV.root");
      }
      f->GetObject("ntuple",tree);

   }
   Init(tree);
}

CreateHistos::~CreateHistos()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t CreateHistos::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t CreateHistos::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void CreateHistos::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("#", &#, &b_#);
   fChain->SetBranchAddress("eventNumber", &eventNumber, &b_eventNumber);
   fChain->SetBranchAddress("pt(mc h+)", &pt(mc h+), &b_pt(mc h+));
   fChain->SetBranchAddress("eta(mc h+)", &eta(mc h+), &b_eta(mc h+));
   fChain->SetBranchAddress("phi(mc h+)", &phi(mc h+), &b_phi(mc h+));
   fChain->SetBranchAddress("e(mc h+)", &e(mc h+), &b_e(mc h+));
   fChain->SetBranchAddress("pt(mc tau)", &pt(mc tau), &b_pt(mc tau));
   fChain->SetBranchAddress("eta(mc tau)", &eta(mc tau), &b_eta(mc tau));
   fChain->SetBranchAddress("phi(mc tau)", &phi(mc tau), &b_phi(mc tau));
   fChain->SetBranchAddress("e(mc tau)", &e(mc tau), &b_e(mc tau));
   fChain->SetBranchAddress("pt(mc nuH)", &pt(mc nuH), &b_pt(mc nuH));
   fChain->SetBranchAddress("eta(mc nuH)", &eta(mc nuH), &b_eta(mc nuH));
   fChain->SetBranchAddress("phi(mc nuH)", &phi(mc nuH), &b_phi(mc nuH));
   fChain->SetBranchAddress("e(mc nuH)", &e(mc nuH), &b_e(mc nuH));
   fChain->SetBranchAddress("pt(mc nuTau)", &pt(mc nuTau), &b_pt(mc nuTau));
   fChain->SetBranchAddress("eta(mc nuTau)", &eta(mc nuTau), &b_eta(mc nuTau));
   fChain->SetBranchAddress("phi(mc nuTau)", &phi(mc nuTau), &b_phi(mc nuTau));
   fChain->SetBranchAddress("e(mc nuTau)", &e(mc nuTau), &b_e(mc nuTau));
   fChain->SetBranchAddress("et(met)", &et(met), &b_et(met));
   fChain->SetBranchAddress("phi(met)", &phi(met), &b_phi(met));
   fChain->SetBranchAddress("ntau", &ntau, &b_ntau);
   fChain->SetBranchAddress("nbjet", &nbjet, &b_nbjet);
   fChain->SetBranchAddress("njet", &njet, &b_njet);
   fChain->SetBranchAddress("pt(reco tau1)", &pt(reco tau1), &b_pt(reco tau1));
   fChain->SetBranchAddress("eta(reco tau1)", &eta(reco tau1), &b_eta(reco tau1));
   fChain->SetBranchAddress("phi(reco tau1)", &phi(reco tau1), &b_phi(reco tau1));
   fChain->SetBranchAddress("m(reco tau1)", &m(reco tau1), &b_m(reco tau1));
   fChain->SetBranchAddress("pt(reco bjet1)", &pt(reco bjet1), &b_pt(reco bjet1));
   fChain->SetBranchAddress("eta(reco bjet1)", &eta(reco bjet1), &b_eta(reco bjet1));
   fChain->SetBranchAddress("phi(reco bjet1)", &phi(reco bjet1), &b_phi(reco bjet1));
   fChain->SetBranchAddress("m(reco bjet1)", &m(reco bjet1), &b_m(reco bjet1));
   fChain->SetBranchAddress("pt(reco bjet2)", &pt(reco bjet2), &b_pt(reco bjet2));
   fChain->SetBranchAddress("eta(reco bjet2)", &eta(reco bjet2), &b_eta(reco bjet2));
   fChain->SetBranchAddress("phi(reco bjet2)", &phi(reco bjet2), &b_phi(reco bjet2));
   fChain->SetBranchAddress("m(reco bjet2)", &m(reco bjet2), &b_m(reco bjet2));
   fChain->SetBranchAddress("pt(reco bjet3)", &pt(reco bjet3), &b_pt(reco bjet3));
   fChain->SetBranchAddress("eta(reco bjet3)", &eta(reco bjet3), &b_eta(reco bjet3));
   fChain->SetBranchAddress("phi(reco bjet3)", &phi(reco bjet3), &b_phi(reco bjet3));
   fChain->SetBranchAddress("m(reco bjet3)", &m(reco bjet3), &b_m(reco bjet3));
   fChain->SetBranchAddress("pt(reco bjet4)", &pt(reco bjet4), &b_pt(reco bjet4));
   fChain->SetBranchAddress("eta(reco bjet4)", &eta(reco bjet4), &b_eta(reco bjet4));
   fChain->SetBranchAddress("phi(reco bjet4)", &phi(reco bjet4), &b_phi(reco bjet4));
   fChain->SetBranchAddress("m(reco bjet4)", &m(reco bjet4), &b_m(reco bjet4));
   fChain->SetBranchAddress("pt(reco jet1)", &pt(reco jet1), &b_pt(reco jet1));
   fChain->SetBranchAddress("eta(reco jet1)", &eta(reco jet1), &b_eta(reco jet1));
   fChain->SetBranchAddress("phi(reco jet1)", &phi(reco jet1), &b_phi(reco jet1));
   fChain->SetBranchAddress("m(reco jet1)", &m(reco jet1), &b_m(reco jet1));
   fChain->SetBranchAddress("pt(reco jet2)", &pt(reco jet2), &b_pt(reco jet2));
   fChain->SetBranchAddress("eta(reco jet2)", &eta(reco jet2), &b_eta(reco jet2));
   fChain->SetBranchAddress("phi(reco jet2)", &phi(reco jet2), &b_phi(reco jet2));
   fChain->SetBranchAddress("m(reco jet2)", &m(reco jet2), &b_m(reco jet2));
   fChain->SetBranchAddress("pt(reco jet3)", &pt(reco jet3), &b_pt(reco jet3));
   fChain->SetBranchAddress("eta(reco jet3)", &eta(reco jet3), &b_eta(reco jet3));
   fChain->SetBranchAddress("phi(reco jet3)", &phi(reco jet3), &b_phi(reco jet3));
   fChain->SetBranchAddress("m(reco jet3)", &m(reco jet3), &b_m(reco jet3));
   fChain->SetBranchAddress("pt(reco jet4)", &pt(reco jet4), &b_pt(reco jet4));
   fChain->SetBranchAddress("eta(reco jet4)", &eta(reco jet4), &b_eta(reco jet4));
   fChain->SetBranchAddress("phi(reco jet4)", &phi(reco jet4), &b_phi(reco jet4));
   fChain->SetBranchAddress("m(reco jet4)", &m(reco jet4), &b_m(reco jet4));
   fChain->SetBranchAddress("unnamed", &unnamed, &b_unnamed);
   fChain->SetBranchAddress("truth_met", &truth_met, &b_truth_met);
   fChain->SetBranchAddress("predicted_pTnu", &predicted_pTnu, &b_predicted_pTnu);
   fChain->SetBranchAddress("resolution_pred_pT", &resolution_pred_pT, &b_resolution_pred_pT);
   fChain->SetBranchAddress("resolution_default_pT", &resolution_default_pT, &b_resolution_default_pT);
   Notify();
}

Bool_t CreateHistos::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void CreateHistos::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t CreateHistos::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef CreateHistos_cxx
