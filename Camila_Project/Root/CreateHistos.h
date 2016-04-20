//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Wed Apr 20 10:28:32 2016 by ROOT version 6.07/03
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
   const Int_t kMaxeta_mc_h = 1;
   const Int_t kMaxphi_mc_h = 1;
   const Int_t kMaxe_mc_h = 1;
   const Int_t kMaxpt_mc_tau = 1;
   const Int_t kMaxeta_mc_tau = 1;
   const Int_t kMaxphi_mc_tau = 1;
   const Int_t kMaxe_mc_tau = 1;
   const Int_t kMaxeta_mc_nuH = 1;
   const Int_t kMaxphi_mc_nuH = 1;
   const Int_t kMaxe_mc_nuH = 1;
   const Int_t kMaxeta_mc_nuTau = 1;
   const Int_t kMaxphi_mc_nuTau = 1;
   const Int_t kMaxe_mc_nuTau = 1;
   const Int_t kMaxeta_reco_tau1 = 1;
   const Int_t kMaxm_reco_tau1 = 1;
   const Int_t kMaxpt_reco_bjet1 = 1;
   const Int_t kMaxeta_reco_bjet1 = 1;
   const Int_t kMaxphi_reco_bjet1 = 1;
   const Int_t kMaxm_reco_bjet1 = 1;
   const Int_t kMaxpt_reco_bjet2 = 1;
   const Int_t kMaxeta_reco_bjet2 = 1;
   const Int_t kMaxphi_reco_bjet2 = 1;
   const Int_t kMaxm_reco_bjet2 = 1;
   const Int_t kMaxpt_reco_bjet3 = 1;
   const Int_t kMaxeta_reco_bjet3 = 1;
   const Int_t kMaxphi_reco_bjet3 = 1;
   const Int_t kMaxm_reco_bjet3 = 1;
   const Int_t kMaxpt_reco_bjet4 = 1;
   const Int_t kMaxeta_reco_bjet4 = 1;
   const Int_t kMaxphi_reco_bjet4 = 1;
   const Int_t kMaxm_reco_bjet4 = 1;
   const Int_t kMaxpt_reco_jet1 = 1;
   const Int_t kMaxeta_reco_jet1 = 1;
   const Int_t kMaxphi_reco_jet1 = 1;
   const Int_t kMaxm_reco_jet1 = 1;
   const Int_t kMaxpt_reco_jet2 = 1;
   const Int_t kMaxeta_reco_jet2 = 1;
   const Int_t kMaxphi_reco_jet2 = 1;
   const Int_t kMaxm_reco_jet2 = 1;
   const Int_t kMaxpt_reco_jet3 = 1;
   const Int_t kMaxeta_reco_jet3 = 1;
   const Int_t kMaxphi_reco_jet3 = 1;
   const Int_t kMaxm_reco_jet3 = 1;
   const Int_t kMaxpt_reco_jet4 = 1;
   const Int_t kMaxeta_reco_jet4 = 1;
   const Int_t kMaxphi_reco_jet4 = 1;
   const Int_t kMaxm_reco_jet4 = 1;

   // Declaration of leaf types
   Float_t         number;
   Float_t         eventNumber;
   Float_t         pt_mc_h;
   Float_t         eta_mc_h_;
   Float_t         phi_mc_h_;
   Float_t         e_mc_h_;
   Float_t         pt_mc_tau_;
   Float_t         eta_mc_tau_;
   Float_t         phi_mc_tau_;
   Float_t         e_mc_tau_;
   Float_t         pt_mc_nuH;
   Float_t         eta_mc_nuH_;
   Float_t         phi_mc_nuH_;
   Float_t         e_mc_nuH_;
   Float_t         pt_mc_nuTau;
   Float_t         eta_mc_nuTau_;
   Float_t         phi_mc_nuTau_;
   Float_t         e_mc_nuTau_;
   Float_t         et_met;
   Float_t         phi_met;
   Float_t         ntau;
   Float_t         nbjet;
   Float_t         njet;
   Float_t         pt_reco_tau1;
   Float_t         eta_reco_tau1_;
   Float_t         phi_reco_tau1;
   Float_t         m_reco_tau1_;
   Float_t         pt_reco_bjet1_;
   Float_t         eta_reco_bjet1_;
   Float_t         phi_reco_bjet1_;
   Float_t         m_reco_bjet1_;
   Float_t         pt_reco_bjet2_;
   Float_t         eta_reco_bjet2_;
   Float_t         phi_reco_bjet2_;
   Float_t         m_reco_bjet2_;
   Float_t         pt_reco_bjet3_;
   Float_t         eta_reco_bjet3_;
   Float_t         phi_reco_bjet3_;
   Float_t         m_reco_bjet3_;
   Float_t         pt_reco_bjet4_;
   Float_t         eta_reco_bjet4_;
   Float_t         phi_reco_bjet4_;
   Float_t         m_reco_bjet4_;
   Float_t         pt_reco_jet1_;
   Float_t         eta_reco_jet1_;
   Float_t         phi_reco_jet1_;
   Float_t         m_reco_jet1_;
   Float_t         pt_reco_jet2_;
   Float_t         eta_reco_jet2_;
   Float_t         phi_reco_jet2_;
   Float_t         m_reco_jet2_;
   Float_t         pt_reco_jet3_;
   Float_t         eta_reco_jet3_;
   Float_t         phi_reco_jet3_;
   Float_t         m_reco_jet3_;
   Float_t         pt_reco_jet4_;
   Float_t         eta_reco_jet4_;
   Float_t         phi_reco_jet4_;
   Float_t         m_reco_jet4_;
   Float_t         unnamed;
   Float_t         truth_met;
   Float_t         predicted_pTnu;
   Float_t         resolution_pred_pT;
   Float_t         resolution_default_pT;

   // List of branches
   TBranch        *b_number;   //!
   TBranch        *b_eventNumber;   //!
   TBranch        *b_pt_mc_h;   //!
   TBranch        *b_eta_mc_h_;   //!
   TBranch        *b_phi_mc_h_;   //!
   TBranch        *b_e_mc_h_;   //!
   TBranch        *b_pt_mc_tau_;   //!
   TBranch        *b_eta_mc_tau_;   //!
   TBranch        *b_phi_mc_tau_;   //!
   TBranch        *b_e_mc_tau_;   //!
   TBranch        *b_pt_mc_nuH;   //!
   TBranch        *b_eta_mc_nuH_;   //!
   TBranch        *b_phi_mc_nuH_;   //!
   TBranch        *b_e_mc_nuH_;   //!
   TBranch        *b_pt_mc_nuTau;   //!
   TBranch        *b_eta_mc_nuTau_;   //!
   TBranch        *b_phi_mc_nuTau_;   //!
   TBranch        *b_e_mc_nuTau_;   //!
   TBranch        *b_et_met;   //!
   TBranch        *b_phi_met;   //!
   TBranch        *b_ntau;   //!
   TBranch        *b_nbjet;   //!
   TBranch        *b_njet;   //!
   TBranch        *b_pt_reco_tau1;   //!
   TBranch        *b_eta_reco_tau1_;   //!
   TBranch        *b_phi_reco_tau1;   //!
   TBranch        *b_m_reco_tau1_;   //!
   TBranch        *b_pt_reco_bjet1_;   //!
   TBranch        *b_eta_reco_bjet1_;   //!
   TBranch        *b_phi_reco_bjet1_;   //!
   TBranch        *b_m_reco_bjet1_;   //!
   TBranch        *b_pt_reco_bjet2_;   //!
   TBranch        *b_eta_reco_bjet2_;   //!
   TBranch        *b_phi_reco_bjet2_;   //!
   TBranch        *b_m_reco_bjet2_;   //!
   TBranch        *b_pt_reco_bjet3_;   //!
   TBranch        *b_eta_reco_bjet3_;   //!
   TBranch        *b_phi_reco_bjet3_;   //!
   TBranch        *b_m_reco_bjet3_;   //!
   TBranch        *b_pt_reco_bjet4_;   //!
   TBranch        *b_eta_reco_bjet4_;   //!
   TBranch        *b_phi_reco_bjet4_;   //!
   TBranch        *b_m_reco_bjet4_;   //!
   TBranch        *b_pt_reco_jet1_;   //!
   TBranch        *b_eta_reco_jet1_;   //!
   TBranch        *b_phi_reco_jet1_;   //!
   TBranch        *b_m_reco_jet1_;   //!
   TBranch        *b_pt_reco_jet2_;   //!
   TBranch        *b_eta_reco_jet2_;   //!
   TBranch        *b_phi_reco_jet2_;   //!
   TBranch        *b_m_reco_jet2_;   //!
   TBranch        *b_pt_reco_jet3_;   //!
   TBranch        *b_eta_reco_jet3_;   //!
   TBranch        *b_phi_reco_jet3_;   //!
   TBranch        *b_m_reco_jet3_;   //!
   TBranch        *b_pt_reco_jet4_;   //!
   TBranch        *b_eta_reco_jet4_;   //!
   TBranch        *b_phi_reco_jet4_;   //!
   TBranch        *b_m_reco_jet4_;   //!
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

   fChain->SetBranchAddress("number", &number, &b_number);
   fChain->SetBranchAddress("eventNumber", &eventNumber, &b_eventNumber);
   fChain->SetBranchAddress("pt_mc_h", &pt_mc_h, &b_pt_mc_h);
   fChain->SetBranchAddress("eta_mc_h_", &eta_mc_h_, &b_eta_mc_h_);
   fChain->SetBranchAddress("phi_mc_h_", &phi_mc_h_, &b_phi_mc_h_);
   fChain->SetBranchAddress("e_mc_h_", &e_mc_h_, &b_e_mc_h_);
   fChain->SetBranchAddress("pt_mc_tau_", &pt_mc_tau_, &b_pt_mc_tau_);
   fChain->SetBranchAddress("eta_mc_tau_", &eta_mc_tau_, &b_eta_mc_tau_);
   fChain->SetBranchAddress("phi_mc_tau_", &phi_mc_tau_, &b_phi_mc_tau_);
   fChain->SetBranchAddress("e_mc_tau_", &e_mc_tau_, &b_e_mc_tau_);
   fChain->SetBranchAddress("pt_mc_nuH", &pt_mc_nuH, &b_pt_mc_nuH);
   fChain->SetBranchAddress("eta_mc_nuH_", &eta_mc_nuH_, &b_eta_mc_nuH_);
   fChain->SetBranchAddress("phi_mc_nuH_", &phi_mc_nuH_, &b_phi_mc_nuH_);
   fChain->SetBranchAddress("e_mc_nuH_", &e_mc_nuH_, &b_e_mc_nuH_);
   fChain->SetBranchAddress("pt_mc_nuTau", &pt_mc_nuTau, &b_pt_mc_nuTau);
   fChain->SetBranchAddress("eta_mc_nuTau_", &eta_mc_nuTau_, &b_eta_mc_nuTau_);
   fChain->SetBranchAddress("phi_mc_nuTau_", &phi_mc_nuTau_, &b_phi_mc_nuTau_);
   fChain->SetBranchAddress("e_mc_nuTau_", &e_mc_nuTau_, &b_e_mc_nuTau_);
   fChain->SetBranchAddress("et_met", &et_met, &b_et_met);
   fChain->SetBranchAddress("phi_met", &phi_met, &b_phi_met);
   fChain->SetBranchAddress("ntau", &ntau, &b_ntau);
   fChain->SetBranchAddress("nbjet", &nbjet, &b_nbjet);
   fChain->SetBranchAddress("njet", &njet, &b_njet);
   fChain->SetBranchAddress("pt_reco_tau1", &pt_reco_tau1, &b_pt_reco_tau1);
   fChain->SetBranchAddress("eta_reco_tau1_", &eta_reco_tau1_, &b_eta_reco_tau1_);
   fChain->SetBranchAddress("phi_reco_tau1", &phi_reco_tau1, &b_phi_reco_tau1);
   fChain->SetBranchAddress("m_reco_tau1_", &m_reco_tau1_, &b_m_reco_tau1_);
   fChain->SetBranchAddress("pt_reco_bjet1_", &pt_reco_bjet1_, &b_pt_reco_bjet1_);
   fChain->SetBranchAddress("eta_reco_bjet1_", &eta_reco_bjet1_, &b_eta_reco_bjet1_);
   fChain->SetBranchAddress("phi_reco_bjet1_", &phi_reco_bjet1_, &b_phi_reco_bjet1_);
   fChain->SetBranchAddress("m_reco_bjet1_", &m_reco_bjet1_, &b_m_reco_bjet1_);
   fChain->SetBranchAddress("pt_reco_bjet2_", &pt_reco_bjet2_, &b_pt_reco_bjet2_);
   fChain->SetBranchAddress("eta_reco_bjet2_", &eta_reco_bjet2_, &b_eta_reco_bjet2_);
   fChain->SetBranchAddress("phi_reco_bjet2_", &phi_reco_bjet2_, &b_phi_reco_bjet2_);
   fChain->SetBranchAddress("m_reco_bjet2_", &m_reco_bjet2_, &b_m_reco_bjet2_);
   fChain->SetBranchAddress("pt_reco_bjet3_", &pt_reco_bjet3_, &b_pt_reco_bjet3_);
   fChain->SetBranchAddress("eta_reco_bjet3_", &eta_reco_bjet3_, &b_eta_reco_bjet3_);
   fChain->SetBranchAddress("phi_reco_bjet3_", &phi_reco_bjet3_, &b_phi_reco_bjet3_);
   fChain->SetBranchAddress("m_reco_bjet3_", &m_reco_bjet3_, &b_m_reco_bjet3_);
   fChain->SetBranchAddress("pt_reco_bjet4_", &pt_reco_bjet4_, &b_pt_reco_bjet4_);
   fChain->SetBranchAddress("eta_reco_bjet4_", &eta_reco_bjet4_, &b_eta_reco_bjet4_);
   fChain->SetBranchAddress("phi_reco_bjet4_", &phi_reco_bjet4_, &b_phi_reco_bjet4_);
   fChain->SetBranchAddress("m_reco_bjet4_", &m_reco_bjet4_, &b_m_reco_bjet4_);
   fChain->SetBranchAddress("pt_reco_jet1_", &pt_reco_jet1_, &b_pt_reco_jet1_);
   fChain->SetBranchAddress("eta_reco_jet1_", &eta_reco_jet1_, &b_eta_reco_jet1_);
   fChain->SetBranchAddress("phi_reco_jet1_", &phi_reco_jet1_, &b_phi_reco_jet1_);
   fChain->SetBranchAddress("m_reco_jet1_", &m_reco_jet1_, &b_m_reco_jet1_);
   fChain->SetBranchAddress("pt_reco_jet2_", &pt_reco_jet2_, &b_pt_reco_jet2_);
   fChain->SetBranchAddress("eta_reco_jet2_", &eta_reco_jet2_, &b_eta_reco_jet2_);
   fChain->SetBranchAddress("phi_reco_jet2_", &phi_reco_jet2_, &b_phi_reco_jet2_);
   fChain->SetBranchAddress("m_reco_jet2_", &m_reco_jet2_, &b_m_reco_jet2_);
   fChain->SetBranchAddress("pt_reco_jet3_", &pt_reco_jet3_, &b_pt_reco_jet3_);
   fChain->SetBranchAddress("eta_reco_jet3_", &eta_reco_jet3_, &b_eta_reco_jet3_);
   fChain->SetBranchAddress("phi_reco_jet3_", &phi_reco_jet3_, &b_phi_reco_jet3_);
   fChain->SetBranchAddress("m_reco_jet3_", &m_reco_jet3_, &b_m_reco_jet3_);
   fChain->SetBranchAddress("pt_reco_jet4_", &pt_reco_jet4_, &b_pt_reco_jet4_);
   fChain->SetBranchAddress("eta_reco_jet4_", &eta_reco_jet4_, &b_eta_reco_jet4_);
   fChain->SetBranchAddress("phi_reco_jet4_", &phi_reco_jet4_, &b_phi_reco_jet4_);
   fChain->SetBranchAddress("m_reco_jet4_", &m_reco_jet4_, &b_m_reco_jet4_);
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
