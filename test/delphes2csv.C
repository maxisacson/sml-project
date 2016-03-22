#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#endif

#include "TChain.h"
#include "TSystem.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

bool isFinalState(GenParticle* part, TClonesArray* particles);
GenParticle* getFinalState(GenParticle* part, TClonesArray* particles);
GenParticle* getChargedHiggs(TClonesArray* particles);
GenParticle* getChargedHiggsNeutrino(GenParticle* part, TClonesArray* particles);
GenParticle* getChargedHiggsTau(GenParticle* part, TClonesArray* particles);
GenParticle* getChargedHiggsTauNeutrino(GenParticle* part, TClonesArray* particles);
int* getDaughtersPID(GenParticle* part, TClonesArray* particles);

int delphes2csv(const char* input) {
    gSystem->Load("libDelphes");

    TChain* chain = new TChain("Delphes");
    chain->Add(input);

    ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
    int64_t nentries = treeReader->GetEntries();
    
    // Branches in the Delphes output
    TClonesArray *b_DEvent = treeReader->UseBranch("Event");
    TClonesArray *b_DParticle = treeReader->UseBranch("Particle");
    //TClonesArray *b_DTrack = treeReader->UseBranch("Track");
    //TClonesArray *b_DTower = treeReader->UseBranch("Tower");
    //TClonesArray *b_DEFlowTrack = treeReader->UseBranch("EFlowTrack");
    //TClonesArray *b_DEFlowPhotoon = treeReader->UseBranch("EFlowPhoton");
    //TClonesArray *b_DEFlowNeutralHadron = treeReader->UseBranch("EFlowNeutralHadron");
    //TClonesArray *b_DGenJet = treeReader->UseBranch("GenJet");
    //TClonesArray *b_DGenMissingET = treeReader->UseBranch("GenMissingET");
    TClonesArray *b_DJet = treeReader->UseBranch("Jet");
    //TClonesArray *b_DElectron = treeReader->UseBranch("Electron");
    //TClonesArray *b_DPhoton = treeReader->UseBranch("Photon");
    //TClonesArray *b_DMuon = treeReader->UseBranch("Muon");
    TClonesArray *b_DMissingET = treeReader->UseBranch("MissingET");
    //TClonesArray *b_DScalarHT = treeReader->UseBranch("ScalarHT"); 

    std::stringstream ss;
    ss << input << ".test.csv";

    FILE* stream = fopen(ss.str().c_str(), "w");
    fprintf(stream, "# eventNumber,");
    fprintf(stream, "pt(mc h+),eta(mc h+),phi(mc h+),e(mc h+),");
    fprintf(stream, "pt(mc tau),eta(mc tau),phi(mc tau),e(mc tau),");
    fprintf(stream, "pt(mc nuH),eta(mc nuH),phi(mc nuH), e(mc nuH),");
    fprintf(stream, "pt(mc nuTau),eta(mc nuTau),phi(eta nuTau),e(mc nuTau),");
    fprintf(stream, "et(met), phi(met),");
    fprintf(stream, "ntau,");
    fprintf(stream, "nbjet,");
    fprintf(stream, "njet,");

    GenParticle* hplus = 0;
    GenParticle* nuH = 0;
    GenParticle* Tau = 0;
    GenParticle* nuTau = 0;

    int maxjet = 4;
    int maxbjet = 4;
    int maxtau = 1;

    int njet = 0;
    int nbjet = 0;
    int ntau = 0;

    for (int i = 0; i < maxtau; ++i)
        fprintf(stream, "pt(reco tau%i),eta(reco tau%i),phi(reco tau%i),m(reco tau%i),", i+1, i+1, i+1, i+1);
    for (int i = 0; i < maxbjet; ++i)
        fprintf(stream, "pt(reco bjet%i),eta(reco bjet%i),phi(reco bjet%i),m(reco bjet%i),", i+1, i+1, i+1, i+1);
    for (int i = 0; i < maxjet; ++i)
        fprintf(stream, "pt(reco jet%i),eta(reco jet%i),phi(reco jet%i),m(reco jet%i),", i+1, i+1, i+1, i+1);

    fprintf(stream, "\n");

    std::vector<Jet*> jets;
    std::vector<Jet*> bjets;
    std::vector<Jet*> taus;


    // Begin event loop
    for (int64_t ientry = 0; ientry < nentries; ++ientry) {
        if (ientry%1000 == 0)
            std::cout << "--- " << ientry << " / " << nentries << std::endl;

        treeReader->ReadEntry(ientry);
        HepMCEvent* event = (HepMCEvent*)b_DEvent->At(0);
        GenParticle* hplus = getChargedHiggs(b_DParticle);
        GenParticle* nuH = getChargedHiggsNeutrino(hplus, b_DParticle);
        GenParticle* Tau = getChargedHiggsTau(hplus, b_DParticle);
        Tau = getFinalState(Tau, b_DParticle);
        GenParticle* nuTau = getChargedHiggsTauNeutrino(Tau, b_DParticle);
        if (hplus && nuH && Tau && nuTau) {

            jets.clear();
            bjets.clear();
            taus.clear();

            njet = 0;
            nbjet = 0;
            ntau = 0;

            for (int ijet = 0; ijet < b_DJet->GetEntries(); ++ijet) {
                Jet* jet = (Jet*)b_DJet->At(ijet);
                if (jet->BTag) {
                    bjets.push_back(jet);
                    if (nbjet < maxbjet) ++nbjet;
                } else if (jet->TauTag) {
                    taus.push_back(jet);
                    if (ntau < maxtau) ++ntau;
                } else {
                    jets.push_back(jet);
                    if (njet < maxjet) ++njet;
                }
            }

            std::sort(bjets.begin(), bjets.end(), [](Jet* j1, Jet* j2){return j1->PT > j2->PT;});
            std::sort(jets.begin(),  jets.end(),  [](Jet* j1, Jet* j2){return j1->PT > j2->PT;});
            std::sort(taus.begin(),  taus.end(),  [](Jet* j1, Jet* j2){return j1->PT > j2->PT;});

            // Fill in truth information about the H+->tau nu decay
            fprintf(stream, "%lli,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,",
                    event->Number,
                    hplus->PT, hplus->Eta, hplus->Phi, hplus->E,
                    Tau->PT, Tau->Eta, Tau->Phi, Tau->E,
                    nuH->PT, nuH->Eta, nuH->Phi, nuH->E,
                    nuTau->PT, nuTau->Eta, nuTau->Phi, nuTau->E);

            // Reco ETMiss information
            MissingET* met = (MissingET*)b_DMissingET->At(0);
            fprintf(stream, "%f,%f,", met->MET, met->Phi);

            // Multiplicities 
            fprintf(stream, "%i,", ntau);
            fprintf(stream, "%i,", nbjet);
            fprintf(stream, "%i,", njet);

            // Reco Taus
            int itau = 0;
            for (; itau < ntau; ++itau)
                fprintf(stream, "%f,%f,%f,%f,",
                        taus[itau]->PT, taus[itau]->Eta, taus[itau]->Phi, taus[itau]->Mass);
            for (int jtau = itau; jtau < maxtau; ++jtau)
                fprintf(stream, "%f,%f,%f,%f,", -999., -999., -999., -999.);

            // Reco bjets
            int ibjet = 0;
            for (; ibjet < nbjet; ++ibjet)
                fprintf(stream, "%f,%f,%f,%f,",
                        bjets[ibjet]->PT, bjets[ibjet]->Eta, bjets[ibjet]->Phi, bjets[ibjet]->Mass);
            for (int jbjet = ibjet; jbjet < maxbjet; ++jbjet)
                fprintf(stream, "%f,%f,%f,%f,", -999., -999., -999., -999.);

            // Reco jets
            int ijet = 0;
            for (; ijet < njet; ++ijet)
                fprintf(stream, "%f,%f,%f,%f,",
                        jets[ijet]->PT, jets[ijet]->Eta, jets[ijet]->Phi, jets[ijet]->Mass);
            for (int jjet = ijet; jjet < maxjet; ++jjet)
               fprintf(stream, "%f,%f,%f,%f,", -999., -999., -999., -999.);


            fprintf(stream, "\n");
        }
    } // End event loop
    fclose(stream);
    return 0;
}

bool isFinalState(GenParticle* part, TClonesArray* particles) {
    GenParticle* d1 = 0;
    GenParticle* d2 = 0;
    if (part->D1 > -1)
        d1 = (GenParticle*)particles->At(part->D1); 
    if (part->D2 > -1)
        d2 = (GenParticle*)particles->At(part->D2); 

    if (d1)
        if (part->PID == d1->PID)
            return false;
    if (d2)
        if (part->PID == d2->PID)
            return false;

    return true;
}

GenParticle* getChargedHiggs(TClonesArray* particles) {
    for (int ip = 0; ip < particles->GetEntries(); ++ip) {
        GenParticle* part = (GenParticle*)particles->At(ip);
        if (isFinalState(part, particles) && abs(part->PID) == 37 ) {
            return part;
        }
    }
    return 0;
}


GenParticle* getFinalState(GenParticle* part, TClonesArray* particles) {
    GenParticle* p = part;
    while(true) {
        int* child = getDaughtersPID(p, particles);
        if (child[0] == p->PID)
            p = (GenParticle*)particles->At(p->D1);
        else if (child[1] == p->PID)
            p = (GenParticle*)particles->At(p->D2);
        else
            break;
    }       
    return p;
}

GenParticle* getChargedHiggsNeutrino(GenParticle* part, TClonesArray* particles) {
    int* child = getDaughtersPID(part, particles);
    if ( abs(child[0]) == 16 )
        return (GenParticle*)particles->At(part->D1);
    if ( abs(child[1]) == 16 )
        return (GenParticle*)particles->At(part->D2);
    return 0;
}

GenParticle* getChargedHiggsTau(GenParticle* part, TClonesArray* particles) {
    int* child = getDaughtersPID(part, particles);
    if ( abs(child[0]) == 15 )
        return (GenParticle*)particles->At(part->D1);
    if ( abs(child[1]) == 15 )
        return (GenParticle*)particles->At(part->D2);
    return 0;
}

GenParticle* getChargedHiggsTauNeutrino(GenParticle* part, TClonesArray* particles) {
    int* child = getDaughtersPID(part, particles);
    if ( abs(child[0]) == 16 )
        return (GenParticle*)particles->At(part->D1);
    if ( abs(child[1]) == 16 )
        return (GenParticle*)particles->At(part->D2);
    return 0;
}

int* getDaughtersPID(GenParticle* part, TClonesArray* particles) {
    int children[2] = {0, 0};

    GenParticle* d1 = 0;
    GenParticle* d2 = 0;
    if (part->D1 > -1)
        d1 = (GenParticle*)particles->At(part->D1); 
    if (part->D2 > -1)
        d2 = (GenParticle*)particles->At(part->D2); 

    if (d1)
        children[0] = d1->PID;
    if (d2)
        children[1] = d2->PID;
    return children;
}
