// main425.cc is a part of the PYTHIA event generator.
// Copyright (C) 2024 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Marius Utheim <marius.m.utheim@jyu.fi>
//          Torbjorn Sjostrand <torbjorn.sjostrand@hep.lu.se>

// Keywords: heavy ions, cross sections.

// This example calculates proton-oxygen cross sections at varying energies.

#include "Pythia8/Pythia.h"

using namespace Pythia8;

int main() {

  // Set up momentum grid for fixed-target option.
  double pMin = 1e2, pMax = 1e11;
  int nPts = 4;
  vector<double> pLabs = logSpace(nPts, pMin, pMax);
  double dr = pow(pMax / pMin, 1. / (nPts - 1));

  // Histograms.
  Hist sigTot("Total", nPts, pMin / sqrt(dr), pMax * sqrt(dr), true);
  Hist sigInel("Inelastic", nPts, pMin / sqrt(dr), pMax * sqrt(dr), true);

  // Iterate over momenta. Initialize for p 16O(xygen).
  for (double pNow : pLabs) {
    Pythia pythia;
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 1000080160");

    // Initialize for equivalent proton-nucleon CM energy.
    pythia.readString("Beams:frameType = 1");
    double eCMNow = ( Vec4(0., 0., pNow, pNow * sqrt(1 + pow2(0.938 / pNow)))
                    + Vec4(0., 0., 0., 0.938) ).mCalc();
    pythia.settings.parm("Beams:eCM", eCMNow);
    // Alternatively use fixed-target frame, but currently this is
    // numerically unstable at the highest energies.
    // pythia.readString("Beams:frameType = 3");
    // pythia.settings.parm("Beams:pzA", pNow);
    // pythia.settings.parm("Beams:pzb", 0.);

    // Optionally reuse initialization information (if it exists, see main424).
    //pythia.readString("HeavyIon:SigFitReuseInit = 2");
    //pythia.readString("HeavyIon:SigFitInitFile = main424.sigfit");

    // Initialize.
    if (!pythia.init()) {
      cout << "Pythia failed to initialize." << endl;
      return -1;
    }

    // Generate events to get statistics.
    for (int iEvent = 0; iEvent < 4000; ++iEvent)
      pythia.next();

    // Fill histograms.
    sigTot.fill(pNow, pythia.info.sigmaGen());
    sigInel.fill(pNow, pythia.info.sigmaGen(101));
  }

  // Print histograms.
  cout << sigTot << sigInel;

  // Plot histograms.
  HistPlot plt("plot425");
  plt.frame("fig425", "p ${}^{16}$O cross sections",
    "$p_{Lab}$ (GeV)", "$\\sigma$ (mb)", 6.4, 4.8);
  plt.add(sigTot, "-", "Total");
  plt.add(sigInel, "--", "Inelastic");
  plt.plot(0.5 * pMin, 2. * pMax, 0., 800., false, true);

  return 0;
}
