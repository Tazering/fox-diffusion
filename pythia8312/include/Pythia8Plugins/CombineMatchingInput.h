// CombineMatchingInput.h is a part of the PYTHIA event generator.
// Copyright (C) 2024 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the following classes:
// JetMatchingAlpgenInputAlpgen: combines Alpgen-style MLM matching
//   with Alpgen native format event input.
// JetMatchingMadgraphInputAlpgen: combines Madgraph-style MLM matching
//   with Alpgen native format event input.
// CombineMatchingInput: invokes Alpgen- or Madgraphs-style MLM matching
//   for Madgraph LHEF or Alpgen native format event input.

#ifndef Pythia8_CombineMatchingInput_H
#define Pythia8_CombineMatchingInput_H

// Includes and namespace
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/GeneratorInput.h"
#include "Pythia8Plugins/JetMatching.h"

namespace Pythia8 {

//==========================================================================

// JetMatchingAlpgenInputAlpgen:
// A small UserHooks class that gives the functionality of both AlpgenHooks
// and JetMatchingAlpgen. These classes have one overlapping function,
// 'initAfterBeams()', which is overridden here such that both are called.

class JetMatchingAlpgenInputAlpgen : public AlpgenHooks,
  public JetMatchingAlpgen {

public:

  // Constructor and destructor.
  JetMatchingAlpgenInputAlpgen(Pythia& pythia) : AlpgenHooks(pythia),
    JetMatchingAlpgen() { }
  ~JetMatchingAlpgenInputAlpgen() {}

  // Initialisation.
  virtual bool initAfterBeams() {
    if (!AlpgenHooks::initAfterBeams()) return false;
    if (!JetMatchingAlpgen::initAfterBeams()) return false;
    return true;
  }

  // Process level vetos.
  virtual bool canVetoProcessLevel() {
    return JetMatchingAlpgen::canVetoProcessLevel();
  }
  virtual bool doVetoProcessLevel(Event & proc) {
    return JetMatchingAlpgen::doVetoProcessLevel(proc);
  }

  // Parton level vetos (before beam remnants and resonance decays).
  virtual bool canVetoPartonLevelEarly() {
    return JetMatchingAlpgen::canVetoPartonLevelEarly();
  }
  virtual bool doVetoPartonLevelEarly(const Event &proc) {
    return JetMatchingAlpgen::doVetoPartonLevelEarly(proc);
  }

};

//==========================================================================

// JetMatchingMadgraphInputAlpgen:
// A small UserHooks class that gives the functionality of both AlpgenHooks
// and JetMatchingMadgraph. These classes have one overlapping function,
// 'initAfterBeams()', which is overridden here such that both are called.

class JetMatchingMadgraphInputAlpgen : public AlpgenHooks,
  public JetMatchingMadgraph {

public:

  // Constructor and destructor.
  JetMatchingMadgraphInputAlpgen(Pythia& pythia) : AlpgenHooks(pythia),
    JetMatchingMadgraph() {}
  ~JetMatchingMadgraphInputAlpgen() {}

  // Initialisation.
  virtual bool initAfterBeams() {
    // Madgraph matching parameters should not be set from Alpgen file.
    settingsPtr->flag("JetMatching:setMad",false);
    if (!AlpgenHooks::initAfterBeams()) return false;
    if (!JetMatchingMadgraph::initAfterBeams()) return false;
    return true;
  }

  // Process level vetos.
  virtual bool canVetoProcessLevel() {
    return JetMatchingMadgraph::canVetoProcessLevel();
  }
  virtual bool doVetoProcessLevel(Event& proc) {
    return JetMatchingMadgraph::doVetoProcessLevel(proc);
  }

  // Parton level vetos (before beam remnants and resonance decays).
  virtual bool canVetoPartonLevelEarly() {
    return JetMatchingMadgraph::canVetoPartonLevelEarly();
  }
  virtual bool doVetoPartonLevelEarly(const Event& proc) {
    return JetMatchingMadgraph::doVetoPartonLevelEarly(proc);
  }

};

//==========================================================================

class CombineMatchingInput {

public:

  // Constructor and destructor.
  CombineMatchingInput() {}
  ~CombineMatchingInput() {}

  // Return a hook relevant for combination of input and matching.
  void setHook(Pythia& pythia) {

    // Find input source and matching scheme.
    bool isAlpgenFile = ( pythia.word("Alpgen:file") != "void" );
    int  scheme = pythia.mode("JetMatching:scheme");

    // Return relevant UserHooks.
    if (isAlpgenFile) {
      if (scheme == 2)
        hook = make_shared<JetMatchingAlpgenInputAlpgen>(pythia);
      if (scheme == 1)
        hook = make_shared<JetMatchingMadgraphInputAlpgen>(pythia);
    } else {
      if (scheme == 2)
        hook = make_shared<JetMatchingAlpgen>();
      if (scheme == 1)
        hook = make_shared<JetMatchingMadgraph>();
    }

    pythia.setUserHooksPtr(hook);

    return;
  }

  shared_ptr<UserHooks> hook;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_CombineMatchingInput_H
