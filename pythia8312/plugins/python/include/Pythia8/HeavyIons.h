// HeavyIons.h is a part of the PYTHIA event generator.
// Copyright (C) 2024 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// This file contains the definition of the HeavyIons class which
// Provides Pythia with infrastructure to combine several nucleon
// collisions into a single heavy ion collision event. This file also
// includes the definition of the Angantyr class which implements the
// default model for heavy ion collisions in Pythia.

#ifndef Pythia8_HeavyIons_H
#define Pythia8_HeavyIons_H

#include "Pythia8/HIInfo.h"
#include "Pythia8/PhysicsBase.h"

namespace Pythia8 {

// Forward declaration of the Pythia class.
class Pythia;

//==========================================================================

// HeavyIons contains several standard Pythia objects to allow for
// the combination of different kinds of nucleon-nucleon events into
// a heavy ion collision event. The actual model for doing this must
// be implemented in a subclass overriding the the virtual'init()'
// and 'next()' functions.

class HeavyIons : public PhysicsBase {

public:

  // The constructor needs a reference to the main Pythia object to
  // which it will belong. A HeavyIons object cannot belong to more
  // than one main Pythia object.
  HeavyIons(Pythia& mainPythiaIn)
    : mainPythiaPtr(&mainPythiaIn), HIHooksPtr(0),
      pythia(vector<Pythia*>(1, &mainPythiaIn)) {
  }

  // Destructor.
  virtual ~HeavyIons() {}

  // Virtual function to be implemented in a subclass. This will be
  // called in the beginning of the Pythia::init function if the mode
  // HeavyIon:mode is set non zero. The return value should be true
  // if this object is able to handle the requested collision
  // type. If false Pythia::init will set HeavyIon:mode to zero but
  // will try to cope anyway.
  virtual bool init() = 0;

  // Virtual function to be implemented in a subclass. This will be
  // called in the beginning of the Pythia::next function if
  // HeavyIon:mode is set non zero. After the call, Pythia::next will
  // return immediately with the return value of this function.
  virtual bool next() = 0;

  // Static function to allow Pythia to duplicate some setting names
  // to be used for secondary Pythia objects.
  static void addSpecialSettings(Settings& settings);

  // Return true if the beams in the Primary Pythia object contains
  // heavy ions.
  static bool isHeavyIon(Settings& settings);

  // Possibility to pass in pointer for special heavy ion user hooks.
  bool setHIUserHooksPtr(HIUserHooksPtr userHooksPtrIn) {
    HIHooksPtr = userHooksPtrIn; return true;
  }

  // Set beam kinematics.
  virtual bool setKinematics(double /*eCMIn*/) {
    loggerPtr->ERROR_MSG("method not implemented for this heavy ion model");
    return false; }
  virtual bool setKinematics(double /*eAIn*/, double /*eBIn*/) {
    loggerPtr->ERROR_MSG("method not implemented for this heavy ion model");
    return false; }
  virtual bool setKinematics(double, double, double, double, double, double) {
    loggerPtr->ERROR_MSG("method not implemented for this heavy ion model");
    return false; }
  virtual bool setKinematics(Vec4, Vec4) {
    loggerPtr->ERROR_MSG("method not implemented for this heavy ion model");
    return false; }

  // Set beam particles.
  virtual bool setBeamIDs(int /*idAIn*/, int /*idBIn*/ = 0) {
    loggerPtr->ERROR_MSG("method not implemented for this heavy ion model");
    return false; }

  // The HIInfo object contains information about the last generated heavy
  // ion event as well as overall statistics of the generated events.
  HIInfo hiInfo;

  // Print out statistics.
  virtual void stat();

public:

  // Update the cross section in the main Pythia Info object using
  // information in the hiInfo object.
  void updateInfo();

  // If subclasses has additional Pythia objects for generating
  // minimum bias nucleon collisions and the main Pythia object is
  // set up to generated a hard signal process, this function can be
  // used to clear all selected processes in a clone of the main
  // Pythia object.
  void clearProcessLevel(Pythia& pyt);

  // Duplicate setting on the form match: to settings on the form HImatch:
  static void setupSpecials(Settings& settings, string match);

  // Copy settings on the form HImatch: to the corresponding match:
  // in the given Pythia object.
  static void setupSpecials(Pythia& p, string match);

  // Save current beam configuration.
  int idProj, idTarg;

  // This is the pointer to the main Pythia object to which this
  // object is assigned.
  Pythia * mainPythiaPtr;

  // Object containing information on inclusive pp cross sections to
  // be used in Glauber calculations in subclasses.
  SigmaTotal sigTotNN;

  // Optional HIUserHooks object able to modify the behavior of the
  // HeavyIon model.
  HIUserHooksPtr HIHooksPtr;

  // The internal Pythia objects. Index zero will always correspond
  // to the mainPythiaPtr.
  vector<Pythia *> pythia;

  // The names associated with the secondary pythia objects.
  vector<string> pythiaNames;

  // The Info objects associated to the secondary the secondary
  // pythia objects.
  vector<Info*> info;

  // Helper class to gain access to the Info object in a pythia
  // instance.
  struct InfoGrabber : public UserHooks {

    // Only one function: return the info object.
    Info * getInfo() {
      return infoPtr;
    }

  };

};

//==========================================================================

// The default HeavyIon model in Pythia.

class Angantyr : public HeavyIons {

public:

  // Enumerate the different internal Pythia objects.
  enum PythiaObject {
    HADRON = 0,   // For hadronization only.
    MBIAS = 1,    // Minimum Bias processed.
    SASD = 2,     // Single diffractive as one side of non-diffractive.
    SIGPP = 3,    // Optional object for signal processes (pp).
    SIGPN = 4,    // Optional object for signal processes (pn).
    SIGNP = 5,    // Optional object for signal processes (np).
    SIGNN = 6,    // Optional object for signal processes (nn).
    ALL = 7       // Indicates all objects.
  };

public:

  // The constructor needs a reference to the main Pythia object to
  // which it will belong. A Angantyr object cannot belong to more
  // than one main Pythia object.
  Angantyr(Pythia& mainPythiaIn);

  virtual ~Angantyr();

  // Initialize Angantyr.
  virtual bool init() override;

  // Produce a collision involving heavy ions.
  virtual bool next() override;

  // Set UserHooks for specific (or ALL) internal Pythia objects.
  bool setUserHooksPtr(PythiaObject sel, UserHooksPtr userHooksPtrIn);

  // Set beam kinematics.
  bool setKinematics(double eCMIn) override;
  bool setKinematics(double eAIn, double eBIn) override;
  bool setKinematics(double, double, double, double, double, double) override;
  bool setKinematics(Vec4, Vec4) override;
  bool setKinematics();

  // Set beam IDs.
  bool setBeamIDs(int idAIn, int idBIn = 0) override;

  // Make sure the correct information is available irrespective of frame type.
  void unifyFrames();

  // Print the Angantyr banner.
  void banner() const;

  // Subcollisions for the current event.
  const SubCollisionSet& subCollisions() const {
    return subColls; }

  // Get the underlying subcollision model.
  const SubCollisionModel& subCollisionModel() const {
    return *collPtr.get(); }

  // Get the underlying impact parameter generator.
  const ImpactParameterGenerator impactParameterGenerator() const {
    return *bGenPtr.get(); }

  // Projectile nucleus configuration for the current event.
  const Nucleus& projectile() const {
    return proj; }

  // Target nucleus configuration for the current event.
  const Nucleus& target() const {
    return targ; }

  // The underlying projectile nucleus model.
  const NucleusModel& projectileModel() const {
    return *projPtr.get(); }

  // The underlying target nucleus model.
  const NucleusModel& targetModel() const {
    return *targPtr.get(); }

  // Hadronic cross sections used by the subcollision model.
  const SigmaTotal sigmaNN() const {
    return sigTotNN; }

public:

  virtual void onInitInfoPtr() override {
    registerSubObject(sigTotNN); }

  // Figure out what beams the user want.
  void setBeamKinematics(int idA, int idB);

  // Initiaize a specific Pythia object and optionally run a number
  // of events to get a handle of the cross section.
  bool init(PythiaObject sel, string name, int n = 0);

  // Setup an EventInfo object from a Pythia instance.
  EventInfo mkEventInfo(Pythia &, Info &, const SubCollision * coll = 0);

  // Generate events from the internal Pythia oblects;
  EventInfo getSignal(const SubCollision& coll);
  EventInfo getND() { return getMBIAS(0, 101); }
  EventInfo getND(const SubCollision& coll) { return getMBIAS(&coll, 101); }
  EventInfo getEl(const SubCollision& coll) { return getMBIAS(&coll, 102); }
  EventInfo getSDP(const SubCollision& coll) { return getMBIAS(&coll, 103); }
  EventInfo getSDT(const SubCollision& coll) { return getMBIAS(&coll, 104); }
  EventInfo getDD(const SubCollision& coll) { return getMBIAS(&coll, 105); }
  EventInfo getCD(const SubCollision& coll) { return getMBIAS(&coll, 106); }
  EventInfo getSDabsP(const SubCollision& coll)
    { return getSASD(&coll, 103); }
  EventInfo getSDabsT(const SubCollision& coll)
    { return getSASD(&coll, 104); }
  EventInfo getMBIAS(const SubCollision * coll, int procid);
  EventInfo getSASD(const SubCollision * coll, int procid);

  bool genAbs(SubCollisionSet& subCollsIn, list<EventInfo>& subEventsIn);
  void addSASD(const SubCollisionSet& subCollsIn);
  bool addDD(const SubCollisionSet& subCollsIn, list<EventInfo>& subEventsIn);
  bool addSD(const SubCollisionSet& subCollsIn, list<EventInfo>& subEventsIn);
  void addSDsecond(const SubCollisionSet& subCollsIn);
  bool addCD(const SubCollisionSet& subCollsIn, list<EventInfo>& subEventsIn);
  void addCDsecond(const SubCollisionSet& subCollsIn);
  bool addEL(const SubCollisionSet& subCollsIn, list<EventInfo>& subEventsIn);
  void addELsecond(const SubCollisionSet& subCollsIn);

  bool buildEvent(list<EventInfo>& subEventsIn);

  bool setupFullCollision(EventInfo& ei, const SubCollision& coll,
    Nucleon::Status projStatus, Nucleon::Status targStatus);
  bool isRemnant(const EventInfo& ei, int i, int past = 1 ) const {
    int statNow = ei.event[i].status()*past;
    if ( statNow == 63 ) return true;
    if ( statNow > 70 && statNow < 80 )
      return isRemnant(ei, ei.event[i].mother1(), -1);
    return false;
  }
  bool fixIsoSpin(EventInfo& ei);
  EventInfo& shiftEvent(EventInfo& ei);
  static int getBeam(Event& ev, int i);

  // Generate a single diffractive
  bool nextSASD(int proc);

  // Add a diffractive event to an exsisting one. Optionally connect
  // the colours of the added event to the original.
  bool addNucleonExcitation(EventInfo& orig, EventInfo& add,
                            bool colConnect = false);

  // Find the recoilers in the current event to conserve energy and
  // momentum in addNucleonExcitation.
  vector<int> findRecoilers(const Event& e, bool tside, int beam, int end,
                            const Vec4& pdiff, const Vec4& pbeam);

  // Add a sub-event to the final event record.
  void addSubEvent(Event& evnt, Event& sub);
  static void addJunctions(Event& evnt, Event& sub, int coloff);

  // Add a nucleus remnant to the given event. Possibly introducing
  // a new particle type.
  bool addNucleusRemnants();

public:

  // Helper function to construct two transformations that would give
  // the vectors p1 and p2 the total four momentum of p1p + p2p.
  static bool
  getTransforms(Vec4 p1, Vec4 p2, const Vec4& p1p,
                pair<RotBstMatrix,RotBstMatrix>& R12);
  static double mT2(const Vec4& p) { return p.pPos()*p.pNeg(); }
  static double mT(const Vec4& p) { return sqrt(max(mT2(p), 0.0)); }

private:

  // Private UserHooks class to select a specific process.
  struct ProcessSelectorHook: public UserHooks {

    ProcessSelectorHook(): proc(0), b(-1.0) {}

    // Yes we can veto event after process-level selection.
    virtual bool canVetoProcessLevel() {
      return true;
    }

    // Veto any unwanted process.
    virtual bool doVetoProcessLevel(Event&) {
      return proc > 0 && infoPtr->code() != proc;
    }

    // Can set the overall impact parameter for the MPI treatment.
    virtual bool canSetImpactParameter() const {
      return b >= 0.0;
    }

    // Set the overall impact parameter for the MPI treatment.
    virtual double doSetImpactParameter() {
      return b;
    }

    // The wanted process;
    int proc;

    // The selected b-value.
    double b;

  };

  // Holder class to temporarily select a specific process
  struct HoldProcess {

    // Set the given process for the given hook object.
    HoldProcess(shared_ptr<ProcessSelectorHook> hook, int proc,
      double b = -1.0) : saveHook(hook), saveProc(hook->proc), saveB(hook->b) {
      hook->proc = proc;
      hook->b = b;
    }

    // Reset the process of the hook object given in the constructor.
    ~HoldProcess() {
      if ( saveHook ) {
        saveHook->proc = saveProc;
        saveHook->b = saveB;
      }
    }

    // The hook object.
    shared_ptr<ProcessSelectorHook> saveHook;

    // The previous process of the hook object.
    int saveProc;

    // The previous b-value of the hook object.
    double saveB;

  };

  // The process selector for standard minimum bias processes.
  shared_ptr<ProcessSelectorHook> selectMB;

  // The process selector for the SASD object.
  shared_ptr<ProcessSelectorHook> selectSASD;

private:

  static const int MAXTRY = 999;
  static const int MAXEVSAVE = 999;

  // Flag set if there is a specific signal process specified beyond
  // minimum bias.
  bool hasSignal;

  // Whether to do hadronization and hadron level effects.
  bool doHadronLevel;

  // Flag to determine whether to do single diffractive test.
  bool doSDTest;

  // Flag to determine whether to do only Glauber modelling.
  bool glauberOnly;

  // All subcollisions in current collision.
  SubCollisionSet subColls;

  // The underlying SubCollisionModel for generating nucleon-nucleon
  // subcollisions.
  shared_ptr<SubCollisionModel> collPtr;

  // The impact parameter generator.
  shared_ptr<ImpactParameterGenerator> bGenPtr;

  // The projectile and target nuclei in the current collision.
  Nucleus proj;
  Nucleus targ;

  // The underlying nucleus model for generating nuclons inside the
  // projectile and target nucleus.
  shared_ptr<NucleusModel> projPtr;
  shared_ptr<NucleusModel> targPtr;

  // Flag to indicate whether variable energy is enabled.
  bool doVarECM;

  // Different choices in choosing recoilers when adding
  // diffractively excited nucleon.
  int recoilerMode;

  // Different choices for handling impact parameters.
  int bMode;

  // Critical internal error, abort the event.
  bool doAbort;

};

//==========================================================================

} // end namespace Pythia8

#endif // Pythia8_HeavyIons_H
