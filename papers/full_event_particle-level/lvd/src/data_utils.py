import numpy as np
import pandas as pd
import math

# filepath dictionaries
mcpath_filepath = {
    "mc0.4-0.5_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.4-0.5_pom.dat",
    "mc0.5-0.6_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.5-0.6_pom.dat",
    "mc0.6-0.7_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.6-0.7_pom.dat",
    "mc0.7-0.8_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.7-0.8_pom.dat",
    "mc0.8-0.9_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.8-0.9_pom.dat",
    "mc0.9-1.0_pom": "/home/tkj9ep/dev/fox-diffusion/papers/full_event_particle-level/lvd/data/mc0.9-1.0_pom.dat"
}

"""
S: number of events
M: # of particles
DV: 4 particle array
DE: 4 global array
"""

# convert pom data into npz file
def main():
    filepath = mcpath_filepath["mc0.4-0.5_pom"]

    results, cols = convert_df_to_np(filepath)
    print(cols)
    print(results[0, :])

    first_row_results = results[0, :]

    requirements_log = check_events(results)

    # for idx in range(len(cols)):
    #     print(f"{cols[idx]}: {first_row_results[idx]}")

    # npz_cols = ["detector_vectors", "detector_event", "detector_mask", "particle_vectors",
    # "particle_event", "particle_mask", "particle_types"]

    # # Particle Level 
    # particle_vectors, particle_event, particle_mask, particle_types = particle_level_arrays(num_data = results)    
    
    # # Detector Level
    # detector_vectors, detector_event, detector_mask = detector_level_arrays(particle_vectors, particle_event)

    # # save into npz file
    # np.savez("./jlab_data.npz", detector_vectors = detector_vectors, detector_event = detector_event,
    # detector_mask = detector_mask, particle_vectors = particle_vectors, particle_event = particle_event,
    # particle_mask = particle_mask, particle_types = particle_types)

    return None

# simply take the particle vectors and event and pollute
# make noise very small 
# don't add noise to last vector
# E^2 = M^2 + P^2
# mass of pion  = 
# mass of proton = 
def detector_level_arrays(particle_vectors, particle_event, mean = 0, var = 1, S = 10, M = 5):

    # add noise
    vectors_noise = np.random.normal(mean, var, particle_vectors.shape)
    event_noise = np.random.normal(mean, var, particle_event.shape)

    detector_vectors = particle_vectors + vectors_noise
    detector_event = particle_event + event_noise
    detector_mask = np.full((S, M), True)

    return detector_vectors, detector_event, detector_mask

# get the particle level npz files
def particle_level_arrays(num_data, N = 5, DV = 4, DE = 4):

    # assume for this case that 0 = q, 1 = p1, 2 = k1, 3 = k2, 4 = p2
    
    # set the variables
    S = num_data.shape[0]
    particle_vector = np.zeros(shape = (S, N, DV))
    particle_event = np.zeros(shape = (S, DE))
    particle_mask = np.full((S, N), True)
    particle_types = np.zeros(shape = (S, N))
    
    # fill the vectors
    for sample_idx in range(S):

        # parse the information from the dataset
        particle_momentae = get_particle_momenta(num_data[sample_idx])
        global_variables = get_global_variables(num_data[sample_idx], DE = DE)

        particle_vector[sample_idx] = particle_momentae
        particle_event[sample_idx] = global_variables
        particle_types[sample_idx] = np.array([0, 1, 2, 3, 4])

    return particle_vector, particle_event, particle_mask, particle_types

# get the particle momentae from the data
# note this is hardcoded for this particular instance
def get_particle_momenta(particle_event):
    q = particle_event[4:8]
    p1 = particle_event[8:12]
    k1 = particle_event[12:16]
    k2 = particle_event[16:20]
    p2 = particle_event[20:24]

    return np.array([q, p1, k1, k2, p2])

# get the global variables
# note: this assumes that the global variables are in the front
def get_global_variables(particle_event, DE):

    return np.array(particle_event[0:DE])

def display_file(filepath):
    count = 0

    with open(filepath, 'r') as file:
        if count <= 10:
            for line in file:
                print(line.strip())
                count+=1
            else:
                return


def convert_df_to_np(filepath):
    df = pd.read_csv(filepath, delimiter = '\t')
    results = []
    cols = df.columns.values.tolist()[0].split() # gets the columns

    for i in range(10):
        entry = df.iloc[i].tolist()[0].split()
        int_list = [float(item) for item in entry]
        results.append(int_list)

    # perform EDA

    return np.array(results), np.array(cols)


def eda(df_np):
    eda_dict = {}

    eda_dict["mean"] = np.mean(df_np, axis = 0)
    eda_dict["std"] = np.std(df_np, axis = 0)


    return eda_dict


"""
Test the validity of each event with some margin of error due to experimental variables. These functions will 
assume a certain order of features. Operations will be done component-wise.

1. Test Conservation of Energy and Momentum
2. Test Conservation of Mass
"""

def check_events(data, acceptable_error = .05):

    num_events = data.shape[0]
    requirement_log = {}
    
    # loop through each event
    for event_idx in range(num_events):
        event = data[event_idx, :]

        q = np.array(event[4:8])
        p1 = np.array(event[8:12])
        k1 = np.array(event[12:16])
        k2 = np.array(event[16:20])
        p2 = np.array(event[20:24])

        print(f"Event {event_idx}: ======================================\n")
        satisfies_momentum = check_momentum(q, p1, p2, k1, k2, acceptable_error)
        satisfies_energy = check_energy(q, p1, p2, k1, k2, acceptable_error)
        satisfies_mass = check_mass(p2, k1, k2, acceptable_error)

        requirement_log[event_idx] = {"Momentum": satisfies_momentum, "Energy": satisfies_energy, "Mass": satisfies_mass}

        print(f"Satisfies Conservation of Momentum: {satisfies_momentum}")
        print(f"Satisfies Conservation of Energy: {satisfies_energy}")
        print(f"Satisfies Conservation of Mass: {satisfies_mass}")
        print(f"===============================================\n")

    return requirement_log

# double checks the conservation of momentum requirement
def check_momentum(q, p1, p2, k1, k2, acceptable_error):
    # initial momentum
    total_initial_momentum = q + p1

    # final momentum
    total_final_momentum = p2 + k1 + k2

    print(f"Initial Momentum: {total_initial_momentum}\nFinal Momentum: {total_final_momentum}\n")

    return abs(total_final_momentum - total_initial_momentum) <= acceptable_error 

# double checks the conservation of energy
# assumes that 0th vector is energy
def check_energy(q, p1, p2, k1, k2, acceptable_error):
    # initial energy
    total_initial_energy = q[0] + p1[0]

    # final energy
    total_final_energy = p2[0] + k1[0] + k2[0]

    return abs(total_final_energy - total_initial_energy) <= acceptable_error


# double check the conservation of mass requirement
def check_mass(p2, k1, k2, acceptable_error):
    # calculate the masses
    p2_mass = p2[0]**2 - (p2[1]**2 + p2[2]**2 + p2[3]**2)
    k2_mass = k2[0]**2 - (k2[1]**2 + k2[2]**2 + k2[3]**2)
    k1_mass = k1[0]**2 - (k1[1]**2 + k1[2]**2 + k1[3]**2)

    print(f"Mass of Recoil Proton (p1): {p2_mass}\nMass of Positive Pion (k1): {k1_mass}\nMass of Negative Pion (k2): {k2_mass}\n")

    return abs(np.array([np.sqrt(p2_mass), np.sqrt(k1_mass), np.sqrt(k2_mass)]) - np.array([.9382721, .1395704, .1395704])) <= acceptable_error


main()