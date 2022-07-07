"""
Implementation of the particle swarm optimization algorithm
Functions:
    - particle_swarm: Implements the algorithm
"""
import logging
import os
from random import random
from typing import Dict, List, Tuple

import numpy as np

def _update_parameters(param, vel, max_param, min_param, inertia_w, ind_cog,
                       soc_learning, pbest, gbest):
    """
    Update equation for the particle swarm algorithm
    V_ij^(t+1) =
        learning rate : w*V_ij^t
        cognitive part : c1*r1*(pbest_ij - p_ij^t)
        social part : c2*r2*(gbest_ij - p_ij^t)
    Args:
        param - input variables (ij array - i particles, i parameters)
        vel - input velocities (ij array)
        max_param - maximum parameter values
        min_param - minimum parameter values
        inertia_w - inertia weight constant
        ind_cog - individual cognition parameter
        soc_learning - social learning parameter
        pbest - best set of parameters for a certain particle (ij array)
        gbest - best global set of parameters (i array)
    Return:
        Updated parameters and velocities
    """
    r1 = random()
    r2 = random()
    max_param = np.broadcast_to(max_param[:, np.newaxis], param.shape)
    min_param = np.broadcast_to(min_param[:, np.newaxis], param.shape)
    # Update velocity
    part_1 = inertia_w * vel
    part_2 = ind_cog * r1 * (pbest - param)
    part_3 = soc_learning * r2 * (gbest - param)
    v_new = part_1 + part_2 + part_3
    # Check if no parameters are outside the allowed ranges for the parameters
    param_new = param + v_new
    mask_min = param_new < min_param
    mask_max = param_new > max_param
    param_new[mask_min] = min_param[mask_min]
    v_new[mask_min] = -v_new[mask_min]
    param_new[mask_max] = max_param[mask_max]
    v_new[mask_max] = -v_new[mask_max]
    return param_new, v_new


def particle_swarm(func,
                   param_dict: Dict[str, List[float]],
                   *,
                   maximize: bool = True,
                   inert_prop: Tuple[float, float, bool] = (0.9, 0.4, True),
                   ind_cog: float = 2.05,
                   soc_learning: float = 2.05,
                   particles: int = 25,
                   iterations: int = 50,
                   export: bool = False,
                   **func_kwargs):
    """Implementation of the particle swarm algorithm
    Args:
        - func: function to be optimized
        - param_dict: dictionary with parameters and variation range
        - maximize: maximize or minimize the problem (default: maximize)
        - inert_prop: Inertial weight factor (the 2 and 3 term is to
        change the value with the iterations
        - particles: Number of particles (default: 25)
        - iterations: Maximum number of iterations (default: 50)
        - export: Export files with the parameter variation with iterations
        - func_kwargs: Extra arguments to pass to the function
    Return:
        - gfitness: Best value obtained
        - gbest: Best parameters
        - pbest: Best parameters for each particle
        - gbest_array: Array with the gfitness value for each iteration
    """
    logging.info("Starting Particle Swarm Algorithm")
    # Create an array for the inertial factor variation
    inert_factor_low, inert_factor_up, inert_sweep = inert_prop
    if inert_sweep:
        inert_factor = np.linspace(inert_factor_low, inert_factor_up,
                                   iterations)
    else:
        inert_factor = np.ones(iterations) * inert_factor_up
    logging.debug(f"Inertial factor array:\n{inert_factor}")
    # Variable initialization
    # Random array with the start value for the parameters
    # Random array with the start value for the velocities
    # pbest and gbest arrays
    param_names = list(param_dict.keys())
    param_max = np.array([p_max[1] for p_max in param_dict.values()])
    param_min = np.array([p_min[0] for p_min in param_dict.values()])
    param_space = [
        np.random.uniform(param_dict[param][0],
                          param_dict[param][1],
                          size=(particles)) for param in param_names
    ]
    param_space = np.stack(param_space)
    vel_space = [
        np.random.uniform(-max(param_dict[param]),
                          max(param_dict[param]),
                          size=(particles)) for param in param_names
    ]
    vel_space = np.stack(vel_space)
    # Calculation of the function for the initial parameters
    # Create the global best (gbest) and particle best (pbest) arrays
    func_input = {
        param_name: param_space[i]
        for i, param_name in enumerate(param_names)
    }
    func_results = func(**func_input, **func_kwargs)
    if maximize:
        fitness_arg = np.argmax(func_results)
    else:
        fitness_arg = np.argmin(func_results)
    gfitness = func_results[fitness_arg]
    pfitness = func_results
    gbest = param_space[:, fitness_arg].flatten()
    gbest_array = [gfitness]
    pbest = param_space

    # Update the optimization arrays
    iteration = 0
    if export:
        export_param = np.concatenate(
            (param_space.T, func_results[:, np.newaxis], vel_space.T), axis=1)
        np.savetxt(os.path.join("PSO_Results", f"results_{iteration}"),
                   export_param)
        iteration += 1
    while iteration < iterations:
        param_space, vel_space = _update_parameters(
            param_space, vel_space, param_max, param_min,
            inert_factor[iteration - 1], ind_cog, soc_learning, pbest,
            gbest[:, np.newaxis])
        # Update gbest and pbest
        func_input = {
            param_name: param_space[i]
            for i, param_name in enumerate(param_names)
        }
        func_results = func(**func_input, **func_kwargs)
        if maximize:
            fitness_candidate_ind = np.argmax(func_results)
            if func_results[fitness_candidate_ind] > gfitness:
                gfitness = func_results[fitness_candidate_ind]
            pfitness_mask = func_results > pfitness
        else:
            fitness_candidate_ind = np.argmin(func_results)
            if func_results[fitness_candidate_ind] < gfitness:
                gfitness = func_results[fitness_candidate_ind]
            pfitness_mask = func_results < pfitness
        # Update gbest, pfitness and pbest
        gbest = param_space[:, fitness_candidate_ind].flatten()
        logging.debug(param_space)
        logging.debug(gbest)
        gbest_array.append(gfitness)
        # Update the FoM plot
        pfitness[pfitness_mask] = func_results[pfitness_mask]
        pbest[:, pfitness_mask] = param_space[:, pfitness_mask]
        if export:
            export_param = np.concatenate(
                (param_space.T, func_results[:, np.newaxis], vel_space.T),
                axis=1)
            np.savetxt(os.path.join("PSO_Results", f"results_{iteration}"),
                       export_param)
        iteration += 1
    if export:
        np.savetxt(os.path.join("PSO_Results", f"gfitness_res"),
                   np.array(gbest_array).T)
    return gfitness, gbest, pbest, gbest_array


if __name__ == "__main__":

    def test_func(x, y):
        return -np.exp(-x**2) * np.exp(-y**2)

    def test_func_2(x, y):
        return np.sin(x) * np.sin(y) / (x * y)

    fit, gbest, pbest, _ = particle_swarm(test_func_2, {
        "x": [-5, 5],
        "y": [-5, 5]
    },
                                          maximize=True,
                                          export=False)
    print("----Results-----")
    print(fit, gbest, pbest, sep="\n")
