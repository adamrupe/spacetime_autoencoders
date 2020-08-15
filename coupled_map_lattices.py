import numpy as np


def map_lattice(initial_condition, time_steps, map_function, coupling, **params):
    '''
    Radius-1 (nearest-neighbor) coupled map lattice simulator. Evolves a given
    initial condition for a given number of time steps using the given iterated map
    and coupling strength, and yields the resulting spacetime field.

    Parameterts
    -----------
    initial_condition: array-like
        Initial condition for the map lattice. As this is a one-dimensional map
        lattice, the initial condition is a one-dimensional array, or array-like
        object. Elements of the array are in general floats, and will typically
        be in the unit invterval [0,1].

    time_steps: int
        Number of time steps to evolve the initial condition.

    map_function: Python function
        Time-evolution function of the chosen iterated map that defines the coupled map lattice.
        Typically a function that maps the unit interval to itself. Any needed
        parameters (e.g. nonlinearity strength) for this map given through
        **params.

    coupling: float
        Coupling strenght of the map lattice; how the left and right neighborhos
        influence the evolution of each site on the lattice. Takes values
        on the unit interval [0,1].

    Returns
    -------
    out: ndarray
        2D ndarray representation of the spacetime field produced by the map lattice.
        The 0th axis (vertical) is time and the 1st axis (horizontal) is space,
        so that out[t] gives the spatial lattice of the coupled map lattice at time
        t.
    '''
    c = coupling

    lattice_size = len(initial_condition)
    current_state = np.copy(initial_condition)

    spacetime = np.zeros((time_steps, lattice_size))
    spacetime[0] = current_state

    for t in range(time_steps):
        mapped_state = map_function(current_state, **params)
        mapped_left = np.roll(mapped_state, 1)
        mapped_right = np.roll(mapped_state, -1)

        current_state = np.mod((1-c)*mapped_state + (c/2)*(mapped_left + mapped_right), 1.0)

        spacetime[t] = current_state

    return spacetime


def map_lattice_ensemble(map_function, coupling, time, space, number_of, transient=0, **params):
    '''
    Generator for an ensemble of coupled map lattice spacetime fields, each with
    a different random initial condition.

    Parameters
    ----------
    map_function: Python function
        Time-evolution function of the chosen iterated map that defines the coupled map lattice.
        Typically a function that maps the unit interval to itself. Any needed
        parameters (e.g. nonlinearity strength) for this map given through
        **params.

    coupling: float
        Coupling strenght of the map lattice; how the left and right neighborhos
        influence the evolution of each site on the lattice. Takes values
        on the unit interval [0,1].

    time: int
        Size of the time dimension for the spacetime fields. That is, number of
        time steps - 1, since the initial condition is (potentially) included.

    space: int
        Size of the space dimension for the spacetime fields. This is the lenght
        of each spatial lattice.

    number_of: int
        Number of spacetime field instances generated for the ensemble.

    transient: int, optional (default=0)
        Number of initial time steps to exclude from each spacetime field generated.
        Resulting spacetime fields will still have the specified space and time
        dimensions, i.e. the resulting ndarrays will have shape (time,space).
    '''
    t = time + transient - 1
    for _ in range(number_of):
        initial = np.random.rand(space)
        yield map_lattice(initial, t, map_function, coupling, **params)[transient:]


def circle_map(state, nonlinearity):
    '''
    Time-evolution function for the circle map. This maps the unit interval to
    itself, using the fuction f below.

    Parameters
    ----------
    state: float
        Initial condition to be mapped one time-step by the circle map. Should be
        a point on the unit interval [0,1].

    nonlinearity: float
        Nonlinearity parameter of the circle map. When nonlinearity=0, the circle map
        is just a linear phase shift.

    Returns
    -------
    out: float
        Output of mapping the given initial state under the circle map with the
        given nonlinearity strength.
    '''
    r = nonlinearity
    x = state

    f = x + 0.5 - ((r/(2.0*np.pi))*np.sin(2.0*np.pi*x))

    return np.mod(f, 1.0)


def logistic_map(state, nonlinearity):
    '''
    Time-evolution function for the logistic map. This maps the unit interval to
    itself, using the fuction f below.

    Parameters
    ----------
    state: float
        Initial condition to be mapped one time-step by the logistic map. Should
        be a point on the unit interval [0,1].

    nonlinearity: float
        Nonlinearity parameter of the logistic map.

    Returns
    -------
    out: float
        Out of mapping the given initial state under the logistic map with the
        given nonlinearity strength.
    '''
    r = nonlinearity
    x = state

    f = r*x*(1-x)

    return np.mod(f, 1.0)
