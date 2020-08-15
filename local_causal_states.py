import numpy as np
import daal4py as d4p

from numba import njit
from scipy.stats import chisquare
from collections import Counter
from itertools import product

def dist_from_data(X, Y, Nx, Ny):
    '''
    Creates a conditional distribution P(X,Y) from sample observations of pairs (x,y).
    Joint distribution given as nd array. X and Y assumed to take values from
    [0,1,...,Nx-1] and [0,1,...,Ny-1].

    Parameters
    ----------
    X: array-like
        Sample observations of X. Must take values from [0,1,...,Nx-1],
        where Nx is the number of unique possible outcomes of X.

    Y: array-like
        Sample observations of Y. Must take values from [0,1,...,Ny-1],
        where Ny is the number of unique possible outcomes of Y.

    row_labels: bool, optional (default=False)
        Option to add an extra column, transpose([0,1,...,Nx-1]),
        to the left (i.e. column 0) of the distribution array. This extra
        column serves as labels for X; useful if permutations are performed
        on the distribution array.

    column_labels: bool, optional (default=False)
        Option to add an exra row, [0,1,...,Ny-1], to the top (i.e. row 0)
        of the distribution array. This extra row serves as a labels for Y;
        useful if permutations are performed on the distribution array.

    Returns
    -------
    dist: ndarray
        An ndarray of shape (Nx, Ny) representing the joint distribution of
        X and Y; dist[i,j] = P(X=i, Y=j). Rows and columns are conditional distributions,
        e.g. dist[i] = P(Y | X=i).
    '''
    dist = np.zeros((Nx, Ny), dtype=np.uint64)

    counts = Counter(zip(X,Y))

    for pair, count in counts.items():
        dist[pair] = count

    return dist

def neighborhood(data, radius):
    '''
    Returns list of arrays of neighborhood values for input data. Formatted to
    index the multidimensional array lookup table.

    Parameters
    ----------
    data: array_like
        Input string of data.

    radius: int
        Radius of the neighborood.

    Returns
    -------
    neighborhood: list of arrays
        List of neighbor arrays formatted to index multidimensional array lookup
        table. Of the form [array(leftmost neighbors), ..., array(left nearest neighbors),
        array(data), array(right nearest neighbors), ..., array(rightmost neighbors)]


    *** Need to update to use tuples for fancy indexing, rather than arrays***

    '''
    #initialize lists for left enighbor values and right neighbor values
    left = []
    right = []
    #run loop through radial values up to the radius. left values are run in reverse so that both sides build
    #from inside out. r = 0 is skipped and added on in the last step
    for r in range(radius):
        left += [np.roll(data, radius-r)]
        right += [np.roll(data, -(r+1))]
    neighbors = left + [np.array(data)] + right
    return neighbors

class CausalState(object):
    '''
    Class for the local causal state objects. Mostly a data container -- keeps
    the integer label for the state (State.index), the set of pasts in the
    state (State.pasts), and the weighted average morph for the state (State.morph).

    Could just use a namedtuple, but for now keeping it as a class.
    '''

    def __init__(self, state_index, first_past, first_morph):
        '''
        Initializes the CausalState instance with the state label, the first
        past that belongs to the state, and the morph of that first past.

        Parameters
        ----------
        state_index: int
            Integer label for the state.

        first_past: int
            Label for the past lightcone cluster (past) that has first been
            placed in the CausalState instance.

        first_morph: array
            1D array of counts of futures seen with first_past, i.e.
            the (non-normalized) morph of first_past.
        '''
        self.index = state_index
        self.pasts = {first_past}
        self.counts = np.copy(first_morph)
        self.morph = np.copy(first_morph)
        self.entropy = None

    def update(self, past, morph_counts):
        '''
        Adds the new past lightcone into the state and updates the state's morph,
        which is the aggregate count over futures from all pasts in the state.

        Parameters
        ----------
        past: int
            Label for the new past lightcone cluster (past) being added into the
            state.

        morph_counts: array
            1D array of counts of futures seen with the new past being added into
            the state.
        '''
        self.pasts.add(past)
        self.counts += morph_counts
        # average counts over the contributions from each past of the state
        self.morph = np.copy(self.counts)
        self.morph = np.divide(self.morph, len(self.pasts))
        self.entropy = None

    def normalized_morph(self):
        '''
        Returns the normalized morph of the state.
        This is not needed for the chi squared distribution comparison currently
        used, so don't want to do this calculation unless desired. May be needed
        in the future for different distribution comparisons.
        '''
        # OPT: memoize
        morph = self.morph / np.sum(self.morph)
        return morph

    def morph_entropy(self):
        '''
        Returns the Shannon entropy of the state's morph.
        '''
        if self.entropy is None:
            morph = self.normalized_morph()
            non_zero = morph[morph != 0]
            self.entropy = np.sum(-non_zero * np.log2(non_zero))
        return self.entropy

def chi_squared(X, Y, *args, offset=10, **kwargs):
    '''
    Returns the p value for the scipy 1-way chi_squared test.
    In our use, X should be the morph for a past, and Y the
    morph for a past cluster (local causal state).

    As the distributions we encounter will sometimes have zero
    counts, this function adds 10 to all counts as a quick hack
    to circumvent this (the scipy chisquare will return nan if
    it encounters zero counts)

    Parameters
    ----------
    X: array
        Array of counts (histogram) for empirical distribution X.

    Y: array
        Array of counts (histogram) for empirical distribution Y.

    Returns
    -------
    p: float
        The p value of the chi square comparison.
    '''
    adjusted_X = X + offset
    adjusted_Y = Y + offset
    return chisquare(adjusted_X, adjusted_Y, *args, **kwargs)[1]

# @njit(parallel=True, fastmath=True)
@njit(fastmath=True)
def lightcone_size(depth, c):
    '''
    Computes lenght of random vector representation of light cone configuration.
    Technically this is the past lightcone depth, as it includes the present site.
    Subtract 1 from this for future lightcone sides.

    Parameters
    ----------
    depth: int
        Depth of the light cone under consideration. Depth = 0 is just the current
        site. Depth = 1 includes one time step into the past, etc.

    c: int
        Speed of information propagation in the spacetime system.

    Returns
    -------
    size: int
        Length of light cone configuration vector.
    '''
    size = 0
    for d in range(depth+1):
        size += 2*c*d + 1
    return size

@njit
def extract_lightcones(padded_data, T, X, past_depth, future_depth, c, base_anchor):
    '''
    Returns arrays of past and future lightcones extracted from the given data.
    If the original data has periodic boundary conditions, it must be pre-padded
    before being given to this function.

    ** I am not very proficient with numba, so I'm sure there is much cleaner way
    to do this**

    Parameters
    ----------
    padded_data: ndarray
        2D Spacetime array of target data from which lightcones are to be extracted.
        Time should be the 0th axis (vertical) and space the 1st axis (horizontal).
        If the original spacetime data has periodic boundary conditions, it should
        be pre-padded accordingly.

    T: int
        Size of the temporal dimension of the original (unpadded) spacetime field.

    X: int
        Size of the spatial dimension of the original (unpadded) spacetime field.

    past_depth: int
        Depth of the past lightcones to be extracted.

    future_depth: int
        Depth of the future lightcones to be extracted.

    past_size: int
        Size of the flattened past lightcone arrays.

    future_size: int
        Size of the flattened future lightcone arrays.

    c: int
        Propagation speed of the spacetime field.

    base_anchor: (int, int)
        Spacetime indices that act as reference point for indices that are "moved"
        throughout the spacetime field to extract lightcones at those points.
        Should start in the top left of the spacetime field, accounting for the
        margin and periodic boundary condition padding.

    Returns
    -------
    lightcones: (array, array)
        Returns tuple of arrays, (past_lightcones, future_lightcones), extracted
        from the spacetime field.
    '''
    dtype = padded_data.dtype
    past_size = lightcone_size(past_depth, c)
    future_size = lightcone_size(future_depth, c) - 1
    plcs = np.zeros((T*X, past_size), dtype=dtype)
    flcs = np.zeros((T*X, future_size), dtype=dtype)
    base_t, base_x = base_anchor # reference point for spacetime indices

    i = 0
    for t in range(T):
        for x in range(X):
            # loops for past lightcone
            p = 0
            for d in range(past_depth + 1):
                window_size = 2*d*c + 1
                for w in range(window_size):
                    a = -d*c + w
                    plcs[i,p] = padded_data[base_t+t-d, base_x+x+a]
                    p += 1

            # loops for future lightcone
            f = 0
            for depth in range(future_depth):
                d = depth + 1
                window_size = 2*d*c + 1
                for w in range(window_size):
                    a = -d*c + w
                    flcs[i,f] = padded_data[base_t+t+d, base_x+x+a]
                    f += 1
            i += 1

    return (plcs, flcs)

# @njit(parallel=True, fastmath=True)
@njit(fastmath=True)
def lightcone_size_2D(depth, c):
    '''
    Computes lenght of random vector representation of light cone configuration.
    Technically this is the past lightcone depth, as it includes the present site.
    Subtract 1 from this for future lightcone sides.

    Parameters
    ----------
    depth: int
        Depth of the light cone under consideration. Depth = 1 is just the current
        site. Depth = 2 includes one time step into the past, etc.

    c: int
        Speed of information propagation in the spacetime system.

    Returns
    -------
    size: int
        Length of light cone configuration vector.
    '''
    size = 0
    for d in range(depth+1):
        size += (2*c*d + 1)**2
    return size

@njit
def extract_lightcones_2D(padded_data, T, Y, X, past_depth, future_depth, c, base_anchor):
    '''
    Returns arrays of past and future lightcones extracted from the given data.
    If the original data has periodic boundary conditions, it must be pre-padded
    before being given to this function.


    Parameters
    ----------
    padded_data: ndarray
        3D Spacetime array of target data from which lightcones are to be extracted.
        Time should be the 0th axis (vertical) and space Y and X on the following axes.
        If the original spacetime data has periodic boundary conditions, it should
        be pre-padded accordingly.

    T: int
        Size of the temporal dimension of the original (unpadded) spacetime field, minus the margin.

    Y: int
        Size of the vertical spatial dimension of the original (upadded) spacetime field, minus the margin.

    X: int
        Size of the horizontal spatial dimension of the original (unpadded) spacetime field, minus the margin.

    past_depth: int
        Depth of the past lightcones to be extracted.

    future_depth: int
        Depth of the future lightcones to be extracted.

    past_size: int
        Size of the flattened past lightcone arrays.

    future_size: int
        Size of the flattened future lightcone arrays.

    c: int
        Propagation speed of the spacetime field.

    base_anchor: (int, int)
        Spacetime indices that act as reference point for indices that are "moved"
        throughout the spacetime field to extract lightcones at those points.
        Should start in the top left of the spacetime field, accounting for the
        margin and periodic boundary condition padding.

    Returns
    -------
    lightcones: (array, array)
        Returns tuple of arrays, (past_lightcones, future_lightcones), extracted
        from the spacetime field.
    '''
    dtype = padded_data.dtype
    past_size = lightcone_size_2D(past_depth, c)
    future_size = lightcone_size_2D(future_depth, c) - 1
    plcs = np.zeros((T*Y*X, past_size), dtype=dtype)
    flcs = np.zeros((T*Y*X, future_size), dtype=dtype)
    base_t, base_y, base_x = base_anchor # reference starting point for spacetime indices

    i = 0
    for t in range(T):
        for y in range(Y):
            for x in range(X):
                # loops for past lightcone
                p = 0
                for d in range(past_depth + 1):
                    span = np.arange(-d*c, d*c + 1)
                    for a in span:
                        for b in span:
                            plcs[i,p] = padded_data[base_t+t-d, base_y+y+a, base_x+x+b]
                            p += 1

                # loops for future lightcone
                f = 0
                for depth in range(future_depth):
                    d = depth + 1
                    span = np.arange(-d*c, d*c + 1)
                    for a in span:
                        for b in span:
                            flcs[i,f] = padded_data[base_t+t+d, base_y+y+a, base_x+x+b]
                            f += 1
                i += 1

    return (plcs, flcs)

#OPT: Consider specializing for plc and flc? Some of this seems extraneous work
#OPT: numba jitting post specialization
def lightcone_decay(depth, c, decay_rate, future_lightcones=False):
    '''

     ***For 1+1 spacetime data***

    Returns an array exponential temporal decays for a given 1+1 D lightcone shape.
    This may be mulitplied to a lightcone array (or ndarray vertical stack of multiple
    lightcones of the same shape) to apply the temporal decay to the lightcone array(s).
    '''
    size = lightcone_size(depth, c)
    n_counters = depth + 1
    depth_incrimenter = 0

    if future_lightcones:
        size -= 1
        n_counters -= 1
        depth_incrimenter += 1
    decays = np.ones(size)
    counters = np.empty(n_counters, dtype=int)

    for d in range(n_counters):
        D = d # little trick to handle both past and future lightcones with same function
        if future_lightcones:
            D += 1
        counters[d] = 2*c*D + 1 # number of l.c. elements at each depth

    # use cumsum of counters to create array of lightcone depth as function of array index
    index_start = 0
    for depth_change in np.cumsum(counters):
        index_end = depth_change
        decays[index_start: index_end] = -1 * decay_rate * depth_incrimenter
        depth_incrimenter += 1
        index_start = depth_change

    return np.exp(decays)

#OPT: Consider specializing for plc and flc? Some of this seems extraneous work
#OPT: numba jitting post specialization
def lightcone_decay_2D(depth, c, decay_rate, future_lightcones=False):
    '''
    Returns an array of exponential temporal decays for a given 2+1 D lightcone shape.
    This is meant to be multiplied to a lightcone array (or ndarray vertical stack of multiple
    lightcones of the same shape) to apply the temporal decay to the lightcone array(s).
    '''
    size = lightcone_size_2D(depth, c)
    n_counters = depth + 1
    depth_incrimenter = 0

    if future_lightcones:
        size -= 1
        n_counters -= 1
        depth_incrimenter += 1
    decays = np.ones(size)
    counters = np.empty(n_counters, dtype=int)

    for d in range(n_counters):
        D = d # trick to handle both past and future lightcones with this function
        if future_lightcones:
            D += 1
        counters[d] = (2*c*D + 1)**2

    # use cumsum of counters to create array of lightcone depth as function of array index
    index_start = 0
    for depth_change in np.cumsum(counters):
        index_end = depth_change
        decays[index_start: index_end] = -1 * decay_rate * depth_incrimenter
        depth_incrimenter += 1
        index_start = depth_change

    return np.exp(decays)


class LocalCausalStates(object):
    '''

    *** NEED TO UPDATE***

    Object that handles the reconstruction of the local causal states for a given
    data set, and the corresponding analysis.

    The workflow is as follows:

    - Initialize a Reconstructor instance with the main inference parameters

    - Run Reconstructor.extract() on the target spacetime field. This is the spacetime
      field that the coherent structure analysis will be performed on.

    - If necessary (will be necessary in all but the most trivial cases), scan
      more spacetime data with Reconstructor.extract_more(). This additional spacetime
      data should be commensurate with the target field -- typically the same equations
      of motion run with the same parameters but different initial conditions.

    - Once all the desired data has been scanned, run Reconstructor.reconstruct_morphs().
      This performs the first clustering step with the given clustering algorithm
      (clusters past and future lightcones, the resulting clusters are referred to as
      "pasts" and "futures", respectively), and uses the resulting clusterings to
      create an empirical joint distriubtion over pasts and futures.

    - Perform the second clustering step using Reconstructor.reconstruct_states().
      This clusters together pasts that have the ~the same~ (close, according
      to the given distribution metric / comparison) distribution over futures.
      The resulting clusters are the local causal states. The list of states is given
      in Reconstructor.states. NAN states are designated with the label "0" and show up
      in the "margin" - points in spacetime that do not have a past or future lightcone,
      as well as for pasts assigned to a noise cluster.
      The epsilon mapping (Reconstructor.epsilon_map) from pasts to their associated
      local causal states is also created here, and used in the next step for
      causal filtering.

    - Perform causal filtering on the original target field using
      Reconstructor.causal_filter(). This creates the associated local causal state
      field, Reconstructor.state_field.

    - If desired, a first pass at a coherent structure analysis can run using
      Reconstructor.complexity_field(), which returns the local statistical complexity
      field of the target field. This is the point-wise entropy of the local causal states.
      The idea is that the field will be mostly background -- the corresponding states then
      have a low entropy value.
    '''

    def __init__(self, past_depth, future_depth, propagation_speed):
        '''
        Initialize Reconstructor instance with main inference parameters.
        These define the shape of the lightcone template.
        Lightcone depths are hyperparameters for the reconstruction. The
        propagation speed is either set by the system, or chosen as an inference
        parameter if not known.

        Parameters
        ----------
        past_depth: int
            Depth of the past lightcones.

        future_depth: int
            Depth of the past lightcones.

        propagation_speed: int
            Finite speed of interaction / perturbation propagation used for inference.
            Either explicitly specified by the system (like with cellular automata) or
            chosen as an inference parameter to capture specific physics (e.g. chosing
            advection scale rather than accoustic scale for climate).
        '''
        # inference params
        self.past_depth = past_depth
        self.future_depth = future_depth
        self.c = propagation_speed

        # for causal clustering and filtering
        self.states = []
        self.epsilon_map = {}
        self._state_index = 1

        # for lightcone extraction
        max_depth = max(self.past_depth, self.future_depth)
        self._padding = max_depth*self.c

        # initialize some attributes to None for pipeline fidelity
        self.plcs = None
        self.joint_dist = None
        self._adjusted_shape = None
        self.state_field = None

    def infer(self, field, past_params, future_params,
                            past_decay=0, future_decay=0,
                            past_init_params=None, future_init_params=None,
                            metric=chi_squared, pval_threshold=0.05, metric_kwargs=None,
                            boundary_condition='open', distributed=False):
        '''
        

        '''
        self.extract(field, boundary_condition, distributed)
        self.kmeans_lightcones(past_params, future_params, past_decay, future_decay,
                               past_init_params, future_init_params)
        self.approximate_morphs()
        self.approximate_states(metric, pval_threshold, metric_kwargs)
        self.causal_filter()
        
        

    def _extract_1D(self, field, boundary_condition, shape):
        '''
        Backend method used by .extract() and .extract_more() to extract lightcone
        vectors from 1+1 D spacetime fields.
        '''
        T, X = shape
        adjusted_T = T - self.past_depth - self.future_depth # always cut out time margin
        if boundary_condition == 'open':
            adjusted_X = X - 2*self._padding # also have spatial margin for open boundaries
            padded_field = field
        elif boundary_condition == 'periodic':
            adjusted_X = X # no spatial margin for periodic boundaries
            padded_field = np.pad(field, ((0,0), (self._padding, self._padding)), 'wrap')
        adjusted_shape = (adjusted_T, adjusted_X)
        if self._adjusted_shape is None: # save adjusted shape of target field only
            self._adjusted_shape = adjusted_shape

        plcs, flcs = extract_lightcones(padded_field, *adjusted_shape,
                                            self.past_depth,
                                            self.future_depth,
                                            self.c,
                                            self._base_anchor)
        return plcs, flcs

    def _extract_2D(self, field, boundary_condition, shape):
        '''
        Backend method used by .extract() and .extract_more() to extract lightcone
        vectors from 2+1 D spacetime fields.
        '''
        T, Y, X = np.shape(field)
        adjusted_T = T - self.past_depth - self.future_depth # always cut out time margin
        if boundary_condition == 'open':
            adjusted_Y = Y - 2*self._padding # also have spatial margin for open boundaries
            adjusted_X = X - 2*self._padding
            padded_field = field
        elif boundary_condition == 'periodic':
            adjusted_Y = Y # no spatial margin for periodic boundaries
            adjusted_X = X
            padded_field = np.pad(field,
                                  (
                                      (0,0),
                                      (self._padding, self._padding),
                                      (self._padding, self._padding)
                                  ),
                                  'wrap')
        adjusted_shape = (adjusted_T, adjusted_Y, adjusted_X)
        if self._adjusted_shape is None: # save adjusted shape of target field only
            self._adjusted_shape = adjusted_shape

        plcs, flcs = extract_lightcones_2D(padded_field, *adjusted_shape,
                                                    self.past_depth,
                                                    self.future_depth,
                                                    self.c,
                                                    self._base_anchor)
        return plcs, flcs

    def extract(self, field, boundary_condition='open', distributed=False):
        '''
        Scans target field that is to be filtered after local causal state reconstruction.
        This is the first method that should be run.

        Parameters
        ----------
        field: ndarray
            2D or 3D array of the target spacetime field. In both cases time should
            be the zero axis.

        boundary_condition: str, optional (default='open')
            Set according to boundary conditions of the target field. Can only be
            either 'open' or 'periodic'. Open leaves a spatial margin where lightcones
            are not collected. Periodic gathers lightcones across the whole spatial
            lattice. Any additional training fields scanned with the .extract_more()
            method will be treated with same boundary conditions specified here.
        '''
        if boundary_condition not in ['open', 'periodic']:
            raise ValueError('boundary_condition must be "open" or "periodic"')
        self._bc = boundary_condition
        self._distributed = distributed

        shape = np.shape(field)

        if len(shape) == 2:
            self._base_anchor = (self.past_depth, self._padding)
            self.plcs, self.flcs = self._extract_1D(field, self._bc, shape)
        elif len(shape) == 3:
            self._base_anchor = (self.past_depth, self._padding, self._padding)
            self.plcs, self.flcs = self._extract_2D(field, self._bc, shape)
        else:
            raise ValueError("Input field must have 2 or 3 dimensions")


    def kmeans_lightcones(self, past_params, future_params,
                            past_decay=0, future_decay=0,
                            past_init_params=None, future_init_params=None):
        '''
        Performs clustering on the global arrays of both past and future lightcones.


        Parameters
        ----------
        past_params: dict,
            Dictionary of keword arguments for past lightcone clustering algorithm.

            If past_cluster == 'kmeans':
                past_params must include values for 'nClusters' and 'maxIterations'

        future_params: dict,
            Dictionary of keword arguments for future lightcone clustering algorithm.

            If future_cluster == 'kmeans':
                future_params must include values for 'nClusters' and 'maxIterations'

        past_decay: int, optional (default=0)
            Exponential decay rate for lightcone distance used for past lightcone clustering.

        future_decay: int, optional (default=0)
            Exponential decay rate for lightcone distance used for future lightcone clustering.
        '''
        if self.plcs is None:
            raise RuntimeError("Must call .extract() on a training field(s) before calling .cluster_lightcones().")
            
        if len(self._adjusted_shape) == 2:
            past_decays = lightcone_decay(self.past_depth, self.c, past_decay, False)
            future_decays = lightcone_decay(self.future_depth, self.c, future_decay, True)
        elif len(self._adjusted_shape) == 3:
            past_decays = lightcone_decay_2D(self.past_depth, self.c, past_decay, False)
            future_decays = lightcone_decay_2D(self.future_depth, self.c, future_decay, True)
        
        self.plcs *= np.sqrt(past_decays)
        self.flcs *= np.sqrt(future_decays)

        # Primarily used for global joint dist in distributed mode
        self._N_pasts = past_params['nClusters']
        self._N_futures = future_params['nClusters']

        if past_init_params is None:
            #method = 'randomDense'
            #method = 'parallelPlusDense'
            method = 'plusPlusDense'
            #method = 'defaultDense'
            past_init_params = {'nClusters':self._N_pasts,
                                   'method': method,
                                   'distributed': self._distributed}
        initial = d4p.kmeans_init(**past_init_params)
        centroids = initial.compute(self.plcs).centroids
        past_cluster = d4p.kmeans(distributed=self._distributed, **past_params).compute(self.plcs, centroids)
        past_local = d4p.kmeans(nClusters=self._N_pasts, distributed=self._distributed, assignFlag=True, maxIterations=0).compute(self.plcs, past_cluster.centroids)
        self.pasts = past_local.assignments.flatten()

        del past_cluster
        del self.plcs

        if future_init_params is None:
            #method = 'randomDense'
            #method = 'parallelPlusDense'
            method = 'plusPlusDense'
            #method = 'defaultDense'
            future_init_params = {'nClusters':self._N_futures,
                                   'method': method,
                                   'distributed': self._distributed}
        initial = d4p.kmeans_init(**future_init_params)
        centroids = initial.compute(self.flcs).centroids
        future_cluster = d4p.kmeans(distributed=self._distributed, **future_params).compute(self.flcs, centroids)
        self._future_centroids = future_cluster.centroids # save for field reconstruction
        future_local = d4p.kmeans(nClusters=self._N_futures, distributed=self._distributed, assignFlag=True, maxIterations=0).compute(self.flcs, self._future_centroids)
        self.futures = future_local.assignments.flatten()

        del future_cluster
        del self.flcs

    def approximate_morphs(self):
        '''
        Counts lightcone cluster labels to build empirical joint distribution.
        '''
        # OPT: comment out for performance runs
        if self.pasts is None:
            raise RuntimeError("Must call .kmeans_lightcones() before calling .approximate_morphs()")
        # morphs accessed through this joint distribution over pasts and futures
        self.joint_dist = dist_from_data(self.pasts, self.futures, self._N_pasts, self._N_futures)
        # *** distributed mode not implemented yet***
        # if self._distributed == True:
        #     self.global_joint_dist = np.zeros((self._N_pasts, self._N_futures), dtype=np.uint64)

        del self.futures


    def approximate_states(self, metric, *metric_args, pval_threshold=0.05, **metric_kwargs):
        '''
        Hierarchical agglomerative clustering of lightcone morphs
        from a given joint distribution array (joint over lightcone clusters),
        where the first column is the labels for the plc clusters (pasts).
        This is needed because a random permutation should be done before
        clustering.

        Any noise clusters are ignored. If there is a noise cluster for pasts,
        it is assigned to the NAN state. If there is a noise cluster for futures,
        the counts of this cluster are removed from the morphs of each past.

        NOTE -- because I'm using dist metrics from scipy, I'm using p values to
        decide whether two distributions are identical, not strictly using a
        minimum distribution distance threshold (p value comparison is opposite
        of minimum distance comparison)

        Parameters
        ----------
        metric: function
            Python function that does a stastical comparison of two empirical distributions.
            In the current use, this function is expected to return a p value for this
            comparison.

        pval_threshold: float, optional (default=0.05)
            p value threshold for the distribution comparison. If the comparison p
            value is greater than pval_threshold, the two distributions are considered
            equivalent.

        '''
        if self.joint_dist is None:
            raise RuntimeError("Must call .reconstruct_morphs() first.")

        rlabels = np.arange(0, self._N_pasts, dtype=np.uint64)[np.newaxis]
        if self._distributed == True:
            # For now, no permutation in distributed mode
            morphs = np.hstack((rlabels.T, self.global_joint_dist))
        else:
            dist = np.hstack((rlabels.T, self.joint_dist))
            morphs = np.random.permutation(dist)

        self._label_map = np.zeros(self._N_pasts, dtype=int) # for vectorized causal_filter

        # hierarchical agglomerative clustering -- clusters pasts into local causal states
        for item in morphs:
            past = item[0]
            morph = item[1:]
            for state in self.states:
                p_value = metric(morph, state.morph, *metric_args, **metric_kwargs)
                if p_value > pval_threshold:
                    state.update(past, morph)
                    self.epsilon_map.update({past : state})
                    self._label_map[past] = state.index
                    break

            else:
                new_state = CausalState(self._state_index, past, morph)
                self.states.append(new_state)
                self._state_index += 1
                self.epsilon_map.update({past : new_state})
                self._label_map[past] = new_state.index

        del self.joint_dist


    def causal_filter(self):
        '''
        Performs causal filtering on target field (input for Reconstructor.extract())
        and creats associated local causal state field (Reconstructor.state_field)

        The margins, spacetime points that don't have a full past or future lightcone,
        are assigned the NAN state with integer label 0.

        *** Should make a state_label_map attribute that is a numpy array
            where state_label_map[i] maps past_i to its local causal state integer label.
            Then can use this as a mask for vectorized filtering, instead of ndenumerate.***
        '''
        if len(self.states) == 0:
            raise RuntimeError("Must call .reconstruct_states() first.")

        past_field = self.pasts.reshape(*self._adjusted_shape)
        self.state_field = np.zeros(self._adjusted_shape, dtype=int)

        # use label_map to map past_field to field of local causal state labels
        self.state_field = self._label_map[past_field]

        # Go back and re-pad state field with margin so it is the same shape as the original data
        if self._bc == 'open':
            spatial_pad = self._padding
        elif self._bc == 'periodic':
            spatial_pad = 0

        if len(self._adjusted_shape) == 2:
            margin_padding = ((self.past_depth, self.future_depth), (spatial_pad, spatial_pad))
        elif len(self._adjusted_shape) == 3:
            margin_padding = (
                                (self.past_depth, self.future_depth),
                                (spatial_pad, spatial_pad),
                                (spatial_pad, spatial_pad)
                            )

        self.state_field = np.pad(self.state_field, margin_padding, 'constant')


    def reconstruct_field(self):
        '''
        Uses reconstructed epsilon map and its inverse to reconstructed the
        target spacetime field given as input to Reconstructor.extract().

        The inverse epsilon map works as follows. For each local causal state in
        Reconstructor.state_field a future (cluster of real-valued future lightcones)
        is randomly chosen according to the morph (distribution over futures given state)
        of that state. Since each future is a cluster over real-valued future lightcones
        clustered by KMeans, the mean of the cluster is chosen as the representative real-valued
        future lightcone for that future and is placed in the reconstructed spacetime field. Since
        multiple predictions will be made for each point in spacetime (i.e. from overlapping
        predicted future lightcones), we average over these multiple predictions.
        '''
        assert self.state_field is not None, "Must call .causal_filter() first."
        assert self._bc == 'periodic', "Currently only works for periodic boudnary conditions."

        if self._bc == 'open':
            spatial_pad = self.future_depth*self.c
        elif self._bc == 'periodic':
            spatial_pad = 0

        if (self.past_depth > self.future_depth) and (self._bc == 'open'):
            spatial_margin = (self.past_depth - self.future_depth)*self.c
        else:
            spatial_margin = 0

        if len(self._adjusted_shape) == 2: # 1+1 D
            self._reconstruct_field_1D(spatial_pad)
            margin_padding = ((0, 0), (spatial_margin, spatial_margin))
        elif len(self._adjusted_shape) == 3: # 2+1 D
            self._reconstruct_field_2D(spatial_pad)
            margin_padding = (
                                (self.past_depth, 0),
                                (spatial_margin, spatial_margin),
                                (spatial_margin, spatial_margin)
                             )

        self.reconstructed_field = np.pad(self.reconstructed_field, margin_padding, 'constant')

    def _reconstruct_field_1D(self, spatial_pad):
        '''
        Support code for reconstructing 1+1 D fields.

        *** Use self.state_field directly, not self._adjusted_shape, so that this can work with or without running forecast first. Use try-except (except ValueError: pass) to make predictions all the way to the final time horizon, and so don't cut out future_depth from the end (may have to take one time step off the end of the state_field). ***

        *** Need to account for open boundary conditions***

        *** Try adding the temporal decay to predicted flcs, like the old "forecast" method***
        '''
        adjusted_T, adjusted_X = self._adjusted_shape
        reconst_shape = np.shape(self.state_field)
        self.reconstructed_field = np.zeros(reconst_shape)
        adjusted_state_field = self.state_field[self.past_depth:-1]

        # search through causal state field and sample future lightcones from each state's morph
        for location, state_index in np.ndenumerate(adjusted_state_field):
            t, x = location
            t += self.past_depth
            x += spatial_pad

            state = self.states[state_index - 1] # index 0 reserved for NAN state
            morph = state.normalized_morph()
            future = np.random.choice(np.arange(0, len(morph)), p=morph)
            predicted_flc = self._future_centroids[future]

            # place future lightcone values in self.reconstructed_field at location
            f = 0
            for depth in range(self.future_depth):
                d = depth + 1
                span = np.arange(-d*self.c, d*self.c + 1)
                for a in span:
                    try:
                        self.reconstructed_field[t+d,
                                    (x+a)%(adjusted_X + 2*spatial_pad)] += predicted_flc[f]
                    except IndexError:
                        pass

                    f += 1

        self.reconstructed_field /= (lightcone_size(self.future_depth, self.c)-1) # for averaging



    def _estimate_state_dynamic(self, boundary_condition='periodic'):
        '''
        Builds the estimate stochast cellular automata dynamic over local causal states from the
        local causal state field.

        Because the current convention is to give the NAN state an index label of 0, the indices for the actual states start at 1. The state dynamic built here does not account for the NAN state (i.e. assumes it will never see the NAN state nor transition to it), so for simplicity of implementation the actual state indices are all decremented by one, so that they start at 0.

        *** Currently for 1+1 D only ***
        '''
        assert boundary_condition == 'periodic', "Currently only works for periodic boundary conditions."
        assert self.state_field is not None, "Must call .causal_filter() first."

        N_states = len(self.states)
        neighb_size = 2*self.c + 1
        shape = tuple(N_states * np.ones(neighb_size+1, dtype=int))
        self.state_dynamic = np.zeros(shape)

        # remove time margin (currently only for periodic b.c. so no space margin to remove)
#         adjusted_T, adjusted_X = self._adjusted_shape
        adjusted_state_field = self.state_field[self.past_depth:-self.future_depth] 

        for t in range(adjusted_T - 1):
            neighborhoods = zip(*neighborhood(adjusted_state_field[t], self.c))
            states = adjusted_state_field[t+1]
            for neighb, state in zip(neighborhoods, states):
                self.state_dynamic[neighb][state] += 1 # remember state here is already adjusted to index



    def forecast(self, time, boundary_condition='periodic'):
        '''
        First estimates stochastic cellular automata dynamic over local causal states
        from the local causal state field, then uses the estimated dynamic to forecast the
        state field forward in time for the given amount of time.

        Remember that there are no states at the last T time steps, where T is the future lightcone depth, and so those times will be forecasted first, then the additional 'time' number of time steps will be forecasted.

        *** Can try decrementing state field here first, before the ._estimate_state_dynamic() call ***

        *** Should figure out a way to normalize the transition distributions just one time, instead of doing it every time ***

        *** Currently for 1+1 D only ***
        '''
        assert boundary_condition == 'periodic', "Currently only works for periodic boundary conditions."
        assert self.state_field is not None, "Must call .causal_filter() first."

        self._estimate_state_dynamic(boundary_condition)

        self.state_field -= 1
        field_addition = np.zeros((time, self._adjusted_shape[1]), dtype=int) - 1
        self.state_field = np.concatenate((self.state_field, field_addition))
        state_labels = np.arange(len(self.states))
        self.unforseen_neighborhoods = 0

        T = time + self.future_depth
        anchor_T = self._adjusted_shape[0]+self.future_depth
        # current_adjusted_states = self.state_field[-(self.future_depth+1)]

        for t in range(T):
            current_adjusted_states = self.state_field[anchor_T+t]
            neighborhoods = zip(*neighborhood(current_adjusted_states, self.c))
            for i, n in enumerate(neighborhoods):
                transition = self.state_dynamic[n]
                Z = np.sum(transition)
                if Z == 0:
                    # raise RuntimeError("The neighborhood {} was not seen when estimating the state dynamic".format(n))
                    self.unforseen_neighborhoods += 1
                    forecasted_state = np.random.choice(state_labels) # !!! update to use previous time spatial distribution instead
                else:
                    transition_probs = transition / Z # OPT: should not normalize every time; probably do in "estimate" method
                    forecasted_state = np.random.choice(state_labels, p=transition_probs)
                self.state_field[anchor_T+t+1, i] = forecasted_state
        self.state_field += 1
