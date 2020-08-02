import numpy as np
import daal4py as d4p
import types

from numba import njit
from scipy.stats import chisquare
from collections import Counter
from itertools import product

from spacetime.analytics.local_causal_states import *
from spacetime.simulators.CAs import neighborhood

class DevReconstructor(Reconstructor):
    '''
    Dev class for prototyping forecasting and decoding
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

        self._dl = None # description length (from MDL)

        # for lightcone extraction
        max_depth = max(self.past_depth, self.future_depth)
        self._padding = max_depth*self.c

        # initialize some attributes to None for pipeline fidelity
        self.plcs = None
        self.joint_dist = None
        self._adjusted_shape = None
        self.state_field = None

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

    def extract(self, field, boundary_condition='open'):
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

        shape = np.shape(field)

        if len(shape) == 2:
            self._base_anchor = (self.past_depth, self._padding)
            self.plcs, self.flcs = self._extract_1D(field, self._bc, shape)
        elif len(shape) == 3:
            self._base_anchor = (self.past_depth, self._padding, self._padding)
            self.plcs, self.flcs = self._extract_2D(field, self._bc, shape)
        else:
            raise ValueError("Input field must have 2 or 3 dimensions")

    # def extract_more(self, input_fields):
    #     '''
    #     Scans extra fields (new instances of the target field) to acquire
    #     better statistics for inference.
    #
    #     Parameters
    #     ----------
    #     input_fields: ndarrays
    #         Ensemble of additional spacetime field arrays (given in a tuple,
    #         list, generator, or as a single additional field) to be scanned.
    #         Should be commensurate with the target field -- typically the same
    #         equations of motion with the same parameters, but different initial
    #         conditions.
    #     '''
    #     if self.plcs is None:
    #         raise RuntimeError("Must call .extract() on a target field before calling .extract_more().")
    #
    #     # Check if input is a single field, if so put it in a tuple
    #     if not isinstance(input_fields, (tuple, list, types.GeneratorType)):
    #         field_set = (input_fields,)
    #     else:
    #         field_set = input_fields
    #
    #     for field in field_set:
    #         shape = np.shape(field)
    #         if len(shape) != len(self._adjusted_shape):
    #             raise ValueError('''input fields must be same dimension as the target
    #             field scaned with .extract() ''')
    #         elif len(shape) == 2:
    #             plcs, flcs = self._extract_1D(field, self._bc, shape)
    #         elif len(shape) == 3:
    #             plcs, flcs = self._extract_2D(field, self._bc, shape)
    #
    #         # add extracted lightcone arrays to master lightcone arrays
    #         self.plcs = np.vstack((self.plcs, plcs))
    #         self.flcs = np.vstack((self.flcs, flcs))

    def reconstruct_morphs(self, past_cluster, future_cluster, past_decay=0, future_decay=0):
        '''
        Performs clustering on the master arrays of both past and future lightcones.

        Expects clustering algorithm to give integer cluster labels start at 0,
        with the "noise cluster" having label -1.

        Diagnostics of this clustering (what are the unique clusters and how many
        lightcones were assigned to each cluster) accessed through namedtuple
        Reconstructor.lc_cluster_diagnostic.

        Parameters
        ----------
        past_cluster: sklearn.cluster object
            Scikit-learn class of the desired clustering algorithm for past lightcones.
            cluster.fit_predict() is run.

        future_cluster: sklearn.cluster class
            Scikit-learn class of the desired clustering algorithm for past lightcones.
            cluster.fit_predict() is run.

        past_decay: int, optional (default=0)
            Exponential decay rate for lightcone distance used for past lightcone clustering.

        future_decay: int, optional (default=0)
            Exponential decay rate for lightcone distance used for future lightcone clustering.

        past_params: dict, optional (default={})
            Dictionary of keyword arguments for past lightcone clustering function.

        future_params: dict, optional (default={})
            Dictionary of keyword arguments for future lightcone clustering function.
        '''
        if self.plcs is None:
            raise RuntimeError("Must call .extract() on a training field(s) before calling .reconstruct_morphs().")

        if len(self._adjusted_shape) == 2: # 1+1 D data
            dim = 1
        elif len(self._adjusted_shape) == 3: # 2+1 D data
            dim = 2

        # save for later use in inference
        self.past_cluster = past_cluster
        self.future_cluster = future_cluster
        self.past_decay = past_decay
        self.future_decay = future_decay

        # perform lightcone clusterings and save diagnostics of the clusterings
        pasts = lightcone_cluster(self.plcs,
                                  self.past_cluster,
                                  self.past_depth,
                                  self.c,
                                  past_decay,
                                  dim)
        diagnostic_pasts, diagnostic_past_counts = np.unique(pasts, return_counts=True)
        del self.plcs
        gc.collect()

        futures = lightcone_cluster(self.flcs,
                                    future_cluster,
                                    self.future_depth,
                                    self.c,
                                    future_decay,
                                    dim,
                                    future_lightcones=True)
        diagnostic_futures, diagnostic_future_counts = np.unique(futures, return_counts=True)
        del self.flcs
        gc.collect()

        # handle "noise clusters" -- want to exclude them from causal clustering (reconstruct_states method)
        if diagnostic_pasts[0] == -1:
            self._past_noise = True
            pasts += 1
        else:
            self._past_noise = False

        if diagnostic_futures[0] == -1:
            self._skip_future_noise = 1
            futures += 1
        else:
            self._skip_future_noise = 0

        # store the pasts of the original target field (need for causal filtering)
        self.target_pasts = pasts[:np.product(self._adjusted_shape)]

        # morphs accessed through this joint distribution over pasts and futures
        self.joint_dist = dist_from_data(pasts, futures, row_labels=True)

        del self.past_cluster.labels_
        del self.future_cluster.labels_
        gc.collect()

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

        past_field = self.target_pasts.reshape(*self._adjusted_shape)
        self.state_field = np.zeros(self._adjusted_shape, dtype=int)

        # Scan through the past field and use the epsilon map to fill out the state field
        for location, past in np.ndenumerate(past_field):
            state = self.epsilon_map[past]
            self.state_field[location] = state.index

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

    def description_length(self, normalized=False):
        '''
        Currently assumes ".extract_more()" has not been called, so all reconstructed
        states appear in self.state_field

        Returns: out, tuple
            A tuple of the two separate components of description length --
            (length of model, length of data given model)
            Full description length is the sum of these componenets.
        '''
        assert self.state_field is not None, "Must call .causal_filter() first."
        if self._dl is None:
            states, counts = np.unique(self.state_field, return_counts=True)
            hist = counts[1:] # don't count NAN state
            P = hist/np.sum(hist)
            len_model = -np.sum(P * np.ma.log(P)) # use masked array to avoid possible log(0)

            len_data_given_model = 0
            for i, prob in enumerate(P):
                state = self.states[i]
                norm_morph = state.normalized_morph()
                conditional_sum = np.sum(norm_morph * np.ma.log(norm_morph))
                len_data_given_model -= prob * conditional_sum

            self._dl = (len_model, len_data_given_model)
        if normalized:
            Z0 = np.log(len(self.states))
            Z1 = np.log(len(self.states[0].morph))
            return(self._dl[0]/Z0, self._dl[1]/Z1)
        else:
            return self._dl

    # def reconstruct_field(self):
    #     '''
    #     Uses reconstructed epsilon map and its inverse to reconstructed the
    #     target spacetime field given as input to Reconstructor.extract().
    #
    #     The inverse epsilon map works as follows. For each local causal state in
    #     Reconstructor.state_field a future (cluster of real-valued future lightcones)
    #     is randomly chosen according to the morph (distribution over futures given state)
    #     of that state. Since each future is a cluster over real-valued future lightcones
    #     clustered by KMeans, the mean of the cluster is chosen as the representative real-valued
    #     future lightcone for that future and is placed in the reconstructed spacetime field. Since
    #     multiple predictions will be made for each point in spacetime (i.e. from overlapping
    #     predicted future lightcones), we average over these multiple predictions.
    #     '''
    #     assert self.state_field is not None, "Must call .causal_filter() first."
    #     assert self._bc == 'periodic', "Currently only works for periodic boudnary conditions."
    #
    #     if self._bc == 'open':
    #         spatial_pad = self.future_depth*self.c
    #     elif self._bc == 'periodic':
    #         spatial_pad = 0
    #
    #     if (self.past_depth > self.future_depth) and (self._bc == 'open'):
    #         spatial_margin = (self.past_depth - self.future_depth)*self.c
    #     else:
    #         spatial_margin = 0
    #
    #     if len(self._adjusted_shape) == 2: # 1+1 D
    #         self._reconstruct_field_1D(spatial_pad)
    #         margin_padding = ((self.past_depth, 0), (spatial_margin, spatial_margin))
    #     elif len(self._adjusted_shape) == 3: # 2+1 D
    #         self._reconstruct_field_2D(spatial_pad)
    #         margin_padding = (
    #                             (self.past_depth, 0),
    #                             (spatial_margin, spatial_margin),
    #                             (spatial_margin, spatial_margin)
    #                          )
    #
    #     self.reconstructed_field = np.pad(self.reconstructed_field, margin_padding, 'constant')
    #
    # def _reconstruct_field_1D(self, spatial_pad):
    #     '''
    #     Support code for reconstructing 1+1 D fields.
    #
    #     *** Use self.state_field directly, not self._adjusted_shape, so that this can work with or without running forecast first. Use try-except (except ValueError: pass) to make predictions all the way to the final time horizon, and so don't cut out future_depth from the end (may have to take one time step off the end of the state_field). ***
    #
    #     *** Need to account for open boundary conditions***
    #
    #     *** Try adding the temporal decay to predicted flcs, like the old "forecast" method***
    #     '''
    #     adjusted_T, adjusted_X = self._adjusted_shape
    #     reconst_shape = (adjusted_T + self.future_depth, adjusted_X + 2*spatial_pad)
    #     self.reconstructed_field = np.zeros(reconst_shape)
    #     adjusted_state_field = self.state_field[self.past_depth:-self.future_depth]
    #
    #     # search through causal state field and sample future lightcones from each state's morph
    #     for location, state_index in np.ndenumerate(adjusted_state_field):
    #         t, x = location
    #         x += spatial_pad
    #
    #         state = self.states[state_index - 1]
    #         morph = state.normalized_morph()
    #         future = np.random.choice(np.arange(0, len(morph)), p=morph)
    #         predicted_flc = self.future_cluster.cluster_centers_[future]
    #
    #         # place future lightcone values in self.reconstructed_field at location
    #         f = 0
    #         for depth in range(self.future_depth):
    #             d = depth + 1
    #             span = np.arange(-d*self.c, d*self.c + 1)
    #             for a in span:
    #                 self.reconstructed_field[t+d,
    #                                 (x+a)%(adjusted_X + 2*spatial_pad)] += predicted_flc[f]
    #                 f += 1
    #
    #     self.reconstructed_field /= (lightcone_size(self.future_depth, self.c)-1) # for averaging


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

            state = self.states[state_index - 1]
            morph = state.normalized_morph()
            future = np.random.choice(np.arange(0, len(morph)), p=morph)
            predicted_flc = self.future_cluster.cluster_centers_[future]

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



    # def static_predict(self):
    #     '''
    #     Using the past lightcones from the target inference field (the field scanned
    #     with Reconstructor.extract()), makes predictions for all the corresponding
    #     future lightcones. Since this is done at every spacetime point (inside the
    #     margin), there will be overlapping predictions that are averaged to create a
    #     prediction at each point. Thus the prediction field will be the same shape
    #     as the given inference field, with the margins padded to 0.
    #
    #     *** Currently just works for KMeans***
    #
    #     *** Need to figure out how to handle NAN states for clustering methods with
    #     noise ***
    #
    #     ***Perhaps at some point will try adding some kind of kernel smoothing***
    #     '''
    #     if len(self.states) == 0:
    #         raise RuntimeError("Must call .reconstruct_states() first.")
    #
    #     if self._bc == 'open':
    #         spatial_pad = self.future_depth*self.c
    #     elif self._bc == 'periodic':
    #         spatial_pad = 0
    #
    #     if (self.past_depth > self.future_depth) and (self._bc == 'open'):
    #         spatial_margin = (self.past_depth - self.future_depth)*self.c
    #     else:
    #         spatial_margin = 0
    #
    #     if len(self._adjusted_shape) == 2: # 1+1 D
    #         self._static_predict_1D(spatial_pad)
    #         margin_padding = ((self.past_depth, 0), (spatial_margin, spatial_margin))
    #     elif len(self._adjusted_shape) == 3: # 2+1 D
    #         self._static_predict_2D(spatial_pad)
    #         margin_padding = (
    #                             (self.past_depth, 0),
    #                             (spatial_margin, spatial_margin),
    #                             (spatial_margin, spatial_margin)
    #                          )
    #
    #     self.prediction_field = np.pad(self.prediction_field, margin_padding, 'constant')
    #
    # def _static_predict_1D(self, spatial_pad):
    #     '''
    #     Using the past lightcones from the target inference field (the field scanned
    #     with Reconstructor.extract()), makes predictions for all the corresponding
    #     future lightcones. Since this is done at every spacetime point (inside the
    #     margin), there will be overlapping predictions that are averaged to create a
    #     prediction at each point. Thus the prediction field will be the same shape
    #     as the given inference field, with the margins padded to 0.
    #
    #     *** Currently just works for KMeans***
    #
    #     *** Need to figure out how to handle NAN states for clustering methods with
    #     noise ***
    #
    #     ***Perhaps at some point will try adding some kind of kernel smoothing***
    #     '''
    #     adjusted_T, adjusted_X = self._adjusted_shape
    #     predict_shape = (adjusted_T + self.future_depth, adjusted_X + 2*spatial_pad)
    #
    #     past_field = self.target_pasts.reshape(*self._adjusted_shape)
    #     self.prediction_field = np.zeros(predict_shape)
    #
    #     # Scan through the past field and use the epsilon map to fill out the prediction field
    #     for location, past in np.ndenumerate(past_field):
    #         t, x = location
    #         x += spatial_pad
    #
    #         # get predicted future lightcone for this location
    #         state = self.epsilon_map[past]
    #         morph = state.normalized_morph()
    #         future = np.random.choice(np.arange(0, len(morph)), p=morph)
    #         predicted_flc = self.future_cluster.cluster_centers_[future]
    #
    #         # place future lightcone values in self.prediction_field at location
    #         f = 0
    #         for depth in range(self.future_depth):
    #             d = depth + 1
    #             span = np.arange(-d*self.c, d*self.c + 1)
    #             for a in span:
    #                 self.prediction_field[t+d,
    #                                 (x+a)%(adjusted_X + 2*spatial_pad)] += predicted_flc[f]
    #                 f += 1
    #
    #     self.prediction_field /= (lightcone_size(self.future_depth, self.c)-1) # for averaging
    #
    # def _static_predict_2D(self, spatial_pad):
    #     '''
    #     Using the past lightcones from the target inference field (the field scanned
    #     with Reconstructor.extract()), makes predictions for all the corresponding
    #     future lightcones. Since this is done at every spacetime point (inside the
    #     margin), there will be overlapping predictions that are averaged to create a
    #     prediction at each point. Thus the prediction field will be the same shape
    #     as the given inference field, with the margins padded to 0.
    #
    #     ***Perhaps at some point will try adding some kind of kernel smoothing***
    #     '''
    #     adjusted_T, adjusted_Y, adjusted_X = self._adjusted_shape
    #     predict_shape = (adjusted_T + self.future_depth,
    #                      adjusted_Y + 2*spatial_pad,
    #                      adjusted_X + 2*spatial_pad)
    #
    #     past_field = self.target_pasts.reshape(*self._adjusted_shape)
    #     self.prediction_field = np.zeros(predict_shape)
    #
    #     # Scan through the past field and use the epsilon map to fill out the prediction field
    #     for location, past in np.ndenumerate(past_field):
    #         t, y, x = location
    #         y += spatial_pad
    #         x += spatial_pad
    #
    #         # get predicted future lightcone for this location
    #         state = self.epsilon_map[past]
    #         morph = state.normalized_morph()
    #         future = np.random.choice(np.arange(0, len(morph)), p=morph)
    #         predicted_flc = self.future_cluster.cluster_centers_[future] # assumes KMeans
    #
    #         # place future lightcone values in self.prediction_field at location
    #         f = 0
    #         for depth in range(future_depth):
    #             d = depth + 1
    #             span = np.arange(-d*c, d*c + 1)
    #             for a,b in product(span, span):
    #                 self.prediction_field[t+d,
    #                                       (y+a)%(adjusted_Y + 2*spatial_pad),
    #                                       (x+b)%(adjusted_X + 2*spatial_pad)
    #                                      ] += predicted_flc[f]
    #                 f += 1
    #
    #     self.prediction_field /= (lightcone_size_2D(self.future_depth, self.c)-1) # for averaging


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

        # remove margin with NAN state and decrement state field so labels adjusted to start at 0
        adjusted_T, adjusted_X = self._adjusted_shape
        adjusted_state_field = self.state_field[self.past_depth:-self.future_depth] - 1

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
                    forecasted_state = np.random.choice(state_labels)
                else:
                    transition_probs = transition / Z
                    forecasted_state = np.random.choice(state_labels, p=transition_probs)
                self.state_field[anchor_T+t+1, i] = forecasted_state
        self.state_field += 1
