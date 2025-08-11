import matplotlib.pyplot as plt
import seaborn as sns
import torch
#from scipy.stats._multivariate import invwishart
import numpy as np
import prody as pr
from copy import deepcopy
from IPython.display import display
from IPython.display import clear_output
#import torch.multiprocessing as mp
#import multiprocessing
import pandas as pd 
#from scipy.special import gamma
import os
import random 
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
import json

clear_output(wait=True)

plt.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (8.5, 8.5)
plt.rcParams['lines.markersize'] = 4
plt.rcParams['lines.linewidth'] = 1.5
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Computer Modern Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['legend.title_fontsize'] = 'medium'

class OPPI():
    def __init__(self, parsed_msa:pr.sequence.msa.MSA, del_t:float, steps_per_traj:int, n_seq_gen:int, N_training_samples:int, 
                 q_ind:np.array, ind_ref:int, random_msa_path=None, T=1, lamda=1, M=None, momentum_model='Normal', warm_up=1, 
                 type_output='flat', hmc_T=0, par=True, zero_sum_gauge=False, force_equi_part=False, kappa=10, 
                 num_uniform_random_sequences=int(10e6), deletions=False, constraint=False, constraint_index=None):
        super().__init__()
        self.msa = parsed_msa
        self.num_sequences = self.msa.numSequences()
        print('num sequences before cleaning = ', self.num_sequences)
        
        self.residue_in_msa = parsed_msa.numResidues()
        self.residue_in_msa_iterator = range(self.residue_in_msa-1)

        self.deletions = deletions
        
        if type(self.deletions)==bool:
            if self.deletions:
                self.alphabet = ['A' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I' ,'K' ,'L' ,'M' ,'N' ,'P', 'Q' ,'R' ,'S' ,'T' ,'V', 'W' ,'Y', '-']
            else:
                self.alphabet = ['A' ,'C' ,'D' ,'E' ,'F' ,'G' ,'H' ,'I' ,'K' ,'L' ,'M' ,'N' ,'P', 'Q' ,'R' ,'S' ,'T' ,'V', 'W' ,'Y']
        else:
            raise Exception('deletions parameter must be a boolean, keep default to sample sequences with no deletions')
                
        self.alphabet_bytes = np.array([residue.encode('utf-8') for residue in self.alphabet], dtype=np.bytes_)
        self.indices_1_21 = torch.tensor(torch.arange(1, len(self.alphabet_bytes)+1))
        self.alphabet_int = torch.tensor(torch.arange(0, len(self.alphabet_bytes)-1))
        self.len_alphabet_bytes = len(self.alphabet_bytes)
        #self.alphabet_int_cuda = self.alphabet_int.to('cuda')
        self.del_t = del_t 
        self.steps_per_traj = steps_per_traj
        self.steps_per_traj_iterator = range(self.steps_per_traj)
        self.n_seq_gen = n_seq_gen 
        self.N_training_samples = N_training_samples
        self.T = T 
        self.lamda = lamda
        self.M = M
        self.momentum_model = momentum_model
        self.warm_up = warm_up
        self.q_ind = q_ind 
        self.ind_ref = ind_ref 
        self.type_output = type_output
        self.aa_iterator = range(self.len_alphabet_bytes)
        self.aa_enumerator = enumerate(self.aa_iterator)
        self.len_q_ind = len(q_ind)
        self.mutants_array = np.empty((self.n_seq_gen, self.len_q_ind), dtype=np.bytes_)
        self.ref_seq = parsed_msa.getArray()[self.ind_ref]
        # Array used to store mutants using the reference as a template
        #self.mutants_sequences = np.reshape(np.repeat(self.ref_seq, self.n_seq_gen, axis=0), (self.n_seq_gen, self.residue_in_msa))
        self.mutants_sequences = np.tile(self.ref_seq, (self.n_seq_gen, 1))
        self.mutated_residue_index_mat = np.empty((self.n_seq_gen, self.len_q_ind))#, dtype=np.int)
        self.q_ind_iterator = range(self.len_q_ind)
        
        self.par = par
        self.dimesionality =  ((self.len_q_ind**2) * (self.len_alphabet_bytes**2)) + (self.len_alphabet_bytes*self.len_q_ind)
        #(self.len_alphabet_bytes*self.len_q_ind*(self.residue_in_msa-self.len_q_ind))
        print('Number of parameters in the model = ', self.dimesionality)
        
        self.v = torch.linspace(-100,100, self.dimesionality*10)
        self.zero_tensor = torch.zeros(1)
        self.zero_sum_gauge = zero_sum_gauge
        self.force_equi_part = force_equi_part
        self.inv_n_mutant = 1/self.n_seq_gen
        self.hmc_T = hmc_T
        self.kappa = kappa

        if random_msa_path:
            if type(random_msa_path)!=str:
                raise Exception('random_msa_path must be a string')
            else:
                try:
                    self.random_msa = pr.parseMSA(random_msa_path, aligned=True)
                except FileNotFoundError as e:
                    print('ERROR file not found, keep default random_msa_path to generate a random MSA')
                    raise e
        else: 
            try:
                self.num_uniform_random_sequences = num_uniform_random_sequences
                
                if self.num_uniform_random_sequences<0:
                    raise(Exception('num_uniform_random_sequences must be a positive integer'))    

                print(f'Generating {self.num_uniform_random_sequences} random sequences')
                os.system('mkdir random_mutants')
                random_msa_path = 'random_mutants/random_mutants.fasta'
                self.random_msa = self.random_mutant_generator(random_msa_path=random_msa_path, 
                                                               num_uniform_random_sequences=self.num_uniform_random_sequences)[0]
            except Exception as e:
                raise(Exception(e))
                                                           
        print('Random sample size = ', self.random_msa.numSequences())
        self.num_sequences_random = 50000
        print(f'Each CSG iteration will use {self.num_sequences_random} random sequences')

        self.uniform_rand_proba_indices = torch.tensor(self.inv_n_mutant).repeat(self.num_sequences_random)
        
        if self.par:
            if not torch.cuda.is_available():
                raise(Exception('NO CUDA DEVICE DETECTED, UNABLE TO DO GPU PROCESSING (To run sequential code : par=False. NOT RECOMMENDED)'))

        self.X = np.unique(np.where(self.msa.getArray()==b'X')[0])
        self.clean_parsed_msa_alternative_ind = list(range(self.num_sequences))
        clean_parsed_msa_alternative_ind = np.where(np.array([x if x not in self.X else None for x in self.clean_parsed_msa_alternative_ind])!=None)[0]
        self.msa = self.msa[clean_parsed_msa_alternative_ind,:]
        print('num sequences after cleaning = ', self.msa.numSequences())
        
        self.constraint = constraint
        self.constraint_index = constraint_index
        
        if self.constraint:
            if not self.constraint_index or (type(self.constraint_index)!=list and type(self.constraint_index)!=np.array):
                raise Exception('constaint_index must be a list or a numpy array with the indices of the residues in the paired MSA corresponding to protein to make design orthogonal to')
            else:
                print('The reference sequence corresponding to the following indices will be used to constraint the design of the 2nd sequence')
                print(constraint_index)
                self.constraint_msa = deepcopy(self.mutants_sequences)
                
                print('self.constraint_msa : ', np.shape(self.constraint_msa))
                
                if type(self.constraint_index)==list:
                    self.constraint_index = np.array(self.constraint_index)

                self.all_indices = np.arange(self.residue_in_msa)
                self.not_constrained_index = np.setdiff1d(self.all_indices, self.constraint_index)
                
                print('IL2RB', np.shape(self.constraint_msa[:,self.not_constrained_index]))
        
##############################################################################################################################################################
    def msa_to_int(self, seq=None):
        '''
            This Function converts the input MSA to a numerical format (used for GPU processing of Kronecker matrix)
        '''
        
        if seq is None:
            msa = self.msa.getArray()
            int_encoded_msa = np.zeros((self.num_sequences, self.residue_in_msa))
        else:
            msa = seq
            int_encoded_msa = np.zeros((self.residue_in_msa,))

        iterator = self.residue_in_msa_iterator
        alphabet = self.alphabet_bytes
        alphabet_int = self.alphabet_int
        count_q = 0 
        
        for col in iterator:
            for aa_state, aa_state_int in zip(alphabet, alphabet_int):
#                print('shape msa: ', np.shape(msa))               
                if seq is None:
                    int_encoding = np.where(msa[:,col]==aa_state)[0]
                    
                    if len(int_encoding)>0:
                        int_encoded_msa[int_encoding, col] = aa_state_int
                else:
                    int_encoding = np.where(msa[col]==aa_state)[0]

                    if len(int_encoding)>0:
                            int_encoded_msa[col] = aa_state_int
            
            count_q+=1
            
        if self.par:
            return (torch.tensor(int_encoded_msa).to('cuda'),)
        else:
            return (torch.tensor(int_encoded_msa),)
            
##############################################################################################################################################################
    def one_by_one_coupling(self, four_D:bool, normalization_factor, col_m, col_q, cubix_tj, reference_freq_vector, pre_freq_matrix, posi_j_q):
        '''
            This Function calculates the 3D coupling parameters tensor between the q mutable spots and the M-q cte spots
        '''
        possible_state_2 = 0
        par = self.par
        
        if not par:
            alphabet = self.alphabet_bytes
            # Iterate over all possible AAs
            for state_j in alphabet:
                kronecker_pairwise = 0
                
                for ind, actual_residue_1 in enumerate(col_m):
                    for actual_residue_2 in col_q:
                  
                        # Kro vector from M-q cte column    # Kro vector from q column vs state_j AA
                        kronecker_pairwise = kronecker_pairwise + (reference_freq_vector[ind] * (1 if state_j == actual_residue_2 else 0))
                                                               # (1 if state_i == actual_residue_1 else 0) #  Oct 14 
                    kronecker_pairwise = normalization_factor*kronecker_pairwise # added Oct 4
                
                cubix_tj[0, possible_state_2] = kronecker_pairwise # modified Oct 4 #normalization_factor*kronecker_pairwise
                possible_state_2+=1

        if par:
            alphabet = self.alphabet_int.to('cuda')#self.alphabet_int_cuda
            
            if not four_D:
                # Iterate over all possible AAs
                for ind, state_j in enumerate(alphabet):
                    state_vector_j = torch.tile(state_j, (self.num_sequences,))
                    #ref_tensor = torch.tensor(reference_freq_vector)
                    kronecker_pairwise = normalization_factor * torch.sum(reference_freq_vector * torch.eq(state_vector_j, col_q))
                    #print('ind, posi_j_q: ', ind, ', ', posi_j_q)
                    #print('pre_freq_matrix[ind, posi_j_q]: ', pre_freq_matrix[ind, posi_j_q])
                   # print('kronecker_pairwise - (reference_freq_vector * pre_freq_matrix[ind, posi_j_q])', kronecker_pairwise - (reference_freq_vector * pre_freq_matrix[ind, posi_j_q]))
                    cubix_tj[0, possible_state_2] = kronecker_pairwise #- (pre_freq_matrix[possible_state_2, posi_j_q] * pre_freq_matrix[possible_state_2, posi_j_q])                  # modified Oct 4 #normalization_factor*kronecker_pairwise
                    possible_state_2+=1
            
            if four_D:
                possible_state_1 = 0
                
                for state_j in alphabet:
                    state_vector_j = torch.tile(state_j, (self.num_sequences,))
                    possible_state_2 = 0
                    
                    for state_i in alphabet:
                        state_vector_i = torch.tile(state_i, (self.num_sequences,))
                        kronecker_pairwise = normalization_factor * torch.sum(torch.eq(state_vector_i, col_m) * torch.eq(state_vector_j, col_q))
                        cubix_tj[possible_state_1, possible_state_2] = kronecker_pairwise #- (pre_freq_matrix[possible_state_1, posi_j_q] * pre_freq_matrix[possible_state_2, posi_j_q])
                        possible_state_2+=1

                    possible_state_1+=1
        
        return (cubix_tj,)

##############################################################################################################################################################
    
    def ref_sequence_frequencies(self, msa=None):
        '''  
        Kronecker vector of AA observed on reference sequence vs AAs in all other sequences at position Mt
        This vector is to be calculated only one time since the M-q fixed AAs will have the same Kronecker vector
        vs the reference AA that belongs to M-q. However, the coupling of positions in M-q and in q will represent the couplings
        between constant positions and mutable positions.
        '''
        position_iterator = self.residue_in_msa_iterator
        ref_seq = self.ref_seq
        num_seq = self.num_sequences
        par = self.par
        
        if not par:
            msa = self.msa.getArray()
            reference_kronecker_vectors = np.zeros((num_seq, self.residue_in_msa))
            ind = 0
        
            for position, residue in zip(position_iterator, ref_seq):
                # Column ti c=
                to_check = msa[:,position]
                kronecker_vector = to_check==residue
                reference_kronecker_vectors[:,position] = kronecker_vector
                ind+=1

        else:
            if msa is None:
                raise(Exception('provide numerical array as MSA to use GPU processing'))
            else:    
                msa = msa#self.msa_int.to('cuda')
                #reference_kronecker_vectors = np.zeros((num_seq, self.residue_in_msa))
                ref_int = self.msa_to_int(ref_seq)[0]
                tiled_ref = torch.tile(ref_int, (num_seq,1)).to('cuda')
                reference_kronecker_vectors = torch.eq(tiled_ref, msa)
            
        return (reference_kronecker_vectors,)
        
##############################################################################################################################################################

    def kronecker_matrix(self):
        
        type_output = self.type_output
        par = self.par
        
        if type_output not in ['flat', 'both', 'normal']:
            raise Exception(f"Please specify a value for type_output parameter. Must be one of : 'flat', 'both', 'normal' and must be a string. OR JUST KEEP DEFAULT!")
        
        type_output = self.type_output
        parsed_msa = self.msa
        q_ind = self.q_ind
        ind_ref = self.ind_ref
        n_seq = self.num_sequences
        n_cols = self.residue_in_msa
        max_q_ind = len(np.where(np.array(q_ind)>=n_cols)[0])
        neg_q_ind = len(np.where(np.array(q_ind) < 0)[0])

        if max_q_ind>0 or neg_q_ind>0:
            raise Exception("q_ind parameter (index of residues to mutate in the MSA) should be > 0 and < than the length of your MSA")
        else:
            if n_seq>ind_ref>=0 or ind_ref==-1:
                ref_seq = parsed_msa.getArray()[ind_ref]
            else:
                raise Exception("ind_ref parameter (index of reference sequence in the MSA) should be > 0 and < than the length of your MSA")

        cols = int(np.shape(parsed_msa.getArray())[1])
        alphabet_bytes_array = self.alphabet_bytes

        N_possible_states = self.len_alphabet_bytes#int(len(alphabet_bytes_array))
        spot_to_mutate = int(self.len_q_ind)
        # Model as of Oct 17 Matrix k x q.
        pre_freq_matrix_shape = (N_possible_states, spot_to_mutate)
        print('pre_freq_matrix_shape : ', pre_freq_matrix_shape)
        pre_freq_matrix = np.zeros(pre_freq_matrix_shape)
        # Model as of Oct 17 vector valued Matrix  M-q x q x 1 x 21. eval all 21 possibilties for all q vs all M not in q.
        pre_coupling_cubix_shape = (cols-spot_to_mutate, spot_to_mutate, 1, N_possible_states) 
        pre_coupling_cubix = np.zeros(pre_coupling_cubix_shape)
        print('3D tensor shape : ', pre_coupling_cubix_shape)
        # Model as of Oct 17 Matrix valued Matrix 21 x 21 x q x q. eval all possible amino acids in all possible mutation spots.
        pre_coupling_4D_shape = (spot_to_mutate, spot_to_mutate, N_possible_states, N_possible_states) # dimensions of tensor inputs to each entry : N_possible_states, cols, spot_to_mutate
        pre_coupling_tensor = np.zeros(pre_coupling_4D_shape)
        print('4D tensor shape : ', pre_coupling_4D_shape)
        
        inv_n_seq = float(1/n_seq)
        alphabet_enumerator = enumerate(alphabet_bytes_array)
        spot_to_mutate_iterator = self.q_ind_iterator#range(spot_to_mutate)

        ####################################### Computing and storing coupling parameters #######################################
        # Iterate over all possible amino acids
        for state_ind, state in alphabet_enumerator:
            #iterate over columns of the MSA :: FREQUENCY PARAMETERS, only q params
            for col_ind, ind_mat in zip(q_ind, spot_to_mutate_iterator):
                col_i = parsed_msa.getArray()[:,col_ind]
                residues = col_i
                kronecker_delta = 0
                
                for residue in residues:
                    # compute sum of kronecker delta over all possible states vs actual states
                    kronecker_delta = kronecker_delta + (1 if residue == state else 0)
                # weight the sum by the length of the MSA
                param = inv_n_seq*kronecker_delta
                pre_freq_matrix[state_ind, ind_mat] = param

            if self.zero_sum_gauge:
                pre_freq_matrix[state_ind,:] = pre_freq_matrix[state_ind,:] - np.mean(pre_freq_matrix[state_ind,:])
        
        pre_freq_matrix = torch.tensor(pre_freq_matrix)            
        #iterate over columns of the MSA :: COUPLING PARAMETERS
        column_iterator = q_ind
        M_minus_q_dim_index_iterator =  range(cols-spot_to_mutate)
        
        if par:
            parsed_msa = self.msa_to_int()[0]
            inv_n_seq = torch.tensor(inv_n_seq).to('cuda')
            pre_freq_matrix = pre_freq_matrix.to('cuda')
            # for 3D couplings
            cubix_tj = torch.zeros((1, N_possible_states)).to('cuda')
            pre_coupling_cubix = torch.tensor(pre_coupling_cubix).to('cuda')
            #pre_coupling_cubix_shape = torch.tensor(pre_coupling_cubix_shape).to('cuda')
            reference_kronecker_vectors = self.ref_sequence_frequencies(msa=parsed_msa)[0]
            # for 4D couplings
            tensor_tj = torch.zeros((N_possible_states, N_possible_states)).to('cuda')
            pre_coupling_tensor = torch.tensor(pre_coupling_tensor).to('cuda')
            #pre_coupling_4D_shape = torch.tensor(pre_coupling_4D_shape).to('cuda')
        else:
            parsed_msa = parsed_msa.getArray()
            reference_kronecker_vectors = self.ref_sequence_frequencies()[0]
            cubix_tj = np.zeros((1, N_possible_states))
            
        print('CALCULATING 3D COUPLINGS')
        position_q_counter = 0
        # Iterate over all positions in MSA
        for posi_t_M, M_minus_q_dim_index in zip(self.residue_in_msa_iterator, M_minus_q_dim_index_iterator): #OCTOBRE 14 MODIF
            if posi_t_M in q_ind:
                four_D = True
                reference_freq_vector = None
            else:
            #  Kronecker vector of AA observed on reference sequence vs AAs in all other sequences at position Mt 
                reference_freq_vector = reference_kronecker_vectors[:,posi_t_M]
                four_D = False
                tensor_to_use = cubix_tj
            # Iterate over all positions to mutate, Each element of this Tensor is a matrix 
            for posi_j_q, q_dim_index in zip(column_iterator, self.q_ind_iterator):
                # Columns for which couplings will be calculated for each possible AA at that Mt-qj position (vector or matrix output)
                if four_D: #posi_j_q in q_ind and 
                    tensor_to_use = tensor_tj 
                    
                col_m = parsed_msa[:,posi_t_M]
                col_q = parsed_msa[:,posi_j_q]
                #print('position_q_counter: 
                coupling_cubix = self.one_by_one_coupling(normalization_factor = inv_n_seq, 
                                                          col_m = col_m, col_q = col_q, cubix_tj=tensor_to_use,
                                                          reference_freq_vector=reference_freq_vector, four_D=four_D, 
                                                          pre_freq_matrix=pre_freq_matrix, posi_j_q=q_dim_index)
                # 4D tensor dimensions (cols-spot_to_mutate, spot_to_mutate, 1, N_possible_states) 
                if not four_D:  # use this loop to enter 
                    pre_coupling_cubix[M_minus_q_dim_index, q_dim_index, :, :] = coupling_cubix[0]
                else:
                    pre_coupling_tensor[position_q_counter, q_dim_index, :, :] = coupling_cubix[0]

            if four_D:
                position_q_counter+=1

       # if self.zero_sum_gauge:
            #for 
            
        print('DONE CALCULATING 3D COUPLINGS')
        
        if type_output=='flat':
            return(pre_coupling_tensor.flatten(), pre_coupling_4D_shape, pre_freq_matrix.flatten(), pre_coupling_cubix.flatten(), ref_seq, pre_freq_matrix_shape, pre_coupling_cubix_shape)
        elif type_output=='both':
            return(pre_freq_matrix.flatten(), pre_coupling_cubix.flatten(), pre_freq_matrix, pre_coupling_cubix, ref_seq, pre_freq_matrix_shape, pre_coupling_cubix_shape)
        elif type_output=='normal':
            return(pre_freq_matrix, pre_coupling_cubix, q_ind, pre_freq_matrix_shape, pre_coupling_cubix_shape)

##############################################################################################################################################################

    def HamiltonianMC(self, reduced:bool, kronecker_matrix=None, minibatch_integration=False, minibatch_integration_size=0.3, CP_tensor=None):
        '''
            This is a function that performs Hamiltonian Monte Carlo.
            
            Optimization:
            -------------
                theta: vector of "frequency" parameters (random or other initialization are ok).
                omega: vector of coupling parameters (random or other initialization are ok).
            
            HMC parameters:
            --------------
                msa: Prody format Multiple Sequence Alignment
                del_t: value of the pseudo-time step. float.
                steps_per_traj: length of one parameters sampling trajectory (number of times Hamiltons equations are integrated)
                T: pseudo temperature parameter controls spread of the Boltzmann distribution
                lambda: L2 regularization parameter. float or int.
                M: pseudo mass matrix. must be an np.array.
                momentum_model: specify model of momentum to sample with. Must be one of : 'MB', 'Normal', 'MB-Wishart-Normal'.
                warm_up : How many samples to skip before starting to record sequences
                N_training_samples : How many samples to draw to train this MC
        '''
        expected_proba_mat = 0
        
        type_output = self.type_output
        msa = self.msa
        del_t = self.del_t 
        steps_per_traj = self.steps_per_traj
        n_mutant = self.n_seq_gen 
        N_training_samples = self.N_training_samples
        T = self.T 
        lamda = self.lamda
        M = self.M
        momentum_model = self.momentum_model
        warm_up = self.warm_up
        q_ind = self.q_ind
        ind_ref = self.ind_ref
        alphabet_bytes_array = self.alphabet_bytes
        par = self.par
        num_res = self.residue_in_msa
        hmc_T = self.hmc_T
        
        if type_output not in ['flat', 'both', 'normal']:
            raise Exception(f"Value for type_output parameter must be one of : 'flat', 'both', 'normal'. str only. OR JUST KEEP DEFAULT!")
        if momentum_model not in ['MB', 'Normal', 'MB-Wishart-Normal']:
            raise Exception(f"Please specify a value for momentum_model parameter. Must be one of : 'MB', 'Normal', 'MB-Wishart-Normal' and must be a string")
        if type(lamda) not in [int, float]or lamda<0:
            raise Exception(f"Lambda parameter must be a POSITIVE VALUED float or integer not {type(lamda)}")
        if type(T) not in [int, float] or T<0:
            raise Exception(f"T parameter must be a POSITIVE VALUED float or integer not {type(T)}")
        if M:
            if not type(M)==torch.Tensor:
                raise Exception(f"M must be a diagonal matrix of type torch.Tensor not {type(M)}")        
        if not warm_up:
            warm_up = int(round(N_training_samples/10,0))
        elif type(warm_up) is not int:
            raise Exception(f"warm_up must be an integer not {type(warm_up)}")  
        if type(minibatch_integration) is not bool:
            raise Exception(f"minibatch_integration must be a boolean not {type(minibatch_integration)}")  
        if type(minibatch_integration_size) is not int and type(minibatch_integration_size) is not float or minibatch_integration_size>1 or minibatch_integration_size<=0:
            raise Exception(f"minibatch_integration must be a float or integer in the range ]0,1] not {type(minibatch_integration_size)}")  
        if type(reduced) is not bool:
            raise Exception(f"reduced must be a boolean not {type(reduced)}")  
        if minibatch_integration and reduced:
            raise Exception(f"CSG does not currently support minibatch_integration and reduced model at the same time")  
            
        half_step_size = del_t/2
        # 1D and 2D associations computed for the MSA
        #msa_copy = deepcopy(msa)
        #print(q_ind)
        if kronecker_matrix and CP_tensor:
            raise Exception('Specify either kronecker_matrix or CP_tensor')
        elif kronecker_matrix:
            results = kronecker_matrix
            #results = self.kronecker_matrix()#, type_output=type_output)
        elif CP_tensor:
            results = CP_tensor[0]#, type_output=type_output)
            factors_dict = CP_tensor[1]
            m = factors_dict['C1']
            m = factors_dict['C2']

        # Flat Kronecker matrices, np format
        pre_freq_matrix = results[2]
        pre_coupling_cubix = results[3]
        pre_coupling_tensor = results[0]
        # Wait what?
        ref_seq = results[-3]
        # Dimensions of K and F
        coupling_cubix_dimension = results[-1]
        freq_matrix_dimension = results[-2]
        coupling_tensor_dimension= results[1]
        #all_F_C1    
        # size of flat frequencies 1D parameters
        len_theta = freq_matrix_dimension[0] * freq_matrix_dimension[1]
        # Size of flat C2 coupling parameters
        len_phi = coupling_tensor_dimension[0] * coupling_tensor_dimension[1] * coupling_tensor_dimension[2] * coupling_tensor_dimension[3]
        # Draw a flat frequencies 1D parameter of size len_theta to initiate HMC
        theta = torch.distributions.normal.Normal(0, 1).sample(sample_shape=(len_theta,))
        # Draw a flat C2 coupling parameters vector of size len_omega to initiate
        phi = torch.distributions.normal.Normal(0, 1).sample(sample_shape=(len_phi,))

        if not reduced:
            # Size of flat C coupling parameters
            len_omega = coupling_cubix_dimension[0] * coupling_cubix_dimension[1] * coupling_cubix_dimension[2] * coupling_cubix_dimension[3]
            # Draw a flat C coupling parameters vector of size len_omega to initiate
            omega = torch.distributions.normal.Normal(0, 1).sample(sample_shape=(len_omega,))
        
        if par:
            theta = theta.to(torch.float).to('cuda')
            phi = phi.to(torch.float).to('cuda')
            T = torch.tensor(T).to('cuda')

            expected_theta = self.zero_tensor.to('cuda')
            expected_phi = self.zero_tensor.to('cuda')
            
            if not reduced:
                omega = omega.to(torch.float).to('cuda')
                expected_omega = self.zero_tensor.to('cuda')
        else:
            # Flat Kronecker tensors, torch format
            pre_freq_matrix = torch.tensor(pre_freq_matrix)
            
            if not reduced:
                pre_coupling_cubix = torch.tensor(pre_coupling_cubix)
                
            pre_coupling_tensor = torch.tensor(pre_coupling_tensor)
        # Reshaped Kronecker tensors, torch format
        reshaped_pre_freq_matrix = torch.reshape(pre_freq_matrix, freq_matrix_dimension)
        reshaped_pre_coupling_tensor = torch.reshape(pre_coupling_tensor, coupling_tensor_dimension)

        size_sample_J = 0
        couplings_i, coupling_vector = None, None
        
        if not reduced:
            reshaped_pre_coupling_cubix = torch.reshape(pre_coupling_cubix, coupling_cubix_dimension)
            coupling_vector = pre_coupling_cubix * omega
            coupling_vector = torch.tensor(coupling_vector, requires_grad=True)
            size_sample_J = len(coupling_vector)
            # populating the space of parameters: C1
            reshaped_coupling_cubix = torch.reshape(coupling_vector, coupling_cubix_dimension)
            couplings_i = reshaped_coupling_cubix

        # Multiply by omega, theta and phi parameter to optimize 
        freq_vector = pre_freq_matrix * theta
        freq_vector = torch.tensor(freq_vector, requires_grad=True)#.clone().detach()#.requires_grad_(True)
        coupling_vector_2 = pre_coupling_tensor * phi
        coupling_vector_2 = torch.tensor(coupling_vector_2, requires_grad=True)#.clone().detach()#.requires_grad_(True)
        # indices of H and J parameters
        size_sample_H = len(freq_vector)
        size_sample_Q = len(coupling_vector_2)
        size_sample = size_sample_H + size_sample_J + size_sample_Q
        # populating the space of parameters: F
        reshaped_freq_vector = torch.reshape(freq_vector, freq_matrix_dimension)
        # populating the space of parameters: C2
        reshaped_coupling_tensor = torch.reshape(coupling_vector_2, coupling_tensor_dimension)
        
        frequencies_i = reshaped_freq_vector 
        couplings_2_i = reshaped_coupling_tensor
        # Generate first sequences (completely random)
        sample_i = -1
        #################### Modification of DB distribution NOV 1, 2024, to account for C2 couplings #################### 
        # First draw, completely random
        scaling = 1#torch.sqrt(torch.tensor(size_sample)).clone().detach().requires_grad_(True).to('cuda')
       # inv_scaling = 1/scaling
        if not reduced:
            evol_param_tuple = (reshaped_pre_freq_matrix, reshaped_pre_coupling_tensor, reshaped_pre_coupling_cubix)
            likelihood_vector_norm = lamda*(torch.linalg.vector_norm(theta, dtype=torch.double)**2) + lamda*(torch.linalg.vector_norm(omega, dtype=torch.double)**2) + lamda*(torch.linalg.vector_norm(phi, dtype=torch.double)**2)
            stochastic_F = torch.reshape(theta, freq_matrix_dimension)
            stochastic_C1 = torch.reshape(omega, coupling_cubix_dimension)
            stochastic_C2 = torch.reshape(phi, coupling_tensor_dimension)
            stochastic_param_tuple = (stochastic_F, stochastic_C1, stochastic_C2)
        else:
            evol_param_tuple = (reshaped_pre_freq_matrix, reshaped_pre_coupling_tensor)
            likelihood_vector_norm = lamda*(torch.linalg.vector_norm(theta, dtype=torch.double)**2) + lamda*(torch.linalg.vector_norm(phi, dtype=torch.double)**2)
            stochastic_F = torch.reshape(theta, freq_matrix_dimension)
            stochastic_C2 = torch.reshape(phi, coupling_tensor_dimension)
            stochastic_param_tuple = (stochastic_F, stochastic_C2)
        
        L2_penalty = likelihood_vector_norm.clone().detach().requires_grad_(True)
        momenta = self.momentum_model_(T=hmc_T, size_sample=size_sample, model=momentum_model, cov=M)[0] 
        ke_2 = 0.5*torch.linalg.vector_norm(momenta)

        random_energy = self.random_energy(stochastic_param_tuple=stochastic_param_tuple, evol_param_tuple=evol_param_tuple, weights_per_sequence=None)[0]
        
        sampling_result = self.DirichletBoltzmannDist(q_ind=q_ind, ref_seq=ref_seq, mode='training',
                                                      frequencies_i=frequencies_i, 
                                                      couplings_i=couplings_i, 
                                                      couplings_2_i = couplings_2_i,
                                                      freq_matrix_dimension=freq_matrix_dimension, 
                                                      sample_i=scaling, n_mutant=n_mutant, T=T, i=sample_i,
                                                      evol_param_tuple=evol_param_tuple, L2_penalty=L2_penalty, 
                                                      ke=ke_2, random_energy=random_energy)
        generated_sequences = sampling_result[0]
        boltzmann_weight_seq = sampling_result[1]#torch.inf#
        # Score of first draw
        E_1 = boltzmann_weight_seq
        mutant_sequences = generated_sequences#np.empty((1, num_res), dtype=np.bytes_)#np.empty((coupling_cubix_dimension[0]), dtype=np.bytes_)
        acceptance_freq = 0 
        Hamiltonian_array = np.array([])
        Boltzmann_measure_array = np.array([])
        equipartition_rescaling_array = np.array([])
        temperature_array = np.array([])
        kinetic_temp = size_sample*hmc_T

       # if momentum_model=='Normal':#anot M:
            # as of Nov 6, 2024, The integration of Hamiltons eq should be performed separately 
        #    print('size_sample: ', size_sample)
           # M = torch.eye(size_sample)
            
          #  if par:
            #    M = M.to('cuda')
        #    inv_M = M
       # else:
        #    inv_M = 1/M
        # momentum variable  #################### Modification of momentum model NOV 1, 2024, to account for C2 couplings #################### 
        #momenta = self.momentum_model(v, cov=M, T=T, size_sample=size_sample, model=momentum_model)[0]    
        inds_freq = np.arange(start=0, stop=size_sample_H, step=1, dtype=np.int32)
        mini_batch_prop_freq = int(size_sample_H*minibatch_integration_size)
        inds_C1 = np.arange(start=0, stop=size_sample_J, step=1, dtype=np.int32)
        mini_batch_prop_C1 = int(size_sample_J*minibatch_integration_size)
        inds_C2 = np.arange(start=0, stop=size_sample_Q, step=1, dtype=np.int32)
        mini_batch_prop_C2 = int(size_sample_Q*minibatch_integration_size)
        size_reduced = mini_batch_prop_freq + mini_batch_prop_C1 + mini_batch_prop_C2
        
        for sample_i in range(N_training_samples):
            print('#'*40)
            print('#'*20, ' TRAINING SAMPLE :', sample_i , f'/{N_training_samples}')
            freq_vector_i = theta
            
            if not reduced:
                coupling_vector_i = omega
                
            coupling_vector_2_i = phi
            # Introducing reduced INTEGRATION OF HAMILTON'S EQUATIONS, FEB 18, 2025
            if minibatch_integration:
                ind_mini_batch_freq = torch.tensor(np.random.choice(inds_freq, size=mini_batch_prop_freq, replace=False)).to('cuda').to(torch.long)
                    
                if not reduced:
                        ind_mini_batch_C1 = torch.tensor(np.random.choice(inds_C1, size=mini_batch_prop_C1, replace=False)).to('cuda').to(torch.long)
                    
                ind_mini_batch_C2 = torch.tensor(np.random.choice(inds_C2, size=mini_batch_prop_C2, replace=False)).to('cuda').to(torch.long)
                momenta = self.momentum_model_(T=hmc_T, size_sample=size_reduced, model=momentum_model, cov=M)[0] 
            else:
                ind_mini_batch_freq = torch.tensor(inds_freq).to('cuda').to(torch.long) 
                
                if not reduced:
                    ind_mini_batch_C1 = torch.tensor(inds_C1).to('cuda').to(torch.long) 
                
                ind_mini_batch_C2 = torch.tensor(inds_C2).to('cuda').to(torch.long) 
                momenta = self.momentum_model_(T=hmc_T, size_sample=size_sample, model=momentum_model, cov=M)[0]    
            # Introducing evol parsm in U computation Apr 30, 2025
            if not reduced:
                evo_freq = pre_freq_matrix[ind_mini_batch_freq]
                evo_3d_cov = pre_coupling_cubix[ind_mini_batch_C1]
                evo_4D_cov = pre_coupling_tensor[ind_mini_batch_C2]
                #print(evo_4D_cov)
                evol_param = torch.cat((evo_freq, evo_3d_cov, evo_4D_cov)).clone().detach()
            else:
                evo_freq = pre_freq_matrix[ind_mini_batch_freq]
                evo_4D_cov = pre_coupling_tensor[ind_mini_batch_C2]
                evol_param = torch.cat((evo_freq, evo_4D_cov)).clone().detach()
                                       
            momenta_tensor = torch.tensor(momenta, requires_grad=True, dtype=torch.double)
            # pseudo Kinetic energy
            #ke_2 = 0.5*torch.linalg.vector_norm(momenta_tensor)

            if reduced:
                all_param_vector_i = torch.cat((freq_vector_i, coupling_vector_2_i)).clone().detach().requires_grad_(True)
                #all_param_vector_i.grad.zero_()
                likelihood_vector_norm = lamda*(torch.linalg.vector_norm(freq_vector_i, dtype=torch.double)**2) + lamda*(torch.linalg.vector_norm(coupling_vector_2_i, dtype=torch.double)**2)
            else:
                all_param_vector_i = torch.cat((freq_vector_i[ind_mini_batch_freq], coupling_vector_i[ind_mini_batch_C1], coupling_vector_2_i[ind_mini_batch_C2]), dim=0).clone().detach().requires_grad_(True)
                #all_param_vector_i.grad.zero_()
                likelihood_vector_norm = lamda*(torch.linalg.vector_norm(freq_vector_i[ind_mini_batch_freq], dtype=torch.double)**2) + lamda*(torch.linalg.vector_norm(coupling_vector_i[ind_mini_batch_C1], dtype=torch.double)**2) + lamda*(torch.linalg.vector_norm(coupling_vector_2_i[ind_mini_batch_C2], dtype=torch.double)**2)
                # L2 regularized neg log likelihood as described by Malinverni and other authors, pseudo Potential energy
            
            L2_penalty = likelihood_vector_norm.clone().detach().requires_grad_(True)
              
            stochastic_F = torch.reshape(freq_vector_i, freq_matrix_dimension)
            stochastic_C2 = torch.reshape(coupling_vector_2_i, coupling_tensor_dimension)
            
            if not reduced:
                stochastic_C1 = torch.reshape(coupling_vector, coupling_cubix_dimension)
                stochastic_param_tuple = (stochastic_F, stochastic_C1, stochastic_C2)
            else:
                stochastic_param_tuple = (stochastic_F, stochastic_C2)
                    
            U_and_H = self.L2_loglikelihood(all_param_vector_i_=all_param_vector_i, L2_penalty=L2_penalty, ke_2=ke_2, 
                                            evol_param=evol_param, random_energy=random_energy)#, scaling=scaling) 
            U = U_and_H[0]
            print('U = ', U)
            U.retain_grad()
            # Compute the gradient of U since we already know the formula for the gradient of K
            U.backward(retain_graph=True, inputs=all_param_vector_i)#inputs=all_param_vector_i)
            dH_dq_half = all_param_vector_i.grad
            # Leapfrog integrator, first half step for momenta
            momenta_half = torch.sub(momenta, dH_dq_half, alpha=half_step_size)
            all_param_vector_i.grad.zero_()
            
            for step in self.steps_per_traj_iterator:
                # Hamiltonian function  :   H = K + U
                # Compute the gradient of U since we already know the formula for the gradient of K
                
                # Leapfrog integrator, first step for parameters
                if not minibatch_integration:
                    vel_freq_vector = momenta_half[:size_sample_H]#torch.matmul(momenta[:size_sample_H], inv_M)
                    freq_vector_i = torch.add(freq_vector_i, vel_freq_vector, alpha=del_t)
                    vel_coupling_vector_2 = momenta_half[size_sample_H+size_sample_J:]
                    coupling_vector_2_i = torch.add(coupling_vector_2_i, vel_coupling_vector_2, alpha=del_t)
                else:
                    vel_freq_vector = momenta_half[:mini_batch_prop_freq]#torch.matmul(momenta[:size_sample_H], inv_M)
                    freq_vector_i = freq_vector_i.index_copy(0, ind_mini_batch_freq, 
                                         torch.add(freq_vector_i[ind_mini_batch_freq], vel_freq_vector, alpha=del_t)).clone().detach().requires_grad_(True)
                    
                    vel_coupling_vector_2 = momenta_half[mini_batch_prop_freq+mini_batch_prop_C1:]
                    coupling_vector_2_i =  coupling_vector_2_i.index_copy(0, ind_mini_batch_C2, 
                                         torch.add(coupling_vector_2_i[ind_mini_batch_C2], vel_coupling_vector_2, alpha=del_t)).clone().detach().requires_grad_(True)
                    if not reduced:
                        vel_coupling_vector = momenta_half[mini_batch_prop_freq:mini_batch_prop_freq+mini_batch_prop_C1]#torch.matmul(momenta[size_sample_J:], inv_M)
                        coupling_vector_i = coupling_vector_i.index_copy(0, ind_mini_batch_C1, 
                                                                torch.add(coupling_vector_i[ind_mini_batch_C1], 
                                                                          vel_coupling_vector, alpha=del_t)).clone().detach().requires_grad_(True)

                    all_param_vector_i = torch.cat((freq_vector_i[ind_mini_batch_freq], coupling_vector_2_i[ind_mini_batch_C2], coupling_vector_i[ind_mini_batch_C1]), dim=0).clone().detach().requires_grad_(True)
 
                    L2_penalty = lamda*(torch.linalg.vector_norm(freq_vector_i[ind_mini_batch_freq])**2) + lamda*(torch.linalg.vector_norm(coupling_vector_i[ind_mini_batch_C1])**2) + lamda*(torch.linalg.vector_norm(coupling_vector_2_i[ind_mini_batch_C2])**2)
                
                if not reduced and not minibatch_integration:
                    vel_coupling_vector = momenta_half[size_sample_H:size_sample_H+size_sample_J]#torch.matmul(momenta[size_sample_J:], inv_M)
                    coupling_vector_i = torch.add(coupling_vector_i, vel_coupling_vector, alpha=del_t)
                    all_param_vector_i = torch.cat((freq_vector_i, coupling_vector_i, coupling_vector_2_i), dim=0).clone().detach().requires_grad_(True)
                    L2_penalty = lamda*(torch.linalg.vector_norm(freq_vector_i)**2) + lamda*(torch.linalg.vector_norm(coupling_vector_i)**2) + lamda*(torch.linalg.vector_norm(coupling_vector_2_i)**2)

                # Leapfrog integrator, 2nd half step for momenta
                elif reduced:
                    all_param_vector_i = torch.cat((freq_vector_i,  coupling_vector_2_i), dim=0).clone().detach().requires_grad_(True)
                    L2_penalty = lamda*(torch.linalg.vector_norm(freq_vector_i)**2) + lamda*(torch.linalg.vector_norm(coupling_vector_2_i)**2)

                ke_2 = 0.5*torch.linalg.vector_norm(momenta_half)

                stochastic_F = torch.reshape(freq_vector_i, freq_matrix_dimension)
                stochastic_C2 = torch.reshape(coupling_vector_2_i, coupling_tensor_dimension)
            
                if not reduced:
                    stochastic_C1 = torch.reshape(coupling_vector, coupling_cubix_dimension)
                    stochastic_param_tuple = (stochastic_F, stochastic_C1, stochastic_C2)
                else:
                    stochastic_param_tuple = (stochastic_F, stochastic_C2)
                
                U_and_H = self.L2_loglikelihood(all_param_vector_i_=all_param_vector_i, L2_penalty=L2_penalty, ke_2=ke_2, 
                                                evol_param=evol_param, random_energy=None)#, stochastic_param_tuple=stochastic_param_tuple,
                                                #evol_param_tuple=evol_param_tuple)
                U = U_and_H[0]

                # Hamiltonian function  :   H = K + U
                # Compute the gradient of U since we already know the formula for the gradient of K
                U.retain_grad()
                U.backward(retain_graph=True, inputs=all_param_vector_i)#inputs=all_param_vector_i)
                dH_dq_half = all_param_vector_i.grad
                
                # Leapfrog integrator, first half step for momenta
                momenta = torch.sub(momenta_half, dH_dq_half, alpha=del_t)#half_step_size)
                ke_2 = 0.5*torch.linalg.vector_norm(momenta)
                momenta_half = momenta #torch.sub(momenta_half, dH_dq_full, alpha=half_step_size)
                Hamiltonian = U_and_H[1].detach().cpu().numpy() 
                Hamiltonian_array = np.append(Hamiltonian_array, Hamiltonian)
                ke_2 = 0.5*torch.linalg.vector_norm(momenta)

                temperature = ke_2/size_sample
                temperature_array = np.append(temperature_array, temperature.detach().cpu())
                
                all_param_vector_i.grad.zero_()

            stochastic_F = torch.reshape(freq_vector_i, freq_matrix_dimension)
            stochastic_C2 = torch.reshape(coupling_vector_2_i, coupling_tensor_dimension)
            
            if not reduced:
                stochastic_C1 = torch.reshape(coupling_vector, coupling_cubix_dimension)
                stochastic_param_tuple = (stochastic_F, stochastic_C1, stochastic_C2)
            else:
                stochastic_param_tuple = (stochastic_F, stochastic_C2)
        
            random_energy = self.random_energy(stochastic_param_tuple=stochastic_param_tuple, evol_param_tuple=evol_param_tuple, 
                                               weights_per_sequence=None)[0]
            U_and_H = self.L2_loglikelihood(all_param_vector_i_=all_param_vector_i, L2_penalty=L2_penalty, 
                                            ke_2=ke_2, evol_param=evol_param, random_energy=random_energy)#, scaling=scaling)
            U = U_and_H[0]
            U.backward(retain_graph=True, inputs=all_param_vector_i)
            dH_dq_full = all_param_vector_i.grad
            momenta = torch.sub(momenta_half, dH_dq_full, alpha=half_step_size)
            all_param_vector_i.grad.zero_()
            # Make a half step for momentum at the end.
            #p = p - epsilon * grad_U(q) / 2
            # Negate momentum at end of trajectory to make the proposal symmetric
            #p = -p   
            momenta = -momenta
            reshaped_freq_vector = torch.reshape(freq_vector_i, freq_matrix_dimension)
            
            if not reduced:
                reshaped_coupling_cubix = torch.reshape(coupling_vector_i, coupling_cubix_dimension)
                couplings_i = reshaped_pre_coupling_cubix*reshaped_coupling_cubix
            
            reshaped_coupling_tensor = torch.reshape(coupling_vector_2_i, coupling_tensor_dimension)
            #print('pre_freq_matrix size: ', pre_freq_matrix.size())
            frequencies_i = reshaped_pre_freq_matrix*reshaped_freq_vector
            couplings_2_i = reshaped_pre_coupling_tensor*reshaped_coupling_tensor
            print('Parameter optimization with Hamiltonian MC done!')
            print(f'Hamiltonian = {Hamiltonian}')
            print(f'HMC kinetic Temperature = {temperature}')

            ke_2 = 0.5*torch.linalg.vector_norm(momenta)
            # generate random sequences
            sampling_result = self.DirichletBoltzmannDist(q_ind=q_ind, ref_seq=ref_seq, mode='training',
                                                          frequencies_i=frequencies_i, 
                                                          couplings_i=couplings_i, couplings_2_i = couplings_2_i,
                                                          freq_matrix_dimension=freq_matrix_dimension, 
                                                          sample_i=scaling, n_mutant=n_mutant, T=T, i=sample_i,
                                                          evol_param_tuple=evol_param_tuple, L2_penalty=L2_penalty,
                                                          ke=ke_2, random_energy=random_energy)
            #print('Sample sequences generated by Boltzmann-Dirichlet-Multinomial process!')
            generated_sequences = sampling_result[0]
            boltzmann_weight_seq = sampling_result[1]
            # Metropolis-Hastings criteria 
            E_2 = boltzmann_weight_seq
            boltzmann_factor = torch.exp(-(E_2-E_1))#E_2/E_1
            print('E1 ( - ELBO i) = ', E_1)
            print('E2 (- ELBO i+1) = ', E_2)
            Boltzmann_measure_array = np.append(Boltzmann_measure_array, E_2.detach().cpu().numpy())

            if boltzmann_factor>=1:
                theta = torch.flatten(freq_vector_i).double()#freq_vector_i#torch.flatten(freq_vector_i)
                
                if not reduced:
                    omega = torch.flatten(coupling_vector_i).double()#coupling_vector_i#torch.flatten(coupling_vector_i)
                
                phi = torch.flatten(couplings_2_i).double()#coupling_vector_2_i#torch.flatten(couplings_2_i)
                E_1 = E_2
                ke_2_prev = ke_2
                print(f'Move accepted! M-H criterion = {boltzmann_factor}')
                acceptance_freq+=1
                
                if sample_i>=warm_up:
                    N = (warm_up<sample_i)+1
                    inv_N = 1/N
                    print('mutant_sequences size :', np.shape(mutant_sequences))
                    print('generated_sequences size :', np.shape(generated_sequences))
                    mutant_sequences = np.concatenate((mutant_sequences, generated_sequences), axis=0)
                    expected_theta = expected_theta  + theta
                    expected_phi = expected_phi + phi
                    theta = inv_N*expected_theta
                    phi = inv_N*expected_phi
                    expected_proba_mat = sampling_result[2]

                    if not reduced:
                        expected_omega = expected_omega + omega
                        omega = inv_N*expected_omega
            else:
                uniform_rv = torch.distributions.uniform.Uniform(0, 1)
                uniform_sample = uniform_rv.sample()
                
                if boltzmann_factor>=uniform_sample:
                    theta = torch.flatten(freq_vector_i).double()#freq_vector_i 

                    if not reduced:
                        omega = torch.flatten(coupling_vector_i).double()#coupling_vector_i
                    
                    phi = torch.flatten(couplings_2_i).double()#coupling_vector_2_i
                    E_1 = E_2
                    ke_2_prev = ke_2
                    print(f'Move accepted! Probability of acceptance = {boltzmann_factor}')
                    acceptance_freq+=1

                    if sample_i>=warm_up:
                        N = (warm_up<sample_i)+1
                        inv_N = 1/N
                        print('mutant_sequences size :', np.shape(mutant_sequences))
                        print('generated_sequences size :', np.shape(generated_sequences))
                        mutant_sequences = np.concatenate((mutant_sequences, generated_sequences), axis=0)
                        expected_theta = expected_theta  + theta
                        expected_phi = expected_phi + phi
                        expected_theta = inv_N*expected_theta
                        expected_phi = inv_N*expected_phi

                        if not reduced:
                            expected_omega = expected_omega + omega
                            expected_omega = inv_N*expected_omega

                        expected_proba_mat = sampling_result[2]
                else:
                    print(f'Move refused! Probability of refusal = {1 - boltzmann_factor}')

            #print(mutant_sequences)

        print('acceptance probability during simulation = ', acceptance_freq/self.N_training_samples)

        #output_path_params = os.path.join('.', 'proba_matrix.npy')

        #with open(output_path_proba_mat, 'wb') as f:
           # np.save(f, expected_proba_mat.cpu().detach().numpy())

        if not reduced:
            stochastic_param_tup = (expected_theta, expected_omega, expected_phi, freq_matrix_dimension)
        else:
            stochastic_param_tup = (expected_theta, expected_phi, freq_matrix_dimension)
        
        #if not self.force_equi_part:
        equipartition_rescaling_array = None
        #[1:,:]
        return (mutant_sequences, Hamiltonian_array, Boltzmann_measure_array, equipartition_rescaling_array, 
                    temperature_array, stochastic_param_tup, expected_proba_mat)

##############################################################################################################################################################
    def random_energy(self, stochastic_param_tuple:tuple, evol_param_tuple:tuple, weights_per_sequence:torch.tensor, msa=None, k=None):

        #### random energy
        random_msa = self.random_msa 
        
        if not k:
            num_seq_random = self.num_sequences_random
        else:
            num_seq_random = k
            
        probabilities = self.uniform_rand_proba_indices
        random_indices  = torch.multinomial(probabilities, num_samples=num_seq_random, replacement=False)
        subset_msa_rndm = random_msa[random_indices,:]


        if len(stochastic_param_tuple)>2:
            stochastic_C1 = stochastic_param_tuple[2]
        else:
            stochastic_C1 = None
            
        stochastic_C2 = stochastic_param_tuple[1]
           #def calc_energy(self, frequencies_i, couplings_i, couplings_2_i, seq, evol_param_tuple:tuple): 
        
        log_likelihood_array_rndm = torch.tensor(self.calc_energy_batch(frequencies_i=stochastic_param_tuple[0], couplings_i=stochastic_C1, 
                                                            couplings_2_i=stochastic_C2, msa=subset_msa_rndm, evol_param_tuple=evol_param_tuple, 
                                                                       weights_per_sequence=weights_per_sequence)[0])

        random_energy = torch.sum(log_likelihood_array_rndm).to('cuda')#(1/num_seq_random)*
       
        #### MSA energy
        if not msa:
            msa = self.msa
            N = self.num_sequences
        else:
            N = msa.numSequences()

        log_likelihood_array_actual = torch.tensor(self.calc_energy_batch(frequencies_i=stochastic_param_tuple[0], couplings_i=stochastic_C1, 
                                                            couplings_2_i=stochastic_C2, msa=msa, evol_param_tuple=evol_param_tuple, 
                                                                          weights_per_sequence=None)[0])

        msa_energy = torch.sum(log_likelihood_array_actual).to('cuda')#(1/N)*

        del_H_r_a = msa_energy - random_energy 
        
        return (del_H_r_a,)
        
##############################################################################################################################################################
    
    def L2_loglikelihood(self, all_param_vector_i_:torch.tensor, L2_penalty:torch.tensor, ke_2:torch.tensor, evol_param:torch.tensor, random_energy=None, stochastic_param_tuple=None, evol_param_tuple=None):#, scaling=scaling)
        '''
              This function calculates the L2 regularized negative log likelihood as described by Malinverni and Babu, pseudo Potential energy.
        '''

        if not random_energy and (stochastic_param_tuple and evol_param_tuple):
            #random_energy = self.zero_tensor.to('cuda')
            random_energy = self.random_energy(stochastic_param_tuple, evol_param_tuple, msa=None, k=100)[0]
        elif random_energy:
            random_energy = self.zero_tensor.to('cuda')
        elif not random_energy and not(stochastic_param_tuple and evol_param_tuple):
            random_energy = self.zero_tensor.to('cuda')
            
        all_penalty = torch.add(random_energy, L2_penalty)
        
        U = torch.add(torch.sum(all_param_vector_i_*evol_param), all_penalty) #torch.tensor(unreg_U, requires_grad=True)
        Hamiltonian = torch.add(U, ke_2)#/self.T)
        # because the -log is taken for exp of a negative quantity -log(exp(-H/T)
        #negloglikelihood = Hamiltonian
         #torch.tensor(loglikelihood + L2_penalty), requires_grad=True)

        return (U, Hamiltonian)#, loglikelihood)

################################################################################################################################################
    def momentum_model_(self, size_sample:int, model:str, cov, T=1):# cov:torch.Tensor, 
        '''
              This is a function that computes the Maxwell-Boltzmann Distribution for: 
                - `v`: np.array of velocities
                - `M`: diagonal pseudo-mass matrix associated to parameters 
                - `T`: pseudo-temperature of the system of interest =1
                - 'cov' : Identity matrix used as Covariance in Normal model
        '''
        r = 1 # Ideal gases constant originally expressed in J / (mol * K)
        pi = torch.pi 
              
        if model=='MB':
            v = self.v 
                                
            if self.par:
              v = v.to('cuda')
            # Maxwell-Boltzmann distribution                                   
            prob = 4.* pi * ((1 / (2. * pi * T))**1.5) * (v**2) * torch.exp(- (v**2)/(2 * T))
            velocities_index = torch.multinomial(prob, size_sample, replacement=True)
            velocities = v[velocities_index]    
            momenta = velocities
            #momenta = torch.matmul(velocities, cov)
            #velocities = np.random.choice(a=v, size=size_sample, p=prob)
            #momenta = numpy.matmul(x1=velocities, x2=cov)
                   
        elif model=='Normal':
            #if type(cov)!=torch.Tensor:
             #   raise Exception('cov must be a square matrix of type torch.Tensor')
                
            #momenta_distribution = #torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(size_sample), covariance_matrix=cov)
            momenta_distribution = torch.distributions.normal.Normal(T, 1)#
            momenta = momenta_distribution.sample((size_sample,))
            
            if self.par:
                momenta = momenta.to('cuda')
            
        elif model=='MB-Normal-Wishart':
            prob = 4.* pi * ((1 / (2 * pi * T))**1.5) * (v**2) * np.exp(- (v**2)/(2 * T))
            velocities = np.random.choice(a=v, size=size_sample, p=prob)
            covariance = invwishart(df=size_sample, scale=cov)
            momenta = np.random.multivariate_normal(mean=np.zeros(velocities), cov=covariance)

        else:
            raise Exception('Use option MB for now for the momentum model')
            
        return(momenta,)    

##############################################################################################################################################################

    def DirichletBoltzmannDist(self, q_ind:np.array, ref_seq:np.bytes_, frequencies_i:torch.Tensor, mode:str, couplings_i, freq_matrix_dimension:tuple, 
                               couplings_2_i:torch.Tensor, evol_param_tuple:tuple, L2_penalty=0, ke=0, n_mutant = 10, T = 1, 
                               random_energy=None, expected_proba_mat=int(0), i=1, sample_i=1):
        '''
            This is a function that draws n_mutant sequences from a Dirichlet-multinomial model parametrized by the Boltzmann
            weights of each residue at each position

            q_ind: indices of columns to mutate, must be an np.array
            ref_seq: template sequence on which to introduce mutations at q_ind spots, must be an np.bytes_
            frequencies_i: 1D parameters matrix, must be an torch.Tensor
            couplings_i: pairwise parameters cubix, must be a torch.Tensor or None if option reduced is used
            alphabet_bytes_array: array of possible states (21 characters 20 residues + Gap ('-'), must be an np.bytes_
            freq_matrix_dimension: Dimensions of frequencies_i, must be a tuple
            n_mutant: number of sequences to generate, must be an integer
            T: pseudo-temperature parameter, must be either an int or float =>0
            couplings_2_i: pairwise parameters tensor, must be a torch.Tensor
        '''
        
        if len(evol_param_tuple)>2:
            reshaped_pre_coupling_cubix = evol_param_tuple[2]
            reshaped_pre_coupling_tensor = evol_param_tuple[1]
            couplings_i = torch.reshape(couplings_i, reshaped_pre_coupling_cubix.size()).to('cuda')
            reshaped_pre_coupling_cubix = reshaped_pre_coupling_cubix.to('cuda')
                
        else:
            reshaped_pre_coupling_tensor = evol_param_tuple[1]
            
        reshaped_pre_freq_matrix = evol_param_tuple[0]

        if type(expected_proba_mat)==int:
            expected_proba_mat_flag = False
            
        elif type(expected_proba_mat)==torch.Tensor:
            print('ok')
            check = (expected_proba_mat.sum(axis=0).sum()==self.len_q_ind)
            
            if check:
                expected_proba_mat_flag = True
            else:
                #raise Exception('Probabilities do not sum up to 1, Review the expected_proba_mat input or ignore it')\
                print('Probabilities do not sum up to 1, Review the expected_proba_mat input or ignore it')
                print('Sampling from A BDM derived from model parameters and data')
                expected_proba_mat_flag = False
        else:
            expected_proba_mat_flag = False
           # stochastic_param_tuple = (frequencies_i, couplings_i, couplings_2_i)
            #evol_param_tuple = (reshaped_pre_freq_matrix, reshaped_pre_coupling_tensor)

        frequencies_i = torch.reshape(frequencies_i, reshaped_pre_freq_matrix.size()).to('cuda')
        couplings_2_i = torch.reshape(couplings_2_i, reshaped_pre_coupling_tensor.size()).to('cuda')
        reshaped_pre_freq_matrix = reshaped_pre_freq_matrix.to('cuda')
        reshaped_pre_coupling_tensor = reshaped_pre_coupling_tensor.to('cuda')

        if len(evol_param_tuple)>2:
           # print('len(evol_param_tuple)>2')
            stochastic_param_tuple = (frequencies_i, couplings_i, couplings_2_i)
        else:
            #print('len(evol_param_tuple)<=2')
            stochastic_param_tuple = (frequencies_i,  couplings_2_i)

        #print('stochastic_param_tuple : ', stochastic_param_tuple)
        residues_in_msa_iterator = self.residue_in_msa_iterator
        alphabet_bytes_array = self.alphabet_bytes
         
        num_states = self.len_alphabet_bytes#
        BWM = torch.tensor([1/num_states]).repeat(freq_matrix_dimension[0], freq_matrix_dimension[1]) #torch.empty(freq_matrix_dimension)
        # N residues to mutate
        len_q_ind = self.len_q_ind

        # where to store mutated residues 
        if mode=='sampling':
            zero_tensor = self.zero_tensor#.to('cuda')
            mutants_array = np.empty((n_mutant, len_q_ind), dtype=np.bytes_)
            # Index of mutated residues in overall 
            mutated_residue_index_mat = np.empty((n_mutant, self.len_q_ind))
            # Mutated sequences (full ength)
            mutants_sequences = np.tile(self.ref_seq, (n_mutant, 1))
            ke = self.zero_tensor

            #1if expected_proba_mat:
            #    prob_mat = expected_proba_mat
            #else:
            #    prob_mat = torch.empty(freq_matrix_dimension)
                
            if not L2_penalty:
                    #print(' couplings_i size = :', couplings_i)    
                #else:
                L2_penalty = 0#self.lamda*(torch.linalg.vector_norm(frequencies_i)**2) + self.lamda*(torch.linalg.vector_norm(couplings_2_i)**2)
                    
                #reshaped_pre_coupling_cubix = reshaped_pre_coupling_cubix.to('cuda')
        #L2_penalty = (self.lamda*(torch.frequencies_i)**2)) + (self.lamda*(torch.flatten(couplings_i)**2)) + (self.lamda*(torch.flatten(couplings_2_i)**2))
                
        elif mode=='training':
            zero_tensor = self.zero_tensor
            mutants_array = self.mutants_array 
            # Index of mutated residues in overall 
            mutated_residue_index_mat = self.mutated_residue_index_mat 
            # Mutated sequences (full ength)
            mutants_sequences = self.mutants_sequences 
            #Probability per residue per mutation spot
           # if not ke:
                #raise Exception("You must specify a 'kinetic energy' ke to train the model")
        else:
            raise Exception("mode option only takes 2 values: 'training' or 'sampling'")

        prob_mat = torch.empty(freq_matrix_dimension)
        aa_enumerator = self.aa_enumerator
        q_ind_iterator = self.q_ind_iterator 

        order2_term = 0
        mutated_column_count = 0
        #posterior = zero_tensor.double().to('cuda')
        #posterior_array = np.empty((n_mutant,len_q_ind)) #zero_tensor.double().to('cuda')
        # Iterating over possible AA states
        ref_hamiltonian_all = self.energy_ref(frequencies_i, couplings_i, couplings_2_i, evol_param_tuple)
        ref_hamiltonian = ref_hamiltonian_all[0]
        ref_hamiltonian_per_position = ref_hamiltonian_all[1]
        ### Protein length correction term March 20th
        #prot_size_correction_factor_p1 = self.len_p1
       # prot_size_correction_factor_p2 = self.len_p2
       # p1 = self.p1
       # p2 = self.p2
        if mode=='training':
            for possible_state_key, possible_state in aa_enumerator:
                # Iterating over positions to mutate
                mutated_column_count = 0
                for ind_q in q_ind_iterator:
                    # Computing Boltzmann weigths of each 21 character for each mutation spot
                    # According to factorization proof we can assign Boltzmann weigths and Dirichlet probabilities 
                    # To each possible AA at each position to be mutated
                    # Dimensions of F tensor :  (N_possible_states, spot_to_mutate)
                    order1_term = frequencies_i[possible_state_key, ind_q] * reshaped_pre_freq_matrix[possible_state_key, ind_q]
                    # Dimensions 4D couplings  : (spot_to_mutate, spot_to_mutate, N_possible_states, N_possible_states) 
                    higher_order_term = torch.sum(couplings_2_i[:, ind_q, possible_state_key, :]*reshaped_pre_coupling_tensor[:, ind_q, possible_state_key, :])
                    # Dimensions 3D couplings  : (cols-spot_to_mutate, spot_to_mutate, 1, N_possible_states)
                    if couplings_i is not None:
                        order2_term =  torch.sum(couplings_i[:, ind_q, 0, possible_state_key]*reshaped_pre_coupling_cubix[:, ind_q, 0, possible_state_key])
                    Boltzmann_weight = torch.exp(-(order2_term+order1_term+higher_order_term/T))#-ref_hamiltonian_position_q)/T)#*sample_i*prot_size_correction_factor_p1))
                    # Populating entries of BWM
                    BWM[possible_state_key, ind_q] = BWM[possible_state_key, ind_q] + Boltzmann_weight #+ torch.finfo(torch.float32).min

        log_prior = 0
        mutated_column_count = 0
        
        for ind_q in q_ind_iterator:
                # probabilities of possible_state AA in all q mutable positions
                if mode=='training' or (mode=='sampling' and not expected_proba_mat_flag):
                    DirichletBoltzmann_mixture = torch.distributions.dirichlet.Dirichlet(BWM[:, ind_q])#/sum(BWM[ind_q,:]))
                    # Populating entries of probability matrix
                    prob_mat[:, ind_q] = DirichletBoltzmann_mixture.sample()
                
                elif mode=='sampling' and expected_proba_mat_flag:
                    prob_mat[:, ind_q] = expected_proba_mat[:, ind_q]
                
                # Draw column j of q to mutate, vector 1 mutation position x 21 AA. Draw n_mutant
                Multinomial_distribution = torch.distributions.multinomial.Multinomial(total_count=n_mutant, probs=prob_mat[:, ind_q])#.numpy() #prob_mat[:,ind_q]
                # draw n_mutant times the 21 indices, Indicating which AA index have been picked by the Boltzamann-Dirichlet-Multinomial mixture
                # example:  [ 0.  2.  0.  0.  2.  9.  1.  1.  5.  4.  1.  1.  1.  3.  0.  2.  1. 12. 0.  3.  2.]
                residue_q_i_index = Multinomial_distribution.sample()#.numpy()

                log_prior+=(Multinomial_distribution.log_prob(residue_q_i_index))#-log_prior_normalization)
               
                #print('residue_q_i_index before correction: ', residue_q_i_index)
                residue_q_i_index = torch.where(residue_q_i_index>0, residue_q_i_index, zero_tensor).cpu().numpy() 
                # November 5, 2024 modification of mutated columns generations 
                #mutated_residue_index_mat[:,ind_q] = np.tile(alphabet_bytes_array[residue_q_i_index], (50,1))
                residues_q_i = np.array([])
                # What are the indices of the mutant residues in the alphabet array
                index_residues_q_i = np.array([])
                #populating the jth column of the generated sequences
            
                for AA_tup, index_sampled_AA in zip(enumerate(alphabet_bytes_array), residue_q_i_index):
                    # index in alphabet, AA
                    ind_AA, AA =  AA_tup[0], AA_tup[1]
                   # usefull_dict[f'{ind_AA}'] = 0
                    
                    if index_sampled_AA:
                        # How many times this AA has been picked ?
                        mutant_AA = [AA]*int(index_sampled_AA) # index_sampled_AA is the number of time the residue x have been picked for column q
                        # Do the same for its index in the alphabet
                        mutated_res_index = [int(ind_AA)]*int(index_sampled_AA)   # ind_AA is the index of that residue in the alphabet
                       # usefull_dict[f'{ind_AA}'] = int(index_sampled_AA)
                        # Store both info
                        residues_q_i = np.append(residues_q_i, mutant_AA)
                        # list of the alphabet indices of residues picked for column q 
                        index_residues_q_i = np.append(index_residues_q_i, mutated_res_index)
            
                mutated_residue_index_mat[:,ind_q] = index_residues_q_i       
                mutants_array[:,ind_q] = residues_q_i
                mutants_sequences[:,self.q_ind[mutated_column_count]] = residues_q_i 
                mutated_column_count+=1
        #print('mutants_sequences : ', mutants_sequences)
        print('Mutants generated by Boltzmann-Dirichlet-Multinomial process!')
        expected_log_prior = self.inv_n_mutant*log_prior
        print('Expected log prior = ', expected_log_prior)

        mutants_sequences = pr.MSA(mutants_sequences, aligned=True)     
        #sequences_and_weights = self.sequence_weights(mutants_sequences)
        #mutants_sequences = sequences_and_weights[0]
        #weights_per_sequence = sequences_and_weights[1]
            
        if mode=='training':
            kappa = self.kappa
            actual_random_energy = kappa*random_energy# torch.sum(log_likelihood_array_rndm).to('cuda')
            generated_random_energy = kappa*self.random_energy(stochastic_param_tuple=stochastic_param_tuple, 
                                                               evol_param_tuple=evol_param_tuple, msa=mutants_sequences,
                                                               weights_per_sequence=None)[0]
        else:
            actual_random_energy = 0
            ind_unique = pr.uniqueSequences(mutants_sequences[:,self.q_ind], seqid=1, turbo=True)
            mutants_sequences = mutants_sequences[ind_unique]
            individual_hamiltonians = self.calc_energy_batch(frequencies_i=frequencies_i, couplings_i=couplings_i, couplings_2_i=couplings_2_i, 
                                                             msa=mutants_sequences, evol_param_tuple=evol_param_tuple, 
                                                             weights_per_sequence=None)[0]
            generated_random_energy = individual_hamiltonians.sum()

        if self.constraint:
            constraint_msa = self.constraint_msa
            print('IL2RB mutants_sequences shape :', np.shape(mutants_sequences.getArray()))#[:, self.not_constrained_index]
            
            if mode=='training':
                constraint_msa[:, self.not_constrained_index] = mutants_sequences.getArray()[:, self.not_constrained_index]
            else:
                constraint_msa[ind_unique, self.not_constrained_index] = mutants_sequences.getArray()[ind_unique, self.not_constrained_index]

            constraint_msa = pr.MSA(constraint_msa, aligned=True)  
            constraint_individual_hamiltonians = self.calc_energy_batch(frequencies_i=frequencies_i, couplings_i=couplings_i, couplings_2_i=couplings_2_i, 
                                                             msa=constraint_msa, evol_param_tuple=evol_param_tuple, 
                                                             weights_per_sequence=None)[0]
            constraint_energy = constraint_individual_hamiltonians.sum()
            
            if mode=='training':
                constraint_energy = constraint_energy*kappa
        else:
            constraint_energy = 0

        print('L2 penalty term = ', L2_penalty)
        print('HMC momentum = ', ke)
        print('k*(Ha - Hr)/T = ', actual_random_energy)
        print('k*(Hg - Hr)/T = ', generated_random_energy)
        print('k*Hcnst/T = ', constraint_energy)
        print('Sample - log likelihood (Ht/T) = ', actual_random_energy + generated_random_energy)
        boltzmann_weight_seq_all_generated = -self.inv_n_mutant*(actual_random_energy+generated_random_energy-constraint_energy+(ke.to('cuda')/T.to('cuda'))) + L2_penalty
        print('Expected L2 regularized log likelihood  = ', boltzmann_weight_seq_all_generated)
        #boltzmann_weight_seq_all_generated = self.inv_n_mutant*(max_hamiltonian + torch.log(torch.sum(torch.exp((Hamiltonian_all_generated-max_hamiltonian)))))                             
            #ELBO
        ELBO = -(boltzmann_weight_seq_all_generated - expected_log_prior)
        #print('ELBO: ', ELBO)
            #print('mutants_sequences: ', mutants_sequences)
        if mode=='training':
            return(mutants_sequences, ELBO, prob_mat)
        else:
            return(mutants_sequences, ELBO, prob_mat, individual_hamiltonians)
        

##############################################################################################################################################################
    def sequence_weights(self, mutants_sequences:pr.MSA):

        df = pd.DataFrame({key:val for key, val in zip(mutants_sequences.getLabels(), 
                                                       mutants_sequences.getArray())}).T#.drop_duplicates()
        # Convert to DataFrame and add probability
        # For bytes to string conversion
        result = (
          df.apply(lambda row: b''.join(row).decode('ascii'), axis=1)  # Join and decode bytes
           .value_counts()
          .rename('count')
           .reset_index()
            .rename(columns={'index': 'sequence'})
            .assign(probability=lambda x: x['count']/x['count'].sum())
        )
        # Add original indices (bytes-aware version)
        result['original_indices'] = [
            df.index[df.apply(lambda row: b''.join(row), axis=1) == seq.encode('ascii')].tolist()
            for seq in result['sequence']
        ]
        weights_per_sequence = torch.tensor(result['probability'])#.to('cuda')
        ind_uniq = result['original_indices']# [int(x) for x in result['original_indices']]
        print('ind_uniq : \n', ind_uniq)
        mutants_sequences = mutants_sequences[ind_uniq,:]

        return(mutants_sequences, weights_per_sequence)

##############################################################################################################################################################
    def energy_ref(self, frequencies_i:torch.tensor, couplings_i:torch.tensor, couplings_2_i:torch.tensor, evol_param_tuple:tuple):

        # indices of the residues of ref sequence in the alphabet array
        #mutated_residue_index_mat = self.mutated_residue_index_mat 
        mutated_res = self.ref_seq[self.q_ind]
        
        freq_contrib_hamiltoninan = torch.tensor([0]).double().to('cuda')
        coupling_contrib_hamiltonian = torch.tensor([0]).double().to('cuda')
        coupling_contrib_hamiltonian_2 = torch.tensor([0]).double().to('cuda')
        len_q_ind = self.len_q_ind
        
        if len(evol_param_tuple)>2:
            reshaped_pre_freq_matrix = evol_param_tuple[0].to('cuda')
            reshaped_pre_coupling_tensor = evol_param_tuple[1].to('cuda')
            reshaped_pre_coupling_cubix = evol_param_tuple[2].to('cuda')
        else:
            reshaped_pre_freq_matrix = evol_param_tuple[0].to('cuda')
            reshaped_pre_coupling_tensor = evol_param_tuple[1].to('cuda')
            coupling_contrib_hamiltonian_position_q = self.zero_tensor.to('cuda')
            
        ref_hamiltonian_per_q = torch.empty((len_q_ind,))
        
        # Iterating over the mutation positions
        for ind_q in range(len_q_ind):
                mutated_res_indices = np.where(mutated_res[ind_q]==self.alphabet_bytes)[0]
                # pick index of AA in alphabet array at position ind_q
                freq_gen_ind = int(mutated_res_indices)
                freq_contrib_hamiltoninan_position_q = (frequencies_i[freq_gen_ind, ind_q]*reshaped_pre_freq_matrix[freq_gen_ind, ind_q])#/size_correction
                #print('freq_contrib_hamiltoninan_position_q : ', freq_contrib_hamiltoninan_position_q)
                #print('frequencies_i.size() : ', frequencies_i.size())
                freq_contrib_hamiltoninan+=freq_contrib_hamiltoninan_position_q
                # Model as of Oct 17 vector valued Matrix  M-q x q x 1 x 21. eval all 21 possibilties for all q vs all M not in q
                if couplings_i is not None:
                    coupling_contrib_hamiltonian_position_q = torch.sum(couplings_i[:,ind_q,:,freq_gen_ind]*reshaped_pre_coupling_cubix[:,ind_q,:,freq_gen_ind])#/size_correction
                    coupling_contrib_hamiltonian+=coupling_contrib_hamiltonian_position_q 
                # Matrix valued Matrix q x q x 21 x 21. eval all possible amino acids in all possible mutation spots.
                #                 21 x 21 x q x q
                coupling_contrib_hamiltonian_2_position_q = torch.sum(couplings_2_i[:,ind_q,:,freq_gen_ind]*reshaped_pre_coupling_tensor[:,ind_q,:,freq_gen_ind])#/size_correction
                coupling_contrib_hamiltonian_2+=coupling_contrib_hamiltonian_2_position_q
                
                ref_hamiltonian_per_q[ind_q] = freq_contrib_hamiltoninan_position_q + coupling_contrib_hamiltonian_position_q + coupling_contrib_hamiltonian_2_position_q
        
        hamiltonian_ref = freq_contrib_hamiltoninan + coupling_contrib_hamiltonian + coupling_contrib_hamiltonian_2

        return (hamiltonian_ref, ref_hamiltonian_per_q)

##############################################################################################################################################################
    def calc_energy(self, frequencies_i, couplings_i, couplings_2_i, seq, evol_param_tuple:tuple):

        # indices of the residues of ref sequence in the alphabet array
        mutated_res = seq[self.q_ind]
        
        freq_contrib_hamiltoninan = torch.tensor([0]).double().to('cuda')
        coupling_contrib_hamiltonian = torch.tensor([0]).double().to('cuda')
        coupling_contrib_hamiltonian_2 = torch.tensor([0]).double().to('cuda')

        if len(evol_param_tuple)>2:
            reshaped_pre_freq_matrix = evol_param_tuple[0]#.cpu()
            reshaped_pre_coupling_tensor = evol_param_tuple[1]#.cpu()
            reshaped_pre_coupling_cubix = evol_param_tuple[2]#.cpu()
        else:
            reshaped_pre_freq_matrix = evol_param_tuple[0]#.cpu()
            reshaped_pre_coupling_tensor = evol_param_tuple[1]#.cpu()
            coupling_contrib_hamiltonian_position_q = self.zero_tensor.to('cuda')
        
        len_q_ind = self.len_q_ind
        hamiltonian_ref = self.energy_ref(frequencies_i, couplings_i, couplings_2_i, evol_param_tuple)[0]
        hamiltonian_per_q = torch.empty((len_q_ind,)).to('cuda')
        
        # Iterating over the mutation positions
        for ind_q in range(len_q_ind):
                mutated_res_indices = np.where(mutated_res[ind_q]==self.alphabet_bytes)[0]
                
                if len(mutated_res_indices)<1:
                    print('problem : ', mutated_res[ind_q])
                    print('mutated_res_indices : ', mutated_res_indices)
                # pick index of AA in alphabet array at position ind_q
                freq_gen_ind = mutated_res_indices.astype(int)#int(mutated_res_indices)
                freq_contrib_hamiltoninan_position_q = frequencies_i[freq_gen_ind, ind_q]*reshaped_pre_freq_matrix[freq_gen_ind, ind_q]#/size_correction
                #print('freq_contrib_hamiltoninan_position_q: ', freq_contrib_hamiltoninan_position_q)
                freq_contrib_hamiltoninan+=freq_contrib_hamiltoninan_position_q
               # print('freq_contrib_hamiltoninan: ', freq_contrib_hamiltoninan)
                # Model as of Oct 17 vector valued Matrix  M-q x q x 1 x 21. eval all 21 possibilties for all q vs all M not in q
                if couplings_i is not None and len(evol_param_tuple)>2:
                    coupling_contrib_hamiltonian_position_q = torch.sum(couplings_i[:,ind_q,:,freq_gen_ind]*reshaped_pre_coupling_cubix[:,ind_q,:,freq_gen_ind])#/size_correction
                    coupling_contrib_hamiltonian+=coupling_contrib_hamiltonian_position_q 
                # Matrix valued Matrix q x q x 21 x 21. eval all possible amino acids in all possible mutation spots.
                #                 21 x 21 x q x q
                coupling_contrib_hamiltonian_2_position_q = torch.sum(couplings_2_i[:,ind_q,:,freq_gen_ind]*reshaped_pre_coupling_tensor[:,ind_q,:,freq_gen_ind])#/size_correction
                coupling_contrib_hamiltonian_2+=coupling_contrib_hamiltonian_2_position_q
                
                hamiltonian_per_q[ind_q] = freq_contrib_hamiltoninan_position_q + coupling_contrib_hamiltonian_position_q + coupling_contrib_hamiltonian_2_position_q
        
        hamiltonian = (freq_contrib_hamiltoninan + coupling_contrib_hamiltonian + coupling_contrib_hamiltonian_2 - hamiltonian_ref)/self.T #

        return (hamiltonian, hamiltonian_per_q)

##############################################################################################################################################################
    def calc_energy_batch(self, msa:pr.MSA, frequencies_i, couplings_i, couplings_2_i, evol_param_tuple, weights_per_sequence=None):
        """
        Batch version of calc_energy_par.
        Returns array of energies matching individual calc_energy_par results.
        """
        seq_length = len(msa[0].getArray())
        if max(self.q_ind) >= seq_length:
            raise ValueError(f"q_ind positions exceed sequence length")
    
        # ===== 2. Precompute Reference Energy =====
        hamiltonian_ref = self.energy_ref(frequencies_i, couplings_i, couplings_2_i, evol_param_tuple)[0]
    
        # ===== 3. Initialize =====
        alphabet_dict = {res: idx for idx, res in enumerate(self.alphabet_bytes)}
    
        # ===== 4. Unpack Evolution Parameters =====
        if len(evol_param_tuple) > 2:
            reshaped_pre_freq_matrix, reshaped_pre_coupling_tensor, reshaped_pre_coupling_cubix = evol_param_tuple
        else:
            reshaped_pre_freq_matrix, reshaped_pre_coupling_tensor = evol_param_tuple
            reshaped_pre_coupling_cubix = None
        
        # Clean out duplicates and calculates sample probabilities (weights)
        mutants_sequences = deepcopy(msa)
        num_uniq_seq = mutants_sequences.numSequences()
        individual_hamiltonians = np.empty(num_uniq_seq)
        uniq_sequence_iterator = range(num_uniq_seq)

        if not weights_per_sequence:
            weights_per_sequence = np.ones((num_uniq_seq,))
            
        # ===== 5. Process Sequences =====
        for ind_seq, proba in zip(uniq_sequence_iterator, weights_per_sequence):
            # Get mutated residues (identical to calc_energy_par)
            mutated_res = mutants_sequences[ind_seq].getArray()[self.q_ind]
            
            try:
                mutated_res_indices = np.array([alphabet_dict[res] for res in mutated_res], dtype=int)
            except KeyError as e:
                invalid_res = str(e).strip("'")
                print(f"Problem: '{invalid_res}' not found in alphabet_bytes")
                raise
    
            # Convert to tensors (identical to calc_energy_par)
            freq_gen_indices = torch.from_numpy(mutated_res_indices).to('cuda')
            pos_indices = torch.arange(self.len_q_ind).to('cuda')
    
            # Frequency contribution (identical calculation)
            freq_contrib_hamiltoninan = (
                frequencies_i[freq_gen_indices, pos_indices] * 
                reshaped_pre_freq_matrix[freq_gen_indices, pos_indices]).sum()
    
            # Coupling contributions (identical calculation)
            coupling_contrib_hamiltonian = torch.tensor([0]).double().to('cuda')
            if couplings_i is not None and reshaped_pre_coupling_cubix is not None:
                coupling_contrib_hamiltonian = (
                    couplings_i[:, pos_indices, :, freq_gen_indices] * 
                    reshaped_pre_coupling_cubix[:, pos_indices, :, freq_gen_indices]).sum()
    
            coupling_contrib_hamiltonian_2 = (
                couplings_2_i[:, pos_indices, :, freq_gen_indices] * 
                reshaped_pre_coupling_tensor[:, pos_indices, :, freq_gen_indices]).sum()
    
            # Final Hamiltonian (identical formula)
            hamiltonian = (
                freq_contrib_hamiltoninan + 
                coupling_contrib_hamiltonian + 
                coupling_contrib_hamiltonian_2 - 
                hamiltonian_ref
            ) / self.T
    
            individual_hamiltonians[ind_seq] = hamiltonian.cpu().item()*proba
    
        return (individual_hamiltonians,)
        
##############################################################################################################################################################

    def simulation_QC(self, hmc_result:tuple, dump_dir='./'):

        n_samples = self.N_training_samples
        total_dyn_step = n_samples*self.steps_per_traj
        hamiltonian = hmc_result[1]
        log_likelihood = hmc_result[2]
        momentum_rescaling = hmc_result[3]
        temperature = hmc_result[4]
        x_index = list(range(total_dyn_step))
        equipartition_state_label = ""
        colors = ['r','g','b']
        param_labels = ['Lower bound (2.5%) ', 'Median ', 'Higher bound (97.5%) ']

        all_results_df = pd.DataFrame({'Hamiltonian': hamiltonian, 'momentum_rescaling': momentum_rescaling, 'temperature':temperature})
        log_liklihood_df = pd.DataFrame({'log_likelihood': log_likelihood})
        
        if self.force_equi_part:
            equipartition_state_label = "System forced to satisfy equipatition"
        else:
            equipartition_state_label = ""# "System NOT forced to satisfy equipatition"
        
        # HAMILTONIAN
        # Initialize the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        hamiltonian_plot = sns.lineplot(y= hamiltonian, x=x_index, label='Hamiltonian', color='black')
        hamiltonian_plot.set(xlabel="Simulation step", ylabel="Hamiltonian (A.U)", 
                             title=f"Profile of Hamiltonian function in the simulation.\n optimization trajectories = {n_samples}. {equipartition_state_label}")
        
        #hamiltonian_avrg = #np.mean(hamiltonian)
        quantity_str = 'Hamiltonian'
        all_results_df[f'{quantity_str} rolling mean'] = all_results_df[quantity_str].rolling(window=100).mean()
        rolling_mean = all_results_df[f'{quantity_str} rolling mean'].to_numpy()
        #sns.lineplot(y= rolling_mean, x=x_index, label=f'{quantity_str} rolling mean', ax=ax)
        hamiltonian_quant = np.quantile(hamiltonian, [0.025, 0.5, 0.975])
        #quant_and_avrg = np.append(hamiltonian_avrg, hamiltonian_quant)
        #quantity_str = 'Hamiltonian'
        #
        for quant, color, param_label in zip(hamiltonian_quant, colors, param_labels):
            param_label = param_label + quantity_str
            plt.axhline(y=quant, color=color, linestyle='--', linewidth=1, label=param_label)#, ax=ax)

        plt.tight_layout()
        hamiltonian_plot_path = os.path.join(dump_dir, "hamiltonian_plot.png")
        plt.savefig(hamiltonian_plot_path) 
        # LOG LIKELIHOOD\
        fig, ax = plt.subplots(figsize=(10, 6))
        log_likelihood_latex_expression = "- ELBO"#"E[-log P(S | {$\\theta$}, {$\\omega$}, {$\\varphi$})]"
        log_likelihood_plot = sns.lineplot(y=log_likelihood, x=list(range(len(log_likelihood))), label='log likelihood', color='black', ax=ax)
        log_likelihood_plot.set(xlabel="Simulation step", ylabel=log_likelihood_latex_expression, 
                                title=f"Profile of -ELBO of sampled sequences in the simulation.\n optimization trajectories = {n_samples}. {equipartition_state_label}. Sampling T = {self.T}")

        quantity_str = 'log_likelihood'
        log_liklihood_df[f'{quantity_str} rolling mean'] = log_liklihood_df[quantity_str].rolling(window=100).mean()
        rolling_mean = log_liklihood_df[f'{quantity_str} rolling mean'].to_numpy()
        sns.lineplot(y=rolling_mean, x=list(range(len(rolling_mean))), label=f'{quantity_str} rolling mean', ax=ax)
        log_likelihood_quant = np.quantile(log_likelihood, [0.025, 0.5, 0.975])
        
        for quant, color, param_label in zip(log_likelihood_quant, colors, param_labels):
            param_label = param_label + quantity_str
            plt.axhline(y=quant, color=color, linestyle='--', linewidth=1, label=param_label)

        plt.tight_layout()
        log_likelihood_plot_path = os.path.join(dump_dir, "log_likelihood_plot.png")
        plt.savefig(log_likelihood_plot_path)

        if self.force_equi_part:
            # MOMENTUM RESCALING
            fig, ax = plt.subplots(figsize=(10, 6))
            lamda_latex_expr = "$\lambda$"
            momentum_rescaling_plot = sns.lineplot(y=momentum_rescaling, x=x_index, label=lamda_latex_expr, color='black', ax=ax)
            momentum_rescaling_plot.set(xlabel="Simulation step", ylabel="Momentum rescaling factor (A.U)", 
                                        title=f"Profile of Momentum rescaling factor in the simulation.\n optimization trajectories = {n_samples}. {equipartition_state_label}. T = {self.T}")
            
            quantity_str = 'momentum_rescaling'
            all_results_df[f'{quantity_str} rolling mean'] = all_results_df[quantity_str].rolling(window=100).mean()
            rolling_mean = all_results_df[f'{quantity_str} rolling mean'].to_numpy()
            #sns.lineplot(y=rolling_mean, x=x_index, label=f'{quantity_str} rolling mean', ax=ax)
            momentum_rescaling_quant = np.quantile(momentum_rescaling, [0.025, 0.5, 0.975])
    
            for quant, color, param_label in zip(momentum_rescaling_quant, colors, param_labels):
                param_label = param_label + quantity_str
                plt.axhline(y=quant, color=color, linestyle='--', linewidth=1, label=param_label)
    
            momentum_rescaling_plot_path = os.path.join(dump_dir, "momentum_rescaling_plot.png")
            plt.tight_layout()
            plt.savefig(momentum_rescaling_plot_path)

        # TEMPERATURE
        fig, ax = plt.subplots(figsize=(10, 6))
        temperature_plot = sns.lineplot(y=temperature, x=x_index, label='Temperature', ax=ax)
        temperature_plot.set(xlabel="Simulation step", ylabel="HMC kinetic Temperature", 
                             title=f"Profile of kinetic Temperature during HMC.\n optimization trajectories = {n_samples}. {equipartition_state_label}.")        

        quantity_str = 'temperature'
        all_results_df[f'{quantity_str} rolling mean'] = all_results_df[quantity_str].rolling(window=1000).mean()
        rolling_mean = all_results_df[f'{quantity_str} rolling mean'].to_numpy()
        sns.lineplot(y=rolling_mean, x=x_index, label='HMC kinetic Temperature rolling mean', ax=ax)
        temperature_quant = np.quantile(temperature, [0.025, 0.5, 0.975])
        
        for quant, color, param_label in zip(temperature_quant, colors, param_labels):
            param_label = param_label + quantity_str
            plt.axhline(y=quant, color=color, linestyle='--', linewidth=1, label=param_label)
        
        plt.tight_layout()
        temperature_plot_path = os.path.join(dump_dir, "temperature_plot.png")
        plt.savefig(temperature_plot_path)
        
        csvfilename = os.path.join(dump_dir, 'mcmc_metrics.csv')
        all_results_df.to_csv(csvfilename) 
        csvfilename2 = os.path.join(dump_dir, 'mcmc_likelihood.csv')
        log_liklihood_df.to_csv(csvfilename2) 

##############################################################################################################################################################

    def mutate_protein(self, deletions:bool ,sequence, positions):

        if deletions:
            amino_acids = list("ACDEFGHIKLMNPQRSTVWY-")
        else:
            amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
            
        choices = [aa for aa in amino_acids]
    
        for pos in positions:
            current = sequence[pos]
            sequence[pos] = random.choice(choices)
        
        return (sequence,)

##############################################################################################################################################################
    def random_mutant_generator(self, random_msa_path:str, num_uniform_random_sequences=10e6):
        
        parsed_msa_alternative =  self.msa
        positions_to_mutate = self.q_ind
        ind_ref = self.ind_ref
        
        protein_seq = list(np.array(self.ref_seq, dtype=str))
        protein_seq_string = ''.join(protein_seq)
        mutated_seqs = []
        num_variants = num_uniform_random_sequences
        
        for i in range(num_variants):
            mutated_seq = self.mutate_protein(sequence=protein_seq, positions=positions_to_mutate, deletions=self.deletions)[0]#[0]
            mutated_seq_str = ''.join(mutated_seq)
            mutated_seqs.append(mutated_seq_str)
        
        ids = [f"random_mutant_{i+1}" for i in range(num_variants)]
    
        records = [SeqRecord(Seq(sequence), id=id_, description="") for sequence, id_ in zip(mutated_seqs, ids)]
        SeqIO.write(records, random_msa_path, "fasta")
        print('Random sample generated!')
        
        random_sample_ready = pr.parseMSA(random_msa_path, aligned=True)

        return (random_sample_ready,)
        
##############################################################################################################################################################
    def collect_MR_samples(self, where:str, generic_mutant_file_name:str, type_:str):
        #### Multi replica
        if type_ not in ['generated', 'random']:
                raise Exception("type_ must be one of ['generated', 'random']")
        else:
            if type_=='generated':
                collection_dir = 'training_samples'
            else:
                collection_dir = 'random_mutants'
        
        count = 0
        
        
        for dir in os.listdir(where):
            source = os.path.join(where, dir)
            final_sam_dir = os.path.join(source, collection_dir)
            
            if os.path.isdir(source) and dir[0:5]=='chain' and dir!='final_sampling_AVRG_PARAM' and os.path.isdir(final_sam_dir): 
                len_dir = len(os.listdir(final_sam_dir))
                print(final_sam_dir)
                
                if len_dir>0:
                    if len(dir)==7:
                        n_chain = dir[-1]
                    elif len(dir)==8:
                        n_chain = dir[-2:]
                    elif len(dir)==9:
                        n_chain = dir[-3:]
                    try:     
                        path_gen = os.path.join(source, collection_dir, generic_mutant_file_name)#f'Replica_{n_chain}_training_mutants.fasta')#final_sampling#IL2_IL2RB_mutants_pfam_scaled_FINAL_SAMPLE.fasta
                        generated_i = pr.parseMSA(path_gen, aligned=True)
                    
                        if not count:
                            generated = pr.MSA(generated_i)
                        else:
                            generated.extend(generated_i)
                            
                    except Exception as e:
                        print(f'Unable to process chain {n_chain} because of the following error : \n{e}')

                    count+=1
        
        return (generated,)
        
#############################################################################################################################################################
    def calc_ensemble_param(self, kronecker_matrix:tuple, where:str, reduced:bool, just_str_path=True):
        ## Important #### Multi replica
        output_path = where
        T = torch.tensor([self.T]).to('cuda')
        ref_seq = self.ref_seq
        ############################## Load MODEL PARAMETERS ##############################
        frequencies_i_avrg = torch.tensor([0]).to('cuda')
        couplings_i_avrg = torch.tensor([0]).to('cuda')
        couplings_2_i_avrg = torch.tensor([0]).to('cuda')
        chain_count = 0 
        expected_proba_mat_list = {}
        
        for chain in os.listdir(output_path):
            where_ = os.path.join(output_path, chain)
            
            if not os.path.isdir(where_) or chain=='.ipynb_checkpoints' or chain[:5]!='chain':
                continue
            else:
                ELBO_estimate_sample_i, previous_sample = 0, 0
                #final_sam_dir = os.path.join(where_, 'final_sampling')
                #if os.path.isdir(final_sam_dir):
                print(where_)
                output_path_params = os.path.join(where_, 'training_samples/csg_parameters.npy')

                try:
                    with open(output_path_params, 'rb') as f:
                        params_1 = np.load(f)
        
                        if not reduced:
                            params_2 = np.load(f)
                            couplings_i = torch.tensor(params_2).to('cuda')
                            couplings_i_avrg = couplings_i_avrg + couplings_i
                            
                        params_3 = np.load(f)
                        params_4 = np.load(f)
                    
                    frequencies_i = torch.tensor(params_1).to('cuda')
                    couplings_2_i = torch.tensor(params_3).to('cuda')
            
                    frequencies_i_avrg = frequencies_i_avrg + frequencies_i
                    couplings_2_i_avrg = couplings_2_i_avrg + couplings_2_i
        
                    expected_proba_mat = torch.tensor(params_4).to('cuda')
                    expected_proba_mat_list[f'{chain}'] = expected_proba_mat
                    chain_count+=1
                    
                except Exception as e:
                    print(f'Unable to process {dir} because of the following error : \n{e}')
                #else:
                   # continue
   
        frequencies_i_avrg  = frequencies_i_avrg * (1/chain_count)
        
        if not reduced:
            couplings_i_avrg = couplings_i_avrg * (1/chain_count)
            pre_coupling_cubix_shape = kronecker_matrix[-1]
            pre_coupling_cubix = torch.reshape(kronecker_matrix[3], pre_coupling_cubix_shape)#.cpu()
        else:
            couplings_i_avrg = None
            
        couplings_2_i_avrg = couplings_2_i_avrg * (1/chain_count)
        
        freq_matrix_dimension = kronecker_matrix[-2]
        pre_freq_matrix = torch.reshape(kronecker_matrix[2], freq_matrix_dimension)#.cpu()
        
        pre_coupling_4D_shape = kronecker_matrix[1]
        pre_coupling_tensor = torch.reshape(kronecker_matrix[0], pre_coupling_4D_shape)#.cpu()
        
        if not reduced:
            evol_param_tuple = (pre_freq_matrix, pre_coupling_tensor, pre_coupling_cubix)
        else:
            evol_param_tuple = (pre_freq_matrix, pre_coupling_tensor)

        ensemble_param_file = os.path.join(output_path, 'csg_ensemble_parameters.npy')

        try:
            self.write_param(output_path_params=ensemble_param_file, reduced=reduced, frequencies_i=frequencies_i_avrg, 
                             couplings_i=couplings_i_avrg, couplings_2_i=couplings_2_i_avrg, expected_proba_mat=None)
            print(f'Ensemble parameters saved : {ensemble_param_file}')
        except Exception as e:
            #print(e)
            raise e

        if not just_str_path:
            if not reduced:
                return (frequencies_i_avrg, couplings_i_avrg, couplings_2_i_avrg, expected_proba_mat_list, evol_param_tuple, freq_matrix_dimension)
            else:
                return (frequencies_i_avrg, couplings_2_i_avrg, expected_proba_mat_list, evol_param_tuple, freq_matrix_dimension)
        else:
            return ensemble_param_file
            
#######################################################################################################################################################
   
    def write_param(self, output_path_params:str, reduced:bool, frequencies_i, couplings_i, couplings_2_i, expected_proba_mat=None):

        with open(output_path_params, 'wb') as f:
            np.save(f, frequencies_i.cpu().detach().numpy())
            
            if not reduced:
                np.save(f, couplings_i.cpu().detach().numpy())
        
            np.save(f, couplings_2_i.cpu().detach().numpy())

            if expected_proba_mat is not None:
                np.save(f, expected_proba_mat.cpu().detach().numpy())

        print(f'Parameters saved! {output_path_params}')

#######################################################################################################################################################
    
    def read_replica_param(self, output_path_params:str, kronecker_matrix:tuple, reduced:bool):#, type='both'):
        
        ############################## Load MODEL PARAMETERS ##############################        
        try:
            with open(output_path_params, 'rb') as f:
                params_1 = np.load(f)

                if not reduced:
                    params_2 = np.load(f)
                    
                params_3 = np.load(f)

                try:
                    params_4 = np.load(f)
                except Exception as e:
                    print(f'ML probability matrix not found in {output_path_params}')
                    params_4 = None
            
        except Exception as e:
            raise e
        
        frequencies_i = torch.tensor(params_1)
        size_couplings_i = 0
        couplings_i = None
        
        freq_matrix_dimension = kronecker_matrix[-2]
        pre_freq_matrix = torch.reshape(kronecker_matrix[2], freq_matrix_dimension)#.cpu()

        if not reduced:
            pre_coupling_cubix_shape = kronecker_matrix[-1]
            pre_coupling_cubix = torch.reshape(kronecker_matrix[3], pre_coupling_cubix_shape)#.cpu()
        
        pre_coupling_4D_shape = kronecker_matrix[1]
        pre_coupling_tensor = torch.reshape(kronecker_matrix[0], pre_coupling_4D_shape)#.cpu()
        
        if not reduced:
            couplings_i = torch.tensor(params_2)
        
        couplings_2_i = torch.tensor(params_3)

        if params_4 is not None:
            expected_proba_mat = torch.tensor(params_4)
        else:
            expected_proba_mat = None
            
        if not reduced:
            evol_param_tuple = (pre_freq_matrix, pre_coupling_tensor, pre_coupling_cubix)
            stochastic_param_tuple = (frequencies_i, couplings_i, couplings_2_i)
        else:
            evol_param_tuple = (pre_freq_matrix, pre_coupling_tensor)
            stochastic_param_tuple = (frequencies_i, couplings_2_i)

        return(evol_param_tuple, stochastic_param_tuple, expected_proba_mat)
          
#############################################################################################################################################################        
    def Replica_sample(self, where:str, reduced:bool, generic_mutant_file_name:str, kronecker_matrix:tuple, output_path:str, 
                       bdm_sampling=10e6, bdm_sample=True, chain=00, type_='random'):

        T = self.T
        ref_seq = self.ref_seq
        to_mutate_alternative = self.q_ind
        kronecker_matrix_ = kronecker_matrix
        
        try:
            output_path_param_ = os.path.join(where, 'training_samples/csg_parameters.npy')
            params = self.read_replica_param(output_path_param_, kronecker_matrix_, reduced)
            print(output_path_param_)
        except Exception as e:
            print(e)
            output_path_param_ = os.path.join(where, 'csg_parameters.npy')
            print(f'Trying the following address for replica parameters: {output_path_param_}')
    
            try:
                params = self.read_replica_param(output_path_param_, kronecker_matrix_, reduced)
            except Exception as e:
                raise e
            
        evol_param_tuple = params[0]
        stochastic_param_tuple = params[1]
        expected_proba_mat = params[2]
        freq_matrix_dimension = kronecker_matrix[-2]
        frequencies_i = stochastic_param_tuple[0]
    
        if not reduced:
            couplings_i = stochastic_param_tuple[1]
            couplings_2_i = stochastic_param_tuple[2]
        else:
            couplings_2_i = stochastic_param_tuple[1]
            couplings_i = None
        
        if bdm_sample:
            sample_path = os.path.join(output_path, generic_mutant_file_name)
            sample = self.DirichletBoltzmannDist(q_ind=to_mutate_alternative, ref_seq=ref_seq, frequencies_i=frequencies_i, 
                                                 couplings_i=couplings_i, freq_matrix_dimension=freq_matrix_dimension, 
                                                 couplings_2_i=couplings_2_i, mode='sampling', n_mutant=bdm_sampling, 
                                                 T=torch.tensor(T).to('cuda'), evol_param_tuple=evol_param_tuple, 
                                                 expected_proba_mat=expected_proba_mat)
            generated_i = sample[0]
            pr.writeMSA(sample_path, generated_i)
            individual_hamiltonians = sample[-1]
        
        else:
            sample_path = os.path.join(where, f'final_sampling/{generic_mutant_file_name}')
            results = self.calc_energy_replica(output_path_params=output_path_param_, kronecker_matrix_=kronecker_matrix_, 
                                               reduced=reduced, msa_path=sample_path, chain=chain, type_=type_)
            generated_i = results[1]   #pr MSA
            individual_hamiltonians = results[0] #pd.DF
    
        return (generated_i, individual_hamiltonians)
    
#############################################################################################################################################################        
    def Ensemble_sample(self, where:str, reduced:bool, generic_mutant_file_name:str, kronecker_matrix:tuple, bdm_sampling=10e6, output_df_name=None):

        if output_df_name:
            if type(output_df_name)!=str:
                raise Exception("output_df_name variable must be a string, (default result dataframe name = 'AVRG_PARAM_unique_proba_matLL.csv')")
        else:
            output_df_name = 'AVRG_PARAM_unique_proba_matLL.csv'
            
        average_params = self.calc_ensemble_param(kronecker_matrix=kronecker_matrix, where=where, reduced=reduced, just_str_path=False)
        output_path = where
        to_mutate_alternative = self.q_ind
        ref_seq = self.ref_seq
        T = self.T
        frequencies_i_avrg = average_params[0]
        bdm_sampling = int(bdm_sampling)
        
        if not reduced:
            couplings_i_avrg = average_params[1]
            pre_coupling_cubix_shape = kronecker_matrix[-1]
            couplings_i_avrg = torch.reshape(couplings_i_avrg, pre_coupling_cubix_shape).to('cuda')
            couplings_2_i_avrg = average_params[2]
            expected_proba_mat_list = average_params[3]
            evol_param_tuple = average_params[4]
            freq_matrix_dimension = average_params[5]
        else:
            couplings_2_i_avrg = average_params[1]
            expected_proba_mat_list = average_params[2]
            evol_param_tuple = average_params[3]
            freq_matrix_dimension = average_params[4]
            couplings_i_avrg = None
            
        ELBO_dict = {}
        count=0
        pre_coupling_4D_shape = kronecker_matrix[1]
        frequencies_i_avrg = torch.reshape(frequencies_i_avrg, freq_matrix_dimension).to('cuda')
        couplings_2_i_avrg = torch.reshape(couplings_2_i_avrg, pre_coupling_4D_shape).to('cuda')
            
        for chain, expected_proba_mat in expected_proba_mat_list.items():
            sample = self.DirichletBoltzmannDist(q_ind=to_mutate_alternative, ref_seq=ref_seq, frequencies_i=frequencies_i_avrg, 
                                            couplings_i=couplings_i_avrg, freq_matrix_dimension=freq_matrix_dimension, 
                                               couplings_2_i=couplings_2_i_avrg, mode='sampling', n_mutant = bdm_sampling, 
                                    T=torch.tensor(T).to('cuda'), sample_i=1, i=1, evol_param_tuple=evol_param_tuple, 
                                                 expected_proba_mat=expected_proba_mat)    
            if count<1:
                sample_msa_i = pr.MSA(sample[0], title=f'Sample sequences from expected probability matrix from chain {chain} (average parameters)')
                unique_generated = pr.uniqueSequences(sample_msa_i[:,self.q_ind], seqid=1, turbo=True)
                generated_i_unique = sample_msa_i[unique_generated]
                sample_msa = generated_i_unique 
            else:
                sample_msa_i = pr.MSA(sample[0], title=f'Sample sequences from expected probability matrix from chain {chain} (average parameters)')
                unique_generated = pr.uniqueSequences(sample_msa_i[:,self.q_ind], seqid=1, turbo=True)
                generated_i_unique = sample_msa_i[unique_generated]
                sample_msa.extend(generated_i_unique.getArray())
                
            previous_sample = sample[1]
            ELBO_estimate_sample_i = previous_sample

            ELBO_estimate = ELBO_estimate_sample_i#/bdm_sampling
            ELBO_dict[chain] = ELBO_estimate
            print(f'- ELBO estimate of {bdm_sampling} sequences generated = {ELBO_estimate}')

            log_likelihood_array = sample[-1]
            #unique_generated = pr.uniqueSequences(sample_msa_i, seqid=1, turbo=True)
            log_likelihood_array_unique = log_likelihood_array[unique_generated]
            #generated_i_unique = sample_msa_i[unique_generated]

            if not count:
                energy_df = pd.DataFrame({'log_likelihood':-log_likelihood_array_unique, 'Chain':[chain]*generated_i_unique.numSequences()})
            else:
                energy_df_i = pd.DataFrame({'log_likelihood':-log_likelihood_array_unique, 'Chain':[chain]*generated_i_unique.numSequences()})
                energy_df = pd.concat([energy_df, energy_df_i], axis=0)

            count+=1
        
        output_path_samples = os.path.join(output_path, 'final_sampling_AVRG_PARAM_unique_proba_mat')
        os.system(f'mkdir {output_path_samples}')
        sample_path = os.path.join(output_path_samples, f'AVRG_PARAM_unique_proba_mat_{generic_mutant_file_name}')
        pr.writeMSA(sample_path, sample_msa)
        LL_df_path = os.path.join(output_path_samples, output_df_name)
        energy_df.to_csv(LL_df_path)
                         
        return (sample_msa, energy_df)
        
#############################################################################################################################################################                
    def Multi_replica_sample(self, kronecker_matrix:tuple, where:str, reduced:bool, generic_mutant_file_name:str, bdm_sampling=10e5, bdm_sample=True, type_='random', output_df_name=None):
        
        if type(type_)!=str:
            raise Exception("type_ variable must be one of :'random', 'generated', 'actual'")
        elif type_ not in ['random', 'generated', 'actual']:
            raise Exception("type_ variable must be one of :'random', 'generated', 'actual'")
        if output_df_name:
            if type(output_df_name)!=str:
                raise Exception("output_df_name variable must be a string, (default result dataframe name = per_chain_sampled_seq_log_likelihood.csv)")
        else:
            output_df_name = 'per_chain_sampled_seq_log_likelihood.csv'
            
        #### Multi replica
        count = 0
        freq_matrix_dimension = kronecker_matrix[-2]
        bdm_sampling = int(bdm_sampling)
        
        if not reduced:
            pre_coupling_cubix_shape = kronecker_matrix[-1]
            
        pre_coupling_4D_shape = kronecker_matrix[1]
        
        for dir in os.listdir(where):
            source = os.path.join(where, dir)
            final_sam_dir = os.path.join(source, 'training_samples')
            
            if os.path.isdir(source) and dir[0:5]=='chain' and dir!='.ipynb_checkpoints' and dir!='final_sampling_AVRG_PARAM' and os.path.isdir(final_sam_dir):
                if len(dir)==7:
                    n_chain = dir[-1]
                elif len(dir)==8:
                    n_chain = dir[-2:]
                try:
                    replica_sample = self.Replica_sample(where=source, reduced=reduced, generic_mutant_file_name=generic_mutant_file_name,
                                                         kronecker_matrix=kronecker_matrix, output_path=final_sam_dir, bdm_sampling=bdm_sampling, 
                                                         bdm_sample=bdm_sample, chain=int(n_chain), type_=type_)
                    generated_i = replica_sample[0]
                    unique_generated = pr.uniqueSequences(generated_i[:,self.q_ind], seqid=1, turbo=True)
                    generated_i_unique = generated_i[unique_generated]
                    log_likelihood_array = replica_sample[1]
                    
                    if not count:
                        if bdm_sample:
                            log_likelihood_array_unique = log_likelihood_array[unique_generated]
                            energy_df = pd.DataFrame({'log_likelihood':-log_likelihood_array_unique, 'Chain':[n_chain]*generated_i_unique.numSequences()})
                        else:
                            energy_df = log_likelihood_array
                    else:
                        if bdm_sample:
                            log_likelihood_array_unique = log_likelihood_array[unique_generated]
                            energy_df_i = pd.DataFrame({'log_likelihood':-log_likelihood_array_unique, 'Chain':[n_chain]*generated_i_unique.numSequences()})
                        else:
                            energy_df_i = log_likelihood_array
                            
                        energy_df = pd.concat([energy_df, energy_df_i], axis=0)
                        
                    count+=1
                        
                except OSError as e:
                    raise e
        
        energy_df_path = os.path.join(where, output_df_name)
        energy_df.to_csv(energy_df_path) 
        print(f'Find the per chain loglikelihood estimates at :\n{energy_df_path}')

############################################################################################################################################################# 
    def calc_energy_replica(self, output_path_params:str, kronecker_matrix_:tuple, reduced:bool, chain:int, type_='random', msa_path=None):

        if type(type_)!=str:
            raise Exception("type_ variable must be one of :'random', 'generated', 'actual'")
        elif type_ not in ['random', 'generated', 'actual']:
            raise Exception("type_ variable must be one of :'random', 'generated', 'actual'")
        if msa_path:
            random_msa_i = pr.parseMSA(msa_path, aligned=True)
        elif not msa_path and type_=='actual':
            random_msa_i = self.msa
        else:
            raise Exception('Specify the path to the MSA with option msa_path')
        
        if type(type_)!=str:
            raise Exception("type_ variable must be one of :'random', 'generated', 'actual'")
        else:
            if type_ not in ['random', 'generated', 'actual']:
                raise Exception("type_ variable must be one of :'random', 'generated', 'actual'")

        #random_msa_i = pr.parseMSA(msa_path, aligned=True)
        freq_matrix_dimension = kronecker_matrix_[-2]
        pre_coupling_4D_shape = kronecker_matrix_[1]
        params = self.read_replica_param(output_path_params=output_path_params, kronecker_matrix=kronecker_matrix_, reduced=reduced)
        evol_param_tuple = params[0]
        stochastic_param_tuple = params[1]
        expected_proba_mat = params[2]
        
        frequencies_i = stochastic_param_tuple[0]

        if not reduced:
            couplings_i = stochastic_param_tuple[1]
            couplings_2_i = stochastic_param_tuple[2]
        else:
            couplings_2_i = stochastic_param_tuple[1]
            couplings_i = None

        frequencies_i = torch.reshape(frequencies_i, freq_matrix_dimension).to('cuda')

        if not reduced:
            pre_coupling_cubix_shape =  kronecker_matrix_[-1]
            couplings_i = torch.reshape(couplings_i,  pre_coupling_cubix_shape).to('cuda')
        else:
            couplings_i = None
            
        couplings_2_i = torch.reshape(couplings_2_i, pre_coupling_4D_shape).to('cuda')

        log_likelihood_array_rndm = self.calc_energy_batch(frequencies_i=frequencies_i, couplings_i=couplings_i, 
                                                           couplings_2_i=couplings_2_i, msa=random_msa_i, 
                                                           evol_param_tuple=evol_param_tuple)[0]

        len_log_likelihood_array_rndm = len(log_likelihood_array_rndm)
        energy_df_rand_i = pd.DataFrame({'log_likelihood':log_likelihood_array_rndm, 'Chain':[chain]*len_log_likelihood_array_rndm,
                                        'label':[type_]*len_log_likelihood_array_rndm})

        return (energy_df_rand_i, random_msa_i)

#############################################################################################################################################################        
    def calc_energy_multireplica(self, where:str, reduced:bool, kronecker_matrix:tuple, type_='random', generic_mutant_file_name=None, 
                                 generic_output_fasta=None, generic_output_csv=None, write_fasta_multireplica=True):

        if generic_output_csv and type(generic_output_csv)!=str:
            raise Exception("If specified generic_output_csv must be a string (generic name of output dataframe with loglikelihood calculations per chain")
        elif not generic_output_csv:
            generic_output_csv = 'per_chain_sampled_seq_log_likelihood.csv'

        if generic_output_fasta and type(generic_output_fasta)!=str:
            raise Exception("If specified generic_output_fasta variable must be a string (generic name of concatenated random MSA from all replicas")
        elif not generic_output_fasta:
            generic_output_fasta = 'all_random_msa.fasta'
            
        if type_ not in ['random', 'generated', 'actual']:
            raise Exception("type_ variable must be one of : ['random', 'generated', 'actual']")
        if type_=='random':
            if not generic_mutant_file_name:
                mutant_path = 'random_mutants/random_mutants.fasta'
            else:
                if type(generic_mutant_file_name)!=str:
                    raise Exception("generic_mutant_file_name must be a string (generic name of random sequences)")
                else:
                    mutant_path = f'random_mutants/{generic_mutant_file_name}'
        else:
            if not generic_mutant_file_name:
                raise Exception("generic_mutant_file_name must be specified when type_='generated' or 'actual'")
            if type(generic_mutant_file_name)!=str:
                raise Exception("generic_mutant_file_name must be a string (generic name of sampled sequences by each replica)")
                
            mutant_path = f'final_sampling/{generic_mutant_file_name}'
            
        i = 0
        output_path = where

        for chain in os.listdir(output_path):
            print(chain)
            where_ = os.path.join(output_path, chain)
            
            if not os.path.isdir(where_) or chain=='.ipynb_checkpoints' or chain[:5]!='chain':
                continue
            else:
                print(where)

                try:
                    final_sam_dir = os.path.join(where_, mutant_path)
                    output_path_param = os.path.join(where_, 'training_samples/csg_parameters.npy')
                    print(output_path_param)

                    chain_num = int(chain[5:])

                    if type_=='actual':
                        final_sam_dir=None
                    
                    results = self.calc_energy_replica(output_path_params=output_path_params, msa_path=final_sam_dir,
                                                       kronecker_matrix_=kronecker_matrix, reduced=reduced, 
                                                       chain=chain_num, type_=type_)[0]
                    random_msa_i = results[1]
                    energy_df_rand_i = results[0]
                    
                    if i:
                        #energy_df_rand_i = pd.DataFrame({'log_likelihood':log_likelihood_array_rndm, 'Chain':[chain_num]*len_log_likelihood_array_rndm})
                        energy_df_rand = pd.concat([energy_df_rand, energy_df_rand_i], axis=0)
                    else:
                        energy_df_rand = energy_df_rand_i
                        #pd.DataFrame({'log_likelihood':log_likelihood_array_rndm, 'Chain':[chain_num]*len_log_likelihood_array_rndm})
                    if i<1:
                        random_msa = random_msa_i
                    else:
                        random_msa.extend(random_msa_i.getArray())
                        
                    i+=1
                
                except Exception as e:
                    raise e
                    #print(e)
        print(f'{i} replicas processed')

        if write_fasta_multireplica:
            if generic_output_fasta:
                path_gen = os.path.join(output_path, generic_output_fasta)
            else:
                path_gen = os.path.join(output_path, 'all_multireplica_sequences.fasta')

            try:
                pr.writeMSA(path_gen, random_msa)
                print(f'{type_} multireplica sample saved! {path_gen }')
            except Exception as e:
                print(e)

        try:
            if generic_output_csv:
                energy_df_path = os.path.join(where, generic_output_csv)
            else:
                energy_df_path = os.path.join(where, 'per_chain_sampled_seq_log_likelihood.csv')
    
            energy_df_rand.to_csv(energy_df_path)
            print(f'{type_} multireplica log probabilities saved! {path_gen}')
        
        except Exception as e:
            print(e)
            
#############################################################################################################################################################  
    def make_af3_json_multireplica(self, output_path:str, confidence_threshold:float, template_file_dir:str, generic_mutant_file_name=None, 
                                   energy_csv='per_chain_sampled_seq_log_likelihood.csv', type_='random', hits_filename=None,
                                  model_type = 'AF3_models'):
        if type(model_type)!=str:
            raise Exception("model_type must be a string")
        if hits_filename and type(hits_filename)!=str :
            raise Exception("hits_filename must be a string (path to hits sequences) or keep default")
        if 0>=confidence_threshold or confidence_threshold>=1:
            raise Exception("confidence_threshold must be in ]0,1[")
        if type(type_)!=str or type_ not in ['random', 'generated']:
            raise Exception("type_ variable must be one of : ['random', 'generated']")
                
        generated_df = pd.read_csv(os.path.join(output_path, energy_csv))
        i = 0
        
        for chain in os.listdir(output_path):
            where = os.path.join(output_path, chain)
            
            if not os.path.isdir(where) or chain=='.ipynb_checkpoints' or chain[:5]!='chain':
                continue
            else:
                if type_=='random':
                    final_sam_dir = os.path.join(where, generic_mutant_file_name)
                else:
                    final_sam_dir = os.path.join(where, f'final_sampling/{generic_mutant_file_name}')
                    
                print(final_sam_dir)
                
                try:
                    generated_i = pr.parseMSA(final_sam_dir, aligned=True)
                    
                    if i<1:
                        generated = generated_i
                    else:
                        generated.extend(generated_i.getArray())
                    i+=1
                except Exception as e:
                    print(e)

        if not generic_mutant_file_name:
            concat_msa_path = os.path.join(output_path, 'all_generated_msa.fasta')
        else:
            concat_msa_path = os.path.join(output_path, generic_mutant_file_name)

        ####CLEAN
        X = np.unique(np.where(generated.getArray()==b'X')[0])
        clean_parsed_msa_alternative_ind = list(range(generated.numSequences()))
        clean_parsed_msa_alternative_ind = np.where(np.array([x if x not in X else None for x in clean_parsed_msa_alternative_ind])!=None)[0]
        generated = generated[clean_parsed_msa_alternative_ind,:]
        pr.writeMSA(concat_msa_path, generated)
            
        print('Start!')
        df_all_AVRG = pd.read_csv(os.path.join(output_path, energy_csv), index_col=0)
        print('Alright!')
        df_rand = df_all_AVRG.loc[df_all_AVRG['label']==f'{type_}']

        print(f'{type_} hit confidence_threshold = ', confidence_threshold)
        random_hit_threshold = df_rand.loc[:, 'log_likelihood'].quantile(confidence_threshold)
        
        rand_hit_ind = df_rand['log_likelihood']> random_hit_threshold
        random_hits = df_rand.loc[rand_hit_ind,:]
        random_hit_ind  = np.array(random_hits.index)
        #print(os.path.join(output_path, fasta_file))
        generated = generated[random_hit_ind,:]
        unique_generated = pr.uniqueSequences(generated_i, seqid=1, turbo=True)
        generated = generated[unique_generated]

        if not hits_filename:
            hits_filename = os.path.join(output_path, f'{type_}_hits.fasta')

        hits_file_path = os.path.join(output_path, hits_filename)
        
        pr.writeMSA(hits_file_path, generated)
        where_results = output_path
        template_path = os.path.join(output_path, 'af3_input.json')
        
        chain_dir = where_results
        af_dir = os.path.join(chain_dir, model_type)
        os.system(f'mkdir {af_dir}')
        
        af_models_dir = os.path.join(af_dir, 'json')
        os.system(f'mkdir {af_models_dir}')
        
        af_models_prediction =  os.path.join(af_dir, 'models')
        os.system(f'mkdir {af_models_prediction}')
        
        #path_gen = os.path.join(chain_dir, fasta_file)#f'Replica_{variable_value}_training_mutants.fasta')
        #generated_i = pr.parseMSA(path_gen)
        
        for index, sequence in enumerate(generated): 
        #for index in top1000_index:
            #sequence = generated_i[index]
            sequence_str = str(sequence)
            sequence_str_nogap = sequence_str.replace('-', '')
                
            with open(f"{template_file_dir}", "r") as f:
                data = json.load(f)
                data.pop("userCCD")
                data.pop("bondedAtomPairs")
        
                ###################
                #del data["sequences"][0]
                new_protein_data = {"protein":{"id":"IL",
                            "sequence":sequence_str_nogap[:153-20],
                            }}#"pairedMsaPath":paired_il2_il2rb
                data["sequences"][0] = new_protein_data
                ###############
                new_protein_data = {"protein":{"id": "ILRB", 
                            "sequence":sequence_str_nogap[153-20:],
                            }}#"pairedMsaPath":paired_il2_il2rb
                data["sequences"][1] = new_protein_data
                ###############
                del data["sequences"][2:]
                ###############
                data["version"] = 1
                ###############
                data["name"] = f"IL_ILRB_{index}"
        
            output_json = os.path.join(af_models_dir, f'af3_complex_{index}.json')
            
            with open(output_json, "w") as f:
                json.dump(data, f, indent=2)