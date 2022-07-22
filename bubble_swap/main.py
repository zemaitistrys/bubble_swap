import qiskit
import scipy as sp
import pandas as pd
import numpy as np
from numpy import pi
import math as mt
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ
from qiskit.circuit import Clbit, Qubit
from qiskit.providers.aer.noise import NoiseModel

# backbone code for playing around with "bubble swap" - a two-qubit operation that acts as
# a swap gate if one of the two qubits is initially in a zero state, with the possibility
# to perform postselection after the swap, by using the information that without noise
# the state that swapped with |0> should now be IN |0>. The advantage over regular swap
# is fewer CNOTs and the additional reduction of errors from postselection.This could be
# used for novel circuit compilers, but my (limited) numerics suggest it's not so good.

class GridCircuit:
    
    # defines a circuit with grid connectivity (unless specified otherwise),
    # 2D definition of qubits (qubit defined by x,y indices not just a single index),
    # wrappers for gates to account for this (qiskit gates take in single index),
    # and fully integrated bubble swap 'gate' with postselection capabilities
    # (fully integrated in the sense that the user never has to directly deal with
    # the mid-circuit-measurements and postselect based on their outcome, it is done
    # by the code as long as the user passes postselect = True)
    # (also mid-circuit-measurement error mitigation but afaik it's fundamentally wrong)
    
    def __init__(self, provider, x = None, y = None, coupling_map = None, no_qubits = None, no_bits = 2, noise_model = 0, max_no_swaps = 10, mit_matrices = None):
        
        # (x,y) - size of the grid
        # either (x,y) or no_qubits should be specified; if neither, do 5x5 grid, if both, overwrite no_qubits
        if x == None and y == None and no_qubits == None:
            x = 5
            y = 5
        if x != None and y != None and no_qubits != None:
            no_qubits = x * y  
            
        self.x = x
        self.y = y
        
        if no_qubits == None:
            self.no_qubits = x * y
            
        if coupling_map == None:
            self.coupling_map = self.grid_coupling_map(self.x, self.y)
        
        # by default we take a noise model from ibm perth - we need one
        # to compare bubble swap to a regular swap in terms of fidelity
        if noise_model == 0:
            self.noise_model = NoiseModel.from_backend(provider.get_backend('ibm_perth'))
        else:
            self.noise_model = noise_model
        
        # size of classical register which will contain bits measured into when bubble swapping
        self.max_no_swaps = max_no_swaps
        
        self.no_bits = no_bits # data bits (the idea is the user never has to deal with postselection
        #so we give the name no_bits to the number of data bits, even though the total number of bits
        # in the circuit is no_bits + max_no_swaps)
        
        self.total_no_bits = self.no_bits + self.max_no_swaps
        
        # initialising registers for circuit
        self.qreg = QuantumRegister(self.no_qubits)
        self.data_clreg = ClassicalRegister(self.no_bits)
        self.post_clreg = ClassicalRegister(self.max_no_swaps)
        self.swap_counter = 0 # to track which bit to measure into
        self.circuit = QuantumCircuit(self.qreg, self.data_clreg, self.post_clreg)
        self.backend = Aer.get_backend('qasm_simulator')
        
        # for mitigation; deprecated by virtue of it being wrong
        if mit_matrices is None:
            self.mit_matrices = self.mit_matrices()
        else:
            self.mit_matrices = mit_matrices
            
        self.measured_bits = [-1]*(self.no_bits+self.max_no_swaps) # i'th member of list represents which qubit was
        # measured into i'th classical bit (HERE 0'th bit is at the START); if -1, no qubit was measured into it.
        # this is used for mitigation. 
        
    def convert_coords(self, x_qubit,y_qubit):
        # convert 2D coordinates of qubit into 1D qubit index
        return y_qubit * self.x + x_qubit
    
    def grid_coupling_map(self, x,y):
        # creates coupling map of a square qubit grid of size x * y
        coupling_map = []

        # add horizontal coupling
        for i in range(y):
            for j in range(x-1):
                coupling_map.append([x*i+j, x*i+j+1])
                coupling_map.append([x*i+j+1, x*i+j])

        # add vertical coupling
        for i in range(y-1):
            for j in range(x):
                coupling_map.append([x*i+j,x*(i+1)+j])
                coupling_map.append([x*(i+1)+j,x*i+j])

        return coupling_map
    
    def gate(self, name, qubit1, qubit2 = None, parameters = None):
        
        # wraps gates in a function able to take in 2D coords
        # but can take in qubit index as well
        # e.g. gcirc.gate('rx', [0,1], parameters = 0.1) implements an rx gate by 0.1 rad
        # on the qubit [0,1]
        
        gates = {
            'cx' : self.circuit.cx,
            'rz' : self.circuit.rz,
            'ry' : self.circuit.ry,
            'rx' : self.circuit.rx,
            'crx' : self.circuit.crx,
            'cry' : self.circuit.cry,
            'crz' : self.circuit.crz,
            'h' : self.circuit.h,
            'x' : self.circuit.x,
            'y' : self.circuit.y,
            'z' : self.circuit.z,
            'swap' : self.circuit.swap,
            'u' : self.circuit.u,
           # can add more 
        }
        
        if type(qubit1) is not int:
            qubit1 = self.convert_coords(qubit1[0], qubit1[1])
            
        if qubit2 is not None and type(qubit2) is not int:
            qubit2 = self.convert_coords(qubit2[0], qubit2[1])
            
        if parameters is not None and (type(parameters) is int or type(parameters) is float):
            parameters = [parameters]
            
        if parameters is None and qubit2 is None:
            gates[name](qubit1)
        
        if parameters is None and qubit2 is not None:
            gates[name](qubit1, qubit2)
            
        if parameters is not None and qubit2 is None:
            gates[name](*parameters, qubit1)
            
        if parameters is not None and qubit2 is not None:
            gates[name](*parameters, qubit1, qubit2)
            
        return
    
    def measure(self, qubit, clbit):
        # qiskit measurement wrapped in a function to accept 2D coords, and also
        # to keep track of which qubit was measured in which bit;
        # this was planned to be used for error mitigation purposes
        if type(qubit) is not int:
            qubit = self.convert_coords(qubit[0], qubit[1])
        self.circuit.measure(qubit, self.data_clreg[clbit])
        self.measured_bits[self.total_no_bits-1-clbit] = qubit # for mitigation
        return

    
    def bswap(self, data_qubit, zero_qubit, postselect = True):
        # performs the bubble swap and measures the qubit that previously had the data
        # to postselect on whether it's found in |0>
        self.gate('cx', data_qubit, zero_qubit)
        self.gate('cx', zero_qubit, data_qubit)
        if postselect:
            if data_qubit is not int:
                data_qubit = self.convert_coords(data_qubit[0], data_qubit[1])
            self.circuit.measure(data_qubit, self.post_clreg[self.swap_counter])
            self.measured_bits[self.max_no_swaps-1-self.swap_counter] = data_qubit # for mitigation
            self.swap_counter += 1
        return
    
    def postselect(self, counts):
        # postselects counts with bitstrings whose bits corresponding to 
        # bubble swap measurements are all in 0
        res = {}
        for count in counts:
            if count[0:len(count)-self.no_bits] == '0'*(len(count)-self.no_bits):
                res[count[len(count)-self.no_bits:]] = counts[count]

        return res

    def not_postselect(self, counts):
        
        # removes bits reserved for postselection from bitstrings
        # i.e. "traces" over the postselection bits
        res = {}
        for count in counts:
            cut_count = count[len(count)-self.no_bits:]
            if cut_count in res:
                res[cut_count] += counts[count]
            else:
                res[cut_count] = counts[count]

        return res
    
    def run(self, shots, postselect = True, mitigate = True, probs = True):
        
        # central function for running a grid circuit, which executes the job
        # corresponding to the circuit, calls mitigation and/or postselection functions
        # and obtains probabilities from counts
        
        job = execute(self.circuit,
                      backend = self.backend,
                      shots = shots,
                      noise_model = self.noise_model,
                      coupling_map = self.coupling_map
                      )
        raw_counts = job.result().get_counts() # this has spaces between every postselection bit, which we better remove
        
        counts = {}
        
        # sometimes spaces appear between register, removing it to more easily keep track of
        # indices
        for count in raw_counts:
            counts[count.replace(' ', '')] = raw_counts[count]
        
        if mitigate:
            counts = self.mitigate(counts)
            
        if postselect:
            counts = self.postselect(counts)
        else:
            counts = self.not_postselect(counts)
        
        # converting to probabilities
        if probs:
            total_counts = 0
            probs = {}
            for count in counts:
                total_counts += counts[count]
            for count in counts:
                probs[count] = counts[count] / total_counts
            return probs
        else:
            return counts
    
    def mit_matrices(self, shots = 10000):
        # creates list of inverted measurement calibration matrices for each qubit
        # i.e. (p0, p1)_observed = M * (p0, p1)_prepared, where M is calibration matrix; we store M^-1 in list 
        res = []
        for i in range(self.no_qubits):
            
            circ0 = QuantumCircuit(self.no_qubits, 1)
            circ0.measure(i, 0)
            counts0 = execute(circ0,
                      backend = self.backend,
                      shots = shots,
                      noise_model = self.noise_model,
                      coupling_map = self.coupling_map
                      ).result().get_counts()
            if '0' not in counts0:
                counts0['0'] = 0
            if '1' not in counts0:
                counts0['1'] = 0
            p00 = counts0['0'] / (counts0['0']+counts0['1'])
            p10 = counts0['1'] / (counts0['0']+counts0['1'])
            
            circ1 = QuantumCircuit(self.no_qubits, 1)
            circ1.x(i)
            circ1.measure(i, 0)
            counts1 = execute(circ1,
                      backend = self.backend,
                      shots = shots,
                      noise_model = self.noise_model,
                      coupling_map = self.coupling_map
                      ).result().get_counts()
            if '0' not in counts1:
                counts1['0'] = 0
            if '1' not in counts1:
                counts1['1'] = 0
            p01 = counts1['0'] / (counts1['0']+counts1['1'])
            p11 = counts1['1'] / (counts1['0']+counts1['1'])
            
            M = np.array([[p00,p01],[p10,p11]])
            
            M_inv = np.linalg.inv(M)
            
            res.append(M_inv)
        
        return res
    
    def remove_bitstrings(self, counts):
        # find postselection bits which were measured into and only keep those bits
        measured_bits_indices = []
        new_measured_bits = []
        for i in range(self.max_no_swaps):
            qubit = self.measured_bits[i]
            if qubit != -1:
                measured_bits_indices.append(i)
                new_measured_bits.append(qubit)
        for i in range(self.max_no_swaps, self.total_no_bits):
            qubit = self.measured_bits[i]
            measured_bits_indices.append(i)
            new_measured_bits.append(qubit)
        self.measured_bits = new_measured_bits
        res = {}
        for count in counts:
            new_count = ''
            for index in measured_bits_indices:
                new_count = new_count + count[index]
            res[new_count] = counts[count]
        
        return res
                
        
        
    def mitigate(self, counts):
        # remove unused bits from bitstring, and correspondingly change measured_bits
        new_counts = self.remove_bitstrings(counts)
        id_matrix = np.array([[1, 0], [0, 1]])
        M_inv = np.array([1])
        for qubit in self.measured_bits:
            # read through which qubit each classical bit corresponds to and tensor product with the right
            # inverse calibration matrix based on that
            if qubit == -1:
                M_inv = np.kron(M_inv, id_matrix)
            else:
                M_inv = np.kron(M_inv, self.mit_matrices[qubit])
        # create array out of unmitigated counts (vector to multiply the matrix with)
        unmitigated_counts_array = np.zeros(2**len(self.measured_bits))
        for i in range(len(unmitigated_counts_array)):
            i_bin = format(i, '0'+str(len(self.measured_bits))+'b')
            if i_bin in new_counts:
                unmitigated_counts_array[i] = new_counts[i_bin]
        print(unmitigated_counts_array)
        # multiply matrix with vector to get mitigated counts
        mitigated_counts_array = np.matmul(M_inv, unmitigated_counts_array)
        
        #convert back to dictionary
        mitigated_counts_dict = {}
        for i in range(len(unmitigated_counts_array)):
            i_bin = format(i, '0'+str(len(self.measured_bits))+'b')
            mitigated_counts_dict[i_bin] = mitigated_counts_array[i]
        
        return mitigated_counts_dict