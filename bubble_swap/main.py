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

class GridCircuit:
    def __init__(self, provider, x = None, y = None, coupling_map = None, no_qubits = None, no_bits = 2, noise_model = None, max_no_swaps = 10, mit_matrices = None):
        
        # either (x,y) or no_qubits should be specified; if neither, do 5x5 grid, if both, overwrite no_qubits
        if x == None and y == None and no_qubits == None:
            x = 5
            y = 5
        if x != None and y != None and no_qubits != None:
            no_qubits = x * y  
            
        self.x = x
        self.y = y
        self.no_bits = no_bits # data bits (the idea is the user never has to deal with postselection so we give the name
        # no_bits to the number of data bits, even though the total number of bits in the circuit is no_bits + max_no_swaps)
        if no_qubits == None:
            self.no_qubits = x * y
            
        if coupling_map == None:
            self.coupling_map = self.grid_coupling_map(self.x, self.y)
        
        if noise_model == None:
            self.noise_model = NoiseModel.from_backend(provider.get_backend('ibm_perth'))
        else:
            self.noise_model = noise_model
        
        self.max_no_swaps = max_no_swaps
        self.total_no_bits = self.no_bits + self.max_no_swaps
        self.qreg = QuantumRegister(self.no_qubits)
        self.data_clreg = ClassicalRegister(self.no_bits)
        self.post_clreg = ClassicalRegister(self.max_no_swaps)
        self.swap_counter = 0 # to track which bit to measure into
        self.circuit = QuantumCircuit(self.qreg, self.data_clreg, self.post_clreg)
        self.backend = Aer.get_backend('qasm_simulator')
        
        if mit_matrices is None:
            self.mit_matrices = self.mit_matrices()
        else:
            self.mit_matrices = mit_matrices
            
        self.measured_bits = [-1]*(self.no_bits+self.max_no_swaps) # i'th member of list represents which qubit was
        # measured into i'th classical bit (HERE 0'th bit is at the START); if -1, no qubit was measured into it. this is used for mitigation. 
        
    def convert_coords(self, x_qubit,y_qubit):
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
        if type(qubit) is not int:
            qubit = self.convert_coords(qubit[0], qubit[1])
        self.circuit.measure(qubit, self.data_clreg[clbit])
        self.measured_bits[self.total_no_bits-1-clbit] = qubit # for mitigation
        return
        
    
    def cnot(self, control, target):
        # input: either ints, or list/array/tuple
        if type(control) is int and type(target) is int:
            self.circuit.cnot(control, target)
        else:
            control_qubit = self.convert_coords(control[0], control[1])
            target_qubit = self.convert_coords(target[0], target[1])
            self.circuit.cnot(control_qubit, target_qubit)
        return
    
    def bswap(self, data_qubit, zero_qubit, postselect = True):
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
        res = {}
        for count in counts:
            if count[0:len(count)-self.no_bits] == '0'*(len(count)-self.no_bits):
                res[count[len(count)-self.no_bits:]] = counts[count]

        return res

    def not_postselect(self, counts):
        res = {}

        for count in counts:
            cut_count = count[len(count)-self.no_bits:]
            if cut_count in res:
                res[cut_count] += counts[count]
            else:
                res[cut_count] = counts[count]

        return res
    
    def run(self, shots, postselect = True, mitigate = True):
        
        job = execute(self.circuit,
                      backend = self.backend,
                      shots = shots,
                      noise_model = self.noise_model,
                      coupling_map = self.coupling_map
                      )
        raw_counts = job.result().get_counts() # this has spaces between every postselection bit, which we better remove
        
        counts = {}
        
        for count in raw_counts:
            counts[count.replace(' ', '')] = raw_counts[count]
            
        if mitigate:
            counts = self.mitigate(counts)
            
        if postselect:
            return self.postselect(counts)
        else:
            return self.not_postselect(counts)
    
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
    
    def mitigate(self, counts):
        
        id_matrix = np.array([[1, 0], [0, 1]])
        M_inv = np.array([1])
        for i in range(self.total_no_bits):
            # read through which qubit each classical bit corresponds to and tensor product with the right
            # inverse calibration matrix based on that
            if self.measured_bits[i] == -1:
                M_inv = np.kron(M_inv, id_matrix)
            else:
                M_inv = np.kron(M_inv, self.mit_matrices[self.measured_bits[i]])
                
        # create array out of unmitigated counts (vector to multiply the matrix with)
        unmitigated_counts_array = np.zeros(2**self.total_no_bits)
        for i in range(2**(self.total_no_bits)):
            i_bin = format(i, '0'+str(self.total_no_bits)+'b')
            if i_bin in counts:
                unmitigated_counts_array[i] = counts[i_bin]
        
        #multiply matrix with vector to get mitigated counts
        mitigated_counts_array = np.matmul(M_inv, unmitigated_counts_array)
        
        #convert back to dictionary
        mitigated_counts_dict = {}
        for i in range(2**(self.total_no_bits)):
            i_bin = format(i, '0'+str(self.total_no_bits)+'b')
            mitigated_counts_dict[i_bin] = mitigated_counts_array[i]
        
        return mitigated_counts_dict