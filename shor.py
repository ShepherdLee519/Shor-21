import random
import math
from fractions import Fraction
from typing import List, Tuple

from pyqpanda import *
import numpy as np


class InitQMachine:
    """ Init Quantum machine and pre-alloc needed qubits/cbits

    Instance used as a context that containes qubits/cbits and related info.

    attributes:
        machine: init by init_quantum_machine()
        qubits: alloced qubits(by qAlloc_many)
        cbits: alloced classical bits(by cAlloc_many)
    
    """
    def __init__(self, qubitsCount: int, cbitsCount: int, machineType = QMachineType.CPU):
        self.machine = init_quantum_machine(machineType)
        
        self.qubits = self.machine.qAlloc_many(qubitsCount)
        self.cbits = self.machine.cAlloc_many(cbitsCount)
        
        print(f'Init Quantum Machine with qubits:[{qubitsCount}] / cbits:[{cbitsCount}] Successfully')
    
    def __del__(self):
        destroy_quantum_machine(self.machine)


def create_transform_circuit(a: int, N: int, ctx: InitQMachine) -> QCircuit:
    """ create a quantum function circuit applied in period-finding's second step 

    In this experiment, we just consider the case that N=21
    thus call with N != 21 will just raise an Error. 

    what's more, {a} should be pre-checked which means {a} can
    just be one of [2,4,5,8,10,11,13,16,17,19,20]

    cites: https://arxiv.org/pdf/1202.6614.pdf

    Args:
        a: a pre-seleted number to run shor-algorithm
        N: A composite number need to factoring
        ctx: Context about global quantum machine(qubits/cbits etc.)
    -------
    Returns:
        constructed transform function's QCircuit

    """
    if N != 21:
        raise NotImplementedError(f'transform circuit for N={N} haven\'t implemented')
    
    qubits = ctx.qubits
    transform_circuit = create_empty_circuit() 
    
    if a == 4:
        return transform_circuit << CNOT(qubits[2], qubits[1]) \
            << X(qubits[1]) << Toffoli(qubits[0], qubits[1], qubits[5]) << X(qubits[1]) \
            << X(qubits[0]) << X(qubits[1]) << Toffoli(qubits[0], qubits[1], qubits[3]) << X(qubits[0]) << X(qubits[1]) \
            << CNOT(qubits[2], qubits[1]) \
            << X(qubits[1]) << Toffoli(qubits[1], qubits[2], qubits[6]) << X(qubits[1]) \
            << X(qubits[0]) << Toffoli(qubits[0], qubits[6], qubits[5]) << X(qubits[0]) \
            << Toffoli(qubits[0], qubits[6], qubits[7]) \
            << CNOT(qubits[2], qubits[1]) \
            << CNOT(qubits[1], qubits[6]) \
            << CNOT(qubits[2], qubits[1]) \
            << X(qubits[0]) << Toffoli(qubits[0], qubits[6], qubits[7]) << X(qubits[0]) \
            << Toffoli(qubits[0], qubits[6], qubits[3]) \
            << X(qubits[2]) << Toffoli(qubits[1], qubits[2], qubits[6]) << X(qubits[2])
    elif a == 11:
        return transform_circuit << X(qubits[2]) << Toffoli(qubits[1], qubits[2], qubits[3]) << X(qubits[2]) \
            << X(qubits[0]) << Toffoli(qubits[0], qubits[3], qubits[7]) << X(qubits[0]) \
            << X(qubits[3]) << Toffoli(qubits[0], qubits[3], qubits[4]) << X(qubits[3]) \
            << CNOT(qubits[2], qubits[1]) \
            << CNOT(qubits[1], qubits[3]) \
            << CNOT(qubits[2], qubits[1]) \
            << X(qubits[0]) << Toffoli(qubits[0], qubits[3], qubits[5]) << X(qubits[0]) \
            << X(qubits[3]) << Toffoli(qubits[0], qubits[3], qubits[6]) << X(qubits[3]) \
            << X(qubits[1]) << Toffoli(qubits[1], qubits[2], qubits[3]) << X(qubits[1]) \
            << CNOT(qubits[2], qubits[1]) \
            << X(qubits[1]) << CNOT(qubits[1], qubits[3]) << X(qubits[1]) \
            << CNOT(qubits[2], qubits[1])
    elif a in [2,5,8,10,13,16,17,19,20]:
        raise NotImplementedError(f'circuit construction for a={a} haven\'t implemented')
    else:
        raise ValueError(f'a={a} is not a proper option for N=21!')

def create_program(a: int, N: int, ctx: InitQMachine) -> QProg:
    """ create qpanda QProg (shor-algorithm quantum circuits)

    input a is pre-checked and guaranted to be proper.

    Args:
        a: a pre-seleted number to run shor-algorithm
        N: A composite number need to factoring
        ctx: Context about global quantum machine(qubits/cbits etc.)
    -------
    Returns:
        constructed QProg

    """
    # Step 0. prepare related environment info
    qubits = ctx.qubits
    cbits = ctx.cbits
    phase_bits_qubits = len(cbits)
    function_value_qubits = len(qubits) - len(cbits)

    # Step 1. create superposition of states
    #
    # by applying Hadamard gates
    #
    hadamard_circuit = create_empty_circuit()

    for i in range(phase_bits_qubits):
        hadamard_circuit << H(qubits[i])
        
    hadamard_circuit << BARRIER(qubits)

    # Step 2. Implement a unitary transform function
    #
    transform_circuit = create_transform_circuit(a, N, ctx)

    # Step 3. perform a inverse quantum Fourier tranform
    # 
    qft_dagger_circuit = create_empty_circuit()

    for i in range(phase_bits_qubits - 1):
        qft_dagger_circuit << H(qubits[i])
        
        for j in range(i + 1, phase_bits_qubits):
            qft_dagger_circuit << U1(qubits[i], np.pi / (2 ** (j - i))).control(qubits[j])
        
        qft_dagger_circuit << BARRIER(qubits)

    qft_dagger_circuit << H(qubits[phase_bits_qubits - 1])

    # Step 4. build full circuit program
    #
    prog = create_empty_qprog()
    prog << hadamard_circuit << transform_circuit << qft_dagger_circuit

    return prog


def period_finding(a: int, N: int, ctx: InitQMachine) -> Tuple[bool, int]:
    """ period_finding subroutine called in shor-algorithm

    General process:
        init circuit => measure(get phase)
            => calculate period r (by continued fraction expansion)
            => validate period (a^r ≡ 1 (mod N)) thus return.

    Args:
        a: a pre-seleted number to run shor-algorithm
        N: A composite number need to factoring
        ctx: Context about global quantum machine(qubits/cbits etc.)
    -------
    Returns:
        ok: Boolean mark means success(true) or failure
        period: a proper period.
    
    """
    qubits = ctx.qubits
    cbits = ctx.cbits
    phase_bits_qubits = len(cbits)

    # Step 1. init period-finding QProg corresponding to a & N
    #
    # while actually in this experiment just implement the case N=21
    # 
    prog = create_program(a, N, ctx)

    # Step 2. Measure it (first phase_bits_qubits qubits) => get phase
    #
    # attention the reading order: qubits[i] => cbits[phase_bits_qubits - i - 1]
    # 
    for i in range(phase_bits_qubits):
        prog << Measure(qubits[i], cbits[phase_bits_qubits - i - 1])

    result = run_with_configuration(prog, cbits, 1)
    print(f'  result: {result}') # like {"101": 1}

    # Convert the reading bits to phase
    #
    # eg. {"101": 1} => ["101"] => 5 => 5 / (2^3) = 0.625
    # 
    phase = int(list(result)[0], 2) / (2 ** phase_bits_qubits)
    print(f'   - corresponding phase: {phase}')

    # Step 3. calculate period r (by continued fraction expansion)
    # 
    # eg. a = 4, phi = 0.625 gives the convergents: {0, 1, 1/2, 2/3, 5/8}
    # while r = 8 is invalid (4^8 mod 21 = 16 !== 1)
    # actually in this case only r = 3 is valid (4^3 mod 21 == 1) (https://arxiv.org/pdf/2103.13855.pdf)
    # 
    # Shor algorithm is designed to work for even orders only,
    # However, for certain choices of square coprime x and odd order, the algorithm works.
    # https://arxiv.org/pdf/1310.6446v2.pdf and https://www.nature.com/articles/nphoton.2012.259 point out it. 
    # 
    # a simple validation way is checking r is even or chosen [a] itself 
    # is a perfect square(https://arxiv.org/pdf/2103.13855.pdf)
    # 
    # we'll use Fraction module and using limit_denominator(limit) to gate r
    # according to the above discussion we'll narrow the limit unitl getting a valid r
    # or narrow to limit = 0 which means fail to find period. 
    #
    limit = N

    while True:
        frac = Fraction(phase).limit_denominator(limit)
        r = frac.denominator

        # simply check period
        if (a ** r) % N == 1:
            break
        else:
            # narrow limit to calculate new period
            limit = r - 1
            if limit <= 0:
                print(f'\n  Rewrite to fraction: {frac}, find period failed')

                return False, None


    # re-check calculated r
    # 
    # a itself is a perfect square thus Shor still works
    # cite: https://arxiv.org/pdf/2103.13855.pdf
    # 
    if (r % 2 != 0 and int(math.sqrt(a)) ** 2 != a) or \
        (int(a ** (r / 2)) % N == -1):
        print(f'\n  Rewrite to fraction: {frac}, find period failed')

        return False, None 


    print(f'\n  Rewrite to fraction: {frac}, thus r = {r}')

    return True, r


def calculate_factors(a: int, N: int, r: int) -> Tuple[bool, list]:
    """ calculate factors based on calculated period r.

    Possible case: r is even => gcd(a ** (r // 2) - 1, N) ... 
        or r is odd but {a} itself is perfect square => gcd(a ** (r / 2) - 1, N)

    Args:
        a: a pre-seleted number to run shor-algorithm
        N: A composite number need to factoring
        r: calculated valid period r
    -------
    Returns:
        ok: Boolean mark means success(true) or failure
        factors: a pair of factors(should be sorted) when success or empty [] when failed.
     
    """
    # According to Shor algorithm, calculate gcd(a^(r/2) - 1, N) and gcd(a^(r/2) + 1, N)
    # 
    guesses = [math.gcd(int(a ** (r / 2)) - 1, N), math.gcd(int(a ** (r / 2)) + 1, N)]
    
    print(f'  calculate final guesses: {guesses}')
    
    # need to check the calculated guesses numbers.
    # 
    factors = []
    for num in guesses:
        if num not in [1, N]:
            factors.append(num)
            print(f'[Find non-trivial factor: {num}]')
    
    if len(factors) == 0:
        print('[Failed]')

        return False, []
    elif len(factors) == 1:
        # may just find one factor and 1
        # calculate another
        # 
        factors.append(N // factors[0])
    
    return True, sorted(factors)


def shor_alg(N: int, *, a: int = None, ctx: InitQMachine) -> List[int]:
    """ Shor algorithm using to factoring a composite number N

    Args:
        N: A composite number need to factoring
        a: a pre-seleted number to run shor-algorithm
        ctx: Context about global quantum machine(qubits/cbits etc.)
    -------
    Returns:
        A pair of non-trivial factors of N (factors should be sorted)
        return {None} if failed.

    """

    # Step 0. If N is even, 2 is the factor
    if N % 2 == 0:
        return [2, N // 2]
    
    # Step 1. Randomly choose a number 1 < a < N
    if a == None:
        a = random.randint(2, N - 1)
        print(f'Randomly choose a = {a}\n')

    # Step 2. Check the randomly choosed number a
    # 
    # compute K = gcd(a, N), 
    # if K != 1, then K is thus a non-trivial factor of N
    # algorithm finished with returned [K, N / K]
    # 
    K = math.gcd(a, N)
    if K != 1:
        # thus K is one of a factor of N
        print(f' - gcd({a}, {N}) = {K}! {K} is a factor of {N}')
        print('\nShor Algorithm Finished!')

        return sorted([K, N // K])

    # Step 3. call quantum period-finding subroutine to find period r
    #
    # should check r that a^r ≡ 1 (mod N) (checked by subroutine function)
    # after getting a proper period r, calculate factors and return.
    #
    # because each time the running result will affected by nondeterministic measurements
    # (during period-finding, will measure the phase result to calculate period) 
    # for selected {a}, will try {MAX_ATTEMPT} to calculate period repeatly
    # 
    MAX_ATTEMPT = 20
    attempt = 1
    
    while True:
        print(f'Execute shor algorithm - attempt time: [{attempt}]')

        # call period-finding subroutine
        valid, r = period_finding(a, N, ctx)
        if valid:
            # valid period, and then calculate factors:
            ok, factors = calculate_factors(a, N, r)
            if ok:
                print(f'\nFactors of {N} are: {factors} (total run times: {attempt})')
                print('\nShor Algorithm Finished!')
                
                # reutrned factors should be sorted(by subroutine)
                return factors
            
        attempt += 1
        if attempt > MAX_ATTEMPT:
            print('\nShor Algorithm Finished [FAIL]')
            
            return None
        
        print('\n' + '-' * 36 + '\n')
    

def solution() -> List[int]:
    # Step 0. Init qMachine and qubits/cbits
    phase_bits_qubits = 3
    function_value_qubits = 5

    total_qubits = phase_bits_qubits + function_value_qubits

    ## init quantum machine
    ## related info(qubits/cbits and counts etc.) packed into a context object
    # 
    ctx = InitQMachine(total_qubits, phase_bits_qubits)

    # Cause the shor algorithm might fail => attempt some rounds
    MAX_ROUND = 8
    
    N = 21

    for round in range(MAX_ROUND):
        print(f'Attempt call shor-algorithm: round {round + 1}\n')

        res = shor_alg(N, ctx = ctx)

        if res != None:
            return res

    return


shor_alg(21, a=11, ctx=InitQMachine(8, 3))