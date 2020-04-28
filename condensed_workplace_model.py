import numpy as np
import matplotlib.pyplot as plt

def get_default_params():
    return {
        'N_pop': int(1e5),  # Population size
        'N_work': 100,      # Number of people in a workplace
        'N_home': 2,        # Number of people in a home

        # 'Betas'. i.e. probability of infection from a single infectious person is 
        'beta_work': 0.1,     # Daily probability of infection if sharing workplace
        'beta_home': 0.2,     # Daily probability of infection if sharing home
        'beta_street': 0.01,  # Random contact on the street, transport, etc.

        # 'Work' condensation parameters
        'lambda_work': 1,   # Enhanced probability of infecting individuals to either side of you
        'condensation_work': 0.2, # Degree of additional infection probability for 'close' colleagues.
                                # 'Should' lie in [0, 1). 

        # 'If' this is True, symptomatic individuals no longer go to work
        'symptomatics_stay_off_work': False,
        'random_seed': 1234,
    }


def run_simulation(params):
    N_pop = params['N_pop']
    N_work = params['N_work']
    N_home = params['N_home']
    beta_work = params['beta_work']
    beta_home = params['beta_home']
    beta_street = params['beta_street']
    lambda_work = params['lambda_work']
    condensation_work = params['condensation_work']
    symptomatics_stay_off_work = params['symptomatics_stay_off_work']

    rng = np.random.RandomState(params['random_seed'])

    # Probability that an infected person recovers after i days of infection.
    # These numbers are completely made up.
    p_recover = np.array(([0.0] * 3) + ([0.1] * 6) + ([0.2] * 3) + ([0.3]*6) + [1.0])
    # Probability that someone becomes symptomatic after i days of infection.
    # Also completely made up.
    p_symptomatic = np.array(([0.0] * 3) + ([0.05] * 6) + ([0.1] * 3) + ([0.2]*6) + [0.2])

    # Static state of the population
    # Note: The way we set up workplaces here means that they are contingious.
    # This is important to acheive faster running times.
    v_work = np.arange(N_pop) // N_work  # Index of workplace
    v_home = np.arange(N_pop) // N_home  # Index of home
    rng.shuffle(v_home)                  # Make workplace and home independent

    # This is more work than needs to be done for constant-size workplaces
    # But it is intended to also cover the case of non-constant sizes.
    size_cache = {}
    def get_workplace_matrix(workplace_size):
        """
        Let W[i, j] be the probability of j infecting i on one day.
        The what we actually compute here is a matrix of log(1 - W[i, j])
        """
        if workplace_size in size_cache:
            return size_cache[workplace_size]
        else:
            b = condensation_work * beta_work / (2 * lambda_work)
            a = (beta_work - 2*lambda_work*b) / (workplace_size - 1)
            w_mat = np.zeros(shape=(size_this_workplace, size_this_workplace))
            for i in range(size_this_workplace):
                for j in range(size_this_workplace):
                    if i != j and np.abs(i - j) <= lambda_work:
                        w_mat[i, j] = a + b
                    elif i != j:
                        w_mat[i, j] = a
            size_cache[workplace_size] = np.log(1.0 - w_mat)
            return np.log(1.0 - w_mat)

    # Set workplace matrices
    w_set = set(v_work)
    workplace_sizes = np.bincount(v_work)

    w_log_matrix_list = [None]*len(w_set)
    for w in w_set:
        size_this_workplace = workplace_sizes[w]
        w_log_matrix_list[w] = get_workplace_matrix(size_this_workplace)

    # For each individual, get their index *within* the workplace.
    # These are needed to compute the infection probabilities using the w matrices.
    v_w_idx = np.zeros_like(v_work)
    for w in w_set:
        size_this_workplace = workplace_sizes[w]
        v_w_idx[v_work == w] = np.arange(size_this_workplace)

    # Initial state of the population - no-one is infected
    v_susceptible = np.full(shape=(N_pop,), fill_value=True)
    v_infected = np.full(shape=(N_pop,), fill_value=False)
    v_symptomatic = np.full(shape=(N_pop,), fill_value=False)
    v_days_infected = np.full(shape=(N_pop,), fill_value=0)

    # Pre-compute these to allow slice access for workplace computations in stochastic_update
    # This is why we need workplaces to be contigious
    # i.e. persons 0, 1, 2, ... n_0 - 1 all work in workplace 0,
    # and persons n_0, n_0 + 1, ..., n_1 - 1 all work in workplace 1, etc.
    work_start_vectors = np.zeros(shape=(len(w_set),), dtype=np.int32)
    work_end_vectors = np.zeros(shape=(len(w_set),), dtype=np.int32)
    for w in range(len(w_set)):
        x = np.argmax([v_work == w])
        work_start_vectors[w] = x
        if w > 0:
            work_end_vectors[w - 1] = x
    work_start_vectors[0] = 0
    work_end_vectors[-1] = len(v_work)

    def stochastic_update():
        # Compute the probability of a susceptible getting infected by various routes
        
        # Workplace infection probability is the most complicated due to inhomogeneous workplaces
        # For each individual, compute their probability of being infected at work
        v_p_work = np.zeros_like(v_work, dtype=np.float32)
        for w in w_set:
            w_log_mat = w_log_matrix_list[w]
            # Compute log(1 - p_i) where i ranges over workers in this workplace
            if not symptomatics_stay_off_work:
                log_one_minus_p = np.matmul(w_log_mat, v_infected[work_start_vectors[w]:work_end_vectors[w]])
            else:
                log_one_minus_p = \
                        np.matmul(w_log_mat,
                                  v_infected[work_start_vectors[w]:work_end_vectors[w]] \
                                  & ~v_symptomatic[work_start_vectors[w]:work_end_vectors[w]])
            p = 1.0 - np.exp(log_one_minus_p)
            v_p_work[work_start_vectors[w]:work_end_vectors[w]] = p
            
        # For each home, compute the infection probability
        h_infected = np.bincount(v_home[v_infected], minlength=np.max(v_home) + 1)
        p_home = 1.0 - np.power(1.0 - beta_home / N_home, h_infected)
        # Compute the probability of being infected in the street
        total_infected = np.sum(v_infected)
        p_street = 1.0 - np.power(1.0 - beta_street / N_pop, total_infected)
        
        # Infection probability for individuals
        v_p = 1.0 - (1.0 - v_p_work) * (1.0 - p_home[v_home]) * (1.0 - p_street)
        # Sample new infections
        new_infection = (rng.binomial(1, p=v_p) == 1)
        v_infected[v_susceptible] = new_infection[v_susceptible]
        v_susceptible[v_susceptible] = ~new_infection[v_susceptible]
        
        # Compute probability of infected individuals recovering
        v_p = p_recover[v_days_infected]
        # Sample recoveries
        new_recoveries = (rng.binomial(1, p=v_p) == 1)
        v_infected[v_infected] = ~new_recoveries[v_infected]
        # Recovered individuals also lose symptoms
        v_symptomatic[v_symptomatic] = ~new_recoveries[v_symptomatic]
        
        # Compute probabilities of infected individuals becoming symptomatic
        v_p = p_symptomatic[v_days_infected]
        # Sample becoming symptomatic
        new_symptoms = (rng.binomial(1, p=v_p) == 1)
        v_symptomatic[v_infected & ~v_symptomatic] = new_symptoms[v_infected & ~v_symptomatic]
        
        # Increment days infected
        v_days_infected[v_infected] += 1

    # Randomly infect a tiny number of people
    initial_infections = rng.choice(N_pop, size=4)
    v_infected[initial_infections] = True
    v_susceptible[initial_infections] = False

    susceptible_vs_t = []
    infected_vs_t = []
    symptomatic_vs_t = []
    recovered_vs_t = []

    for t in range(500):
        susceptible_count = np.sum(v_susceptible)
        infected_count = np.sum(v_infected)
        symptomatic_count = np.sum(v_symptomatic)
        recovered_count = N_pop - susceptible_count - infected_count
        susceptible_vs_t.append(susceptible_count)
        infected_vs_t.append(infected_count)
        symptomatic_vs_t.append(symptomatic_count)
        recovered_vs_t.append(recovered_count)

        if infected_count == 0:
            break
        
        stochastic_update()

    return {
        'susceptible_vs_t': susceptible_vs_t,
        'infected_vs_t': infected_vs_t,
        'symptomatic_vs_t': symptomatic_vs_t,
        'recovered_vs_t': recovered_vs_t,
        'total_ever_infected_vs_t': [N_pop - s for s in susceptible_vs_t]
    }

