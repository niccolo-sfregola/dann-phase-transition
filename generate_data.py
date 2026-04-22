import numpy as np
from tqdm import tqdm


N = 20
temperatures = np.linspace(0.2, 5.0, 25)
interaction = 1
burnin_time = 10000
n_sample_distance = 1000
n_samples = 100
n_runs = 50


def get_energy(J, spins, random_field):
    energy = -J * np.sum(
        spins * np.roll(spins, 1, axis=0) +
        spins * np.roll(spins, -1, axis=0) +
        spins * np.roll(spins, 1, axis=1) +
        spins * np.roll(spins, -1, axis=1)
    ) / 2

    if random_field is not None:
        energy -= np.sum(random_field * spins)

    return energy

def deltaE(J, spins, i, j, random_field):
    flip = -spins[i, j]
    delta = -2 * J * flip * (
        spins[(i+1) % N, j] + spins[(i-1) % N, j] +
        spins[i, (j+1) % N] + spins[i, (j-1) % N]
    )

    if random_field is not None:
        delta -= 2 * flip * random_field[i, j]

    return delta

def do_gibbs_sampling(interaction, spins, energy, temperature, n_samples, random_field):
    for _ in range(n_samples):
        i = np.random.randint(spins.shape[0])
        j = np.random.randint(spins.shape[1])
        delta = deltaE(interaction, spins, i, j, random_field)
        if delta < 0 or np.exp(-delta / temperature) > np.random.random():
            spins[i, j] *= -1
            energy += delta
    return spins, energy

def get_source_data(N, temperatures, interaction, n_runs, burnin_time, n_samples, n_sample_distance, Tc=2.27):
    data_source = []
    labels_source = []

    total_iterations = len(temperatures) * n_runs * n_samples

    with tqdm(total=total_iterations, desc="Generating Source Data") as pbar:
        for temp in temperatures:
            for run in range(n_runs):
                spins = np.random.choice([-1, 1], size=(N, N))
                energy = get_energy(interaction, spins, random_field=None)

                # Burn-in
                spins, energy = do_gibbs_sampling(interaction, spins, energy, temp, burnin_time, random_field=None)

                for _ in range(n_samples):
                    # Campionamento
                    spins, energy = do_gibbs_sampling(interaction, spins, energy, temp, n_sample_distance, random_field=None)

                    # Etichetta: 1 per T < Tc, 0 per T >= Tc
                    label = 1 if temp < Tc else 0
                    data_source.append(spins.copy())
                    labels_source.append(label)

                    pbar.update(1)

    # Salvataggio su file
    data_source = np.array(data_source)
    labels_source = np.array(labels_source)
    np.savetxt('data_source.txt', data_source.reshape(data_source.shape[0], -1), fmt='%d')
    np.savetxt('labels_source.txt', labels_source, fmt='%d')

    return data_source, labels_source

def get_target_data(N, temperatures, interaction, n_runs, burnin_time, n_samples, n_sample_distance, disorder_strengths, sigma):

    
    for h in disorder_strengths:
        
        data_target = []
        total_iterations = len(temperatures) * n_runs * n_samples

        print(f"\nGenerazione dei dati target con campo magnetico h = {h}")

        with tqdm(total=total_iterations, desc=f"Generating Target Data (h={h})") as pbar:
            for temp in temperatures:
                for run in range(n_runs):
                    # Genera un nuovo campo magnetico casuale per ogni run
                    mixture_choice = np.random.choice([0, 1], size=(N, N))
                    random_field = np.where(
                        mixture_choice == 0,
                        np.random.normal(-h, sigma, size=(N, N)),
                        np.random.normal(h, sigma, size=(N, N))
                    )

                    spins = np.random.choice([-1, 1], size=(N, N))
                    energy = get_energy(interaction, spins, random_field=random_field)


                    spins, energy = do_gibbs_sampling(interaction, spins, energy, temp, burnin_time, random_field)

                    for _ in range(n_samples):

                        spins, energy = do_gibbs_sampling(interaction, spins, energy, temp, n_sample_distance, random_field)
                        data_target.append(spins.copy())

                        pbar.update(1)

        # Salvataggio su file
        data_target = np.array(data_target)
        filename = f'data_target_h_{h}.txt'
        np.savetxt(filename, data_target.reshape(data_target.shape[0], -1), fmt='%d')

        print(f"Dati target per h = {h} salvati in '{filename}'.")

    return



np.random.seed(42)

'''data_source, labels_source = get_source_data(
    N, temperatures, interaction, n_runs, burnin_time, n_samples, n_sample_distance
)
'''

disorder_strengths = [0.5, 1.0, 1.5]

get_target_data(
    N, temperatures, interaction, n_runs, burnin_time, n_samples, n_sample_distance, disorder_strengths, sigma = 0.05
)