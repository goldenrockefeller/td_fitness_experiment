from td_fitness import *
import sys
import numpy as np
import random

class Runner:
    def __init__(self, experiment_name, setup_funcs):
        self.setup_funcs = setup_funcs
        self.stat_runs_completed = 0
        self.experiment_name = experiment_name
        setup_names = []
        for setup_func in setup_funcs:
            setup_names.append(setup_func.__name__)
        self.trial_name = "_".join(setup_names)

        # Create experiment folder if not already created.
        try:
            os.makedirs(os.path.join("log", experiment_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

        # Save experiment details
        filenames_in_folder = (
            glob.glob("./**.py", recursive = True)
            + glob.glob("./**.pyx", recursive = True)
            + glob.glob("./**.pxd", recursive = True))
        for filename in filenames_in_folder:
            copy(filename, os.path.join("log", experiment_name, filename))


    def new_run(self):
        datetime_str = (
            dt.datetime.now().isoformat()
            .replace("-", "").replace(':', '').replace(".", "_")
        )

        print(
            "Starting trial.\n"
            f"experiment: {self.experiment_name}\n"
            f"trial: {self.trial_name}\n"
            f"stat run #: {self.stat_runs_completed}\n"
            "datetime: {datetime_str}\n\n"
            .format(**locals()) )
        sys.stdout.flush()

        args = {"n_steps" : 100, "horizon" : 32}

        for setup_func in self.setup_funcs:
            setup_func(args)

        critic = args["critic"]
        n_steps = args["n_steps"]
        domain_noise = 0.

        domain = Domain()
        domain.domain_noise = domain_noise
        domain.n_steps = n_steps
        n_epochs =  n_steps
        n_policies = 50

        dist = {}
        dist[State.B] = np.ones(2)
        dist[State.A] = np.ones(2)
        dist[State.C] = np.ones(2)

        kl_penalty_factor = 10.

        population = [new_policy(dist) for _ in range(n_policies)]

        n_epochs_elapsed = list(range(1, n_epochs + 1))
        n_training_episodes_elapsed = [n_epochs_elapsed[epoch_id] * n_policies for epoch_id in range(n_epochs)]
        n_training_steps_elapsed = [n_epochs_elapsed[epoch_id] * n_steps * n_policies for epoch_id in range(n_epochs)]
        max_expected_value = domain.max_value()
        max_expected_returns = [max_expected_value for _ in range(n_epochs)]
        scores = []
        expected_returns = []
        critic_evals = []
        critic_score_losses = []
        critic_expected_return_losses = []


        for epoch_id in range(n_epochs):

            phenotypes = phenotypes_from_population(population)

            new_critic = critic.copy() if critic is not None else None

            for phenotype in phenotypes:
                policy = phenotype["policy"]
                states, actions, rewards = domain.execute(policy)

                if critic is not None:
                    fitness = critic.eval(states, actions)
                else:
                    fitness = sum(rewards)

                if new_critic is not None:
                    new_critic.update(states, actions, rewards)
                phenotype["fitness"] = fitness

            critic = new_critic


            update_dist(dist, kl_penalty_factor, phenotypes)

            phenotypes.sort(reverse = False, key = lambda phenotype : phenotype["fitness"])
            for phenotype in phenotypes[0: 3 * len(phenotypes)//4]:
               phenotype["policy"] = new_policy(dist)
            random.shuffle(phenotypes)

            population = population_from_phenotypes(phenotypes)

            candidate_policy = population[0]
            states, actions, rewards = domain.execute(candidate_policy)
            print(f"Score: {domain.expected_value(candidate_policy)}")


            score = list_sum(rewards)
            expected_return = domain.expected_value(candidate_policy)
            if critic is not None:
                critic_eval = critic.eval(states, actions)
                critic_score_loss = 0.5 * (critic_eval - score) ** 2
                critic_expected_return_loss = 0.5 * (critic_eval - expected_return) ** 2

            scores.append(score)
            expected_returns.append(expected_return)
            if critic is not None:
                critic_evals.append(  critic.eval(states, actions) )
                critic_score_losses.append( critic_score_loss )
                critic_expected_return_losses.append( critic_expected_return_loss)

            # print(critic.learning_rate_scheme.denoms)
        # end for epoch in range(n_epochs):

        save_filename = (
            os.path.join(
                "log",
                self.experiment_name,
                self.trial_name,
                f"record_{datetime_str}.csv"
            )
        )

        # Create File Directory if it doesn't exist
        if not os.path.exists(os.path.dirname(save_filename)):
            try:
                os.makedirs(os.path.dirname(save_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(save_filename, 'w', newline='') as save_file:
            writer = csv.writer(save_file)

            writer.writerow(['n_epochs_elapsed'] + n_epochs_elapsed)
            writer.writerow(['n_training_episodes_elapsed'] + n_training_episodes_elapsed)
            writer.writerow(['n_training_steps_elapsed'] + n_training_steps_elapsed)

            writer.writerow(['max_expected_returns'] + max_expected_returns)
            writer.writerow(['scores'] + scores)
            writer.writerow(['expected_returns'] + expected_returns)
            writer.writerow(['critic_evals'] + critic_evals)
            writer.writerow(['critic_score_losses'] + critic_score_losses)
            writer.writerow(['critic_expected_return_losses'] + critic_expected_return_losses)

            if isinstance(critic, TrajCritic):
                writer.writerow(['Critic(State.B, Action.U)'] + [critic.core[(State.B, Action.U)]])
                writer.writerow(['Critic(State.B, Action.V)'] + [critic.core[(State.B, Action.V)]])
                writer.writerow(['Critic(State.A, Action.U)'] + [critic.core[(State.A, Action.U)]])
                writer.writerow(['Critic(State.A, Action.V)'] + [critic.core[(State.A, Action.V)]])
                writer.writerow(['Critic(State.C, Action.U)'] + [critic.core[(State.C, Action.U)]])
                writer.writerow(['Critic(State.C, Action.V)'] + [critic.core[(State.C, Action.V)]])

            if isinstance(critic, UqTrajCritic):
                writer.writerow(['Critic(State.B, Action.U)'] + [critic.q_critic.core[(State.B, Action.U)] + critic.u_critic.core[State.B]])
                writer.writerow(['Critic(State.B, Action.V)'] + [critic.q_critic.core[(State.B, Action.V)] + critic.u_critic.core[State.B]])
                writer.writerow(['Critic(State.A, Action.U)'] + [critic.q_critic.core[(State.A, Action.U)] + critic.u_critic.core[State.A]])
                writer.writerow(['Critic(State.A, Action.V)'] + [critic.q_critic.core[(State.A, Action.V)] + critic.u_critic.core[State.A]])
                writer.writerow(['Critic(State.C, Action.U)'] + [critic.q_critic.core[(State.C, Action.U)] + critic.u_critic.core[State.C]])
                writer.writerow(['Critic(State.C, Action.V)'] + [critic.q_critic.core[(State.C, Action.V)] + critic.u_critic.core[State.C]])

            if isinstance(critic, ATrajCritic):
                writer.writerow(['Critic(State.B, Action.U)'] + [critic.q_critic.core[(State.B, Action.U)] + critic.v_critic.core[State.B]])
                writer.writerow(['Critic(State.B, Action.V)'] + [critic.q_critic.core[(State.B, Action.V)] + critic.v_critic.core[State.B]])
                writer.writerow(['Critic(State.A, Action.U)'] + [critic.q_critic.core[(State.A, Action.U)] + critic.v_critic.core[State.A]])
                writer.writerow(['Critic(State.A, Action.V)'] + [critic.q_critic.core[(State.A, Action.V)] + critic.v_critic.core[State.A]])
                writer.writerow(['Critic(State.C, Action.U)'] + [critic.q_critic.core[(State.C, Action.U)] + critic.v_critic.core[State.C]])
                writer.writerow(['Critic(State.C, Action.V)'] + [critic.q_critic.core[(State.C, Action.V)] + critic.v_critic.core[State.C]])


            writer.writerow(['Policy(State.B, Action.U)'] + [policy[State.B]])
            writer.writerow(['Policy(State.B, Action.V)'] + [1. - policy[State.B]])
            writer.writerow(['Policy(State.A, Action.U)'] + [policy[State.A]])
            writer.writerow(['Policy(State.A, Action.V)'] + [1. - policy[State.A]])
            writer.writerow(['Policy(State.C, Action.U)'] + [policy[State.C]])
            writer.writerow(['Policy(State.C, Action.V)'] + [1. - policy[State.C]])


        self.stat_runs_completed += 1


        self.population = population
        self.critic = critic
        self.dist = dist