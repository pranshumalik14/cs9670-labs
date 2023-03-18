import numpy as np
import mdp


class RL:
    def __init__(self, mdp, sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distribution and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self, state, action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action, state])
        cumProb = np.cumsum(self.mdp.T[action, state, :])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward, nextState]

    def qLearning(self, s0, initialQ, nEpisodes, nSteps, nTrials=10, epsilon=0, temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probability epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        nTrials -- # times to re-run the learning to gain confidence on the results
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        Q_est = np.zeros([self.mdp.nActions, self.mdp.nStates])  # Q estimate
        policy = np.zeros(self.mdp.nStates, int)
        cum_rewards = np.zeros(nEpisodes)

        for _ in range(nTrials):
            Q = initialQ  # all trials start with same initial Q estimate
            counts = np.zeros([self.mdp.nActions, self.mdp.nStates])
            rewards = np.zeros(nEpisodes)  # rewards collection for curr trial
            for epi in range(nEpisodes):   # repeat rollouts to improve Q estimates
                s = s0
                for t in range(nSteps):
                    # select action
                    if epsilon > 0:  # epsilon-greedy exploration
                        if np.random.rand(1) < epsilon:
                            a = np.random.randint(self.mdp.nActions)
                        else:
                            a = np.argmax(Q[:, s])
                    elif temperature > 0:  # Boltzmann exploration
                        # softmax on Q/T
                        probs_a = [np.exp(Q[a][s] / temperature)
                                   for a in range(self.mdp.nActions)]
                        probs_a /= np.sum(probs_a)
                        a = np.random.choice(self.mdp.nActions, p=probs_a)
                    else:
                        a = np.argmax(Q[:, s])
                    # execute action
                    [reward, s_prime] = self.sampleRewardAndNextState(s, a)
                    # update rewards
                    rewards[epi] += reward * (self.mdp.discount**t)
                    # update counts
                    counts[a][s] += 1
                    alpha = 1.0/counts[a][s]
                    # update Q value
                    Q[a][s] += alpha*(
                        reward + (self.mdp.discount *
                                  np.max(Q[:, s_prime])) - Q[a][s]
                    )
                    # proceed to next step/decision
                    s = s_prime
            cum_rewards += rewards/nTrials
            Q_est += Q/nTrials
        # extract policy
        policy = Q_est.argmax(axis=0)  # best action per state

        return [Q_est, policy, cum_rewards]
