import random
import numpy as np

from tqdm import tqdm
from .agent import Agent
from random import random, randrange

from utils.memory_buffer import MemoryBuffer
from utils.networks import tfSummary
from utils.stats import gather_stats
from utils.csvwriter import WritetoCsvFile
class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim, args):
        """ Initialization
        """
        # Environment and DDQN parameters
        self.with_per = args.with_per
        self.action_dim = action_dim
        self.state_dim = state_dim
        #
        self.lr = 2.5e-4
        self.gamma = 0.95
        self.epsilon = 0.8
        self.epsilon_decay = 0.99
        self.buffer_size = 20000
        #
        self.tau = 1e-2

        # Create actor and critic networks
        self.agent = Agent(self.state_dim, action_dim, self.lr, self.tau, args.dueling)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size, args.with_per)

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            return np.argmax(self.agent.predict(s)[0])

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, new_s, idx = self.buffer.sample_batch(batch_size)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.agent.predict(s)
        next_q = self.agent.predict(new_s)
        q_targ = self.agent.target_predict(new_s)

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i,:])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]
            if(self.with_per):
                # Update PER Sum Tree
                self.buffer.update(idx[i], abs(old_q - q[i, a[i]]))
        # Train on batch
        self.agent.fit(s, q)
        # Decay epsilon
        self.epsilon *= self.epsilon_decay


    def train(self, env, args, summary_writer):
        """ Main DDQN Training Algorithm
        """

        results = []
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        epoch=0
        gross_profit = 0
        WritetoCsvFile("logFile_1.csv", ["stage", "file", "history_win", "usevol", "maxProfit", "maxLOSS", "avgProfit", "avgLOSS",
                         "maxdrop", "Total profit", "TRADES", "epoch"])
        WritetoCsvFile("logFileDetail.csv",['stage', 'maxProfit', 'maxLOSS', 'avgProfit', 'avgLOSS', 'maxdrop', 'Total profit', 'gross profit', 'TRADES', 'epoch'])

        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, done  = 0, 0, False
            old_state = env.reset()
            ##########################################
            total_reward = 0
            total_profit = 0
            total_loss = 0
            total_profitMax = 0
            total_profitMin = 0
            max_drop = 0
            profitLst = []
            lossLst = []
            trades =0
            step = 0
            #####################################3####

            while not done:
                if args.render: env.render()
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)
                # Retrieve new state, reward, and whether the state is terminal
                #new_state, r, done, _ = env.step(a)

                #######################################################
                new_state, r, done, buy, sell, profit = env.step(a)

                total_reward += r
                if profit != 0:
                    trades += 1
                    total_profit += profit
                    if total_profit > total_profitMax:
                        total_profitMax = total_profit
                        total_profitMin = total_profit
                    if total_profit < total_profitMin:
                        total_profitMin = total_profit
                        try:
                            if total_profitMax != 0 and max_drop < (total_profitMax - total_profitMin) / total_profitMax :
                                max_drop = (total_profitMax - total_profitMin) / total_profitMax
                        except:
                            max_drop=0


                if profit > 0:
                    profitLst.append(profit)
                elif profit < 0:
                    lossLst.append(profit)

                step += 1
                if step % 1500 == 0:
                    print('maxProfit: {} maxLOSS: {} avgProfit: {:01.2f} avgLOSS: {:01.2f} maxdrop: {:.2%} Total profit: {}/{} TRADES: {}  '.format(
                        np.max(profitLst), -np.min(lossLst), np.mean(profitLst), -np.mean(lossLst),
                        max_drop, total_profit, gross_profit, trades))

                    WritetoCsvFile("logFileDetail.csv", ["train", np.max(profitLst), -np.min(lossLst), np.mean(profitLst), -np.mean(lossLst),
                                                        max_drop, total_profit, gross_profit, trades, epoch])
                #done = True if step == len(env.data) - 3 else False
                ######################################################
                # Memorize for experience replay
                self.memorize(old_state, a, r, done, new_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1
                # Train DDQN and transfer weights to target network
                if(self.buffer.size() > args.batch_size):
                    self.train_agent(args.batch_size)
                    self.agent.transfer_weights()

            gross_profit += total_profit
            # Gather stats every episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            l_profit = tfSummary('profit', total_profit)
            l_aprofit = tfSummary('average profit', np.mean(profitLst))
            l_aloss = tfSummary('l_aloss', -np.mean(lossLst))
            l_trades = tfSummary('l_trades', trades)
            np.mean(profitLst), -np.mean(lossLst)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.add_summary(l_profit, global_step=e)
            summary_writer.add_summary(l_aprofit, global_step=e)
            summary_writer.add_summary(l_aloss, global_step=e)
            summary_writer.add_summary(l_trades, global_step=e)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()
            self.agent.saveModel("./models/model_ep", "")
            results = [np.max(profitLst), -np.min(lossLst), np.mean(profitLst), -np.mean(lossLst), max_drop,
                       total_profit, trades]
            epoch +=1
            WritetoCsvFile("logFile_1.csv",["train", args.trainf, args.history_win, args.usevol] + results + [epoch])

        return results

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """

        if(self.with_per):
            q_val = self.agent.predict(state)
            q_val_t = self.agent.target_predict(new_state)
            next_best_action = np.argmax(q_val)
            new_val = reward + self.gamma * q_val_t[0, next_best_action]
            td_error = abs(new_val - q_val)[0]
        else:
            td_error = 0
        self.buffer.memorize(state, action, reward, done, new_state, td_error)

    def evaluate(self, env, args, summary_writer, model, andtrain=True):
        """ Evaluate            """
        results = []
        self.agent.loadModel(model, "")
        done = False
        old_state = env.reset()
        ##########################################
        total_reward = 0
        total_profit = 0
        total_loss = 0
        total_profitMax = 0
        total_profitMin = 0
        max_drop = 0
        profitLst = []
        lossLst = []
        step = 0
        trades =0
        #####################################3####
        while not done:
            # if args.render: env.render()
            # Actor picks an action (following the policy)
            a = self.policy_action(old_state)
            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, buy, sell, profit = env.step(a)

            #######################################################
            total_reward += r
            if profit != 0:
                trades += 1
                total_profit += profit
                if total_profit > total_profitMax:
                    total_profitMax = total_profit
                    total_profitMin = total_profit
                if total_profit < total_profitMin:
                    total_profitMin = total_profit
                    try:
                        if total_profitMax != 0 and max_drop < (total_profitMax - total_profitMin) / total_profitMax:
                            max_drop = (total_profitMax - total_profitMin) / total_profitMax
                    except:
                        max_drop = 0
            if profit > 0:
                profitLst.append(profit)
            elif profit < 0:
                lossLst.append(profit)
            step += 1
            if step % 1500 == 0:
                print(
                    'maxProfit: {} maxLOSS: {} avgProfit: {:01.2f} avgLOSS: {:01.2f} maxdrop: {:.2%} Total profit: {} TRADES: {}  '.format(
                        np.max(profitLst), -np.min(lossLst), np.mean(profitLst), -np.mean(lossLst),
                        max_drop, total_profit,  trades))
                WritetoCsvFile("logFileDetail.csv",
                               ["eval", np.max(profitLst), -np.min(lossLst), np.mean(profitLst), -np.mean(lossLst),
                                max_drop, total_profit, gross_profit, trades, 'eval'])
            #done = True if step == len(env.data) - 2 else False
            ######################################################
            # Memorize for experience replay
            if andtrain:
                self.memorize(old_state, a, r, done, new_state)
                # Train DDQN and transfer weights to target network
                if (self.buffer.size() > args.batch_size):
                    self.train_agent(args.batch_size)
                    self.agent.transfer_weights()
            # Update current state
            old_state = new_state
        print('maxProfit: {} maxLOSS: {} avgProfit: {:01.2f} avgLOSS: {:01.2f} maxdrop: {:.2%} Total profit: {} TRADES: {}  '.format(np.max(profitLst), -np.min(lossLst), np.mean(profitLst), -np.mean(lossLst), max_drop, total_profit, trades))
        results=[np.max(profitLst), -np.min(lossLst), np.mean(profitLst), -np.mean(lossLst), max_drop, total_profit, trades]
        WritetoCsvFile("logFile_1.csv", ["eval", args.trainf, args.history_win, args.usevol] + results + 'eval')
        return results