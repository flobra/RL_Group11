import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, RNDModel, RNDModel_Action
from utils import global_grad_norm_


class RNDAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            num_gpu=None,
            use_noisy_net=False,
            actionVector_opt = 0,
            bn = False):
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.update_proportion = update_proportion
        
        # self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.device = torch.device(f'cuda:{num_gpu}' if use_cuda else 'cpu')
        torch.cuda.set_device(self.device)
        if use_cuda: print('gpu:', torch.cuda.current_device())

        self.actionVector_opt = actionVector_opt
        self.actionVector_bn = bn

        if actionVector_opt:
            self.rnd = RNDModel_Action(input_size, output_size, self.actionVector_opt, self.actionVector_bn)
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.rnd.predictor_convs.parameters()) + list(self.rnd.predictor_fc1.parameters()) + list(self.rnd.predictor_fc2.parameters()),
                                    lr=learning_rate)
        else:
            self.rnd = RNDModel(input_size, output_size) # original RND model
            self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.rnd.predictor.parameters()),
                                    lr=learning_rate)
            
        self.rnd = self.rnd.to(self.device)
        self.model = self.model.to(self.device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value_ext, value_int = self.model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(action_prob)

        return action, value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)


    def compute_intrinsic_reward(self, next_obs):
        next_obs = torch.FloatTensor(next_obs).to(self.device)

        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()


    def compute_intrinsic_reward_actions(self, next_obs, actions):

        next_obs = torch.FloatTensor(next_obs).to(self.device)
        target_next_feature, predict_next_feature = self.rnd(next_obs, actions) # 
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()


    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)

                # print(next_obs_batch.shape) # torch.Size([16384, 1, 84, 84])
                # print(y_batch.shape) # torch.Size([16384])
                next_obs_batch_sampled = next_obs_batch[sample_idx]
                action_sampled = y_batch[sample_idx]
                # print(next_obs_batch_sampled.shape) # [4096, 1, 84, 84]
                # print(action_sampled.shape) # [4096]

                # predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch[sample_idx])
                if self.actionVector_opt:
                    predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch_sampled, action_sampled)
                else:
                    predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch_sampled)

                # print(predict_next_state_feature.shape) # [4096, 512]

                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                # ---------------------------------------------------------------------------------

                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_ext_loss + critic_int_loss

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + forward_loss
                loss.backward()
                if self.actionVector_opt:
                    global_grad_norm_(list(self.model.parameters())+list(self.rnd.predictor_convs.parameters())+list(self.rnd.predictor_fc1.parameters()) + list(self.rnd.predictor_fc2.parameters()))
                else:
                    global_grad_norm_(list(self.model.parameters())+list(self.rnd.predictor.parameters()))
                self.optimizer.step()
