import numpy as np, math, statistics

# define immutable constants/ expressions here
length_of_a_timestep = 300  # 300 seconds/ 5 minutes
charging_efficiency_of_the_battery = 0.7  # ----random
discharging_efficiency_of_the_battery = 0.8  # --random
max_energy_charging_rate_per_unit_time = 0  # -------randomly placing zero for now, dont know for sure--------------- in terms of per unit time
max_energy_discharging_rate_per_unit_time = 0  # ------randomly placing zero for now-------------in terms of per unit time
max_energy_charging_possible_per_timestep = length_of_a_timestep * max_energy_charging_rate_per_unit_time  # ---------random, in terms of energy for now
max_energy_discharging_possible_per_timestep = length_of_a_timestep * max_energy_discharging_rate_per_unit_time  # -------in terms of energy for now
BSOC_max = 5000  # ---------random
BSOC_min = 2500  # ---------0
coefficient_vector_alpha_beta_gamma = np.array([charging_efficiency_of_the_battery, discharging_efficiency_of_the_battery, 0])
energy_consumption_from_utility_cost_per_unit = 0.20
peak_demand_cost_per_unit = 20
battery_degradation_cost_per_unit_energy = 0.05  # random for now
large_coefficient_for_constraints = 100000


class DataCenter_Env:
    def __init__(self):
        ''''''
        '''--------------------------------State Variables--------------------------------------------------'''
        self.last_n_timesteps_demand = None  # ----None for now, if we wanna use n time steps in the state variable, need to skip the first n timesteps-----
        self.current_BSOC = BSOC_max
        self.old_BSOC = 0
        self.current_timestep = 1  # -----------------------------first timestep
        self.current_slot = 1  # ---------------------------------first slot



        '''--------------------------------Other Variables-------------------------------------------------------'''
        self.utility_demand_buffer_for_timesteps_in_a_slot = []  # getting the average of three values on this list gives the average of the slot
        self.max_avg_peak_demand_comparision_variable = 0  # we put the average of the aboe list in this variable for comparison
        self.old_avg_peak_demand_in_a_slot = 0  # we compare the above variable with this variable
        self.max_peak_demand_average = 0  # --------------0 in the beginning, we replace this value with the max of above two variables

        self.demand_timestep_iteration_count = 0
        self.true_total_DCI_demand_next_timestep = 0




        '''-----------------------------------actions given by the agent-------------------------------------------------------'''
        self.agent_given_Dt = 0  # ----------------------->Dt
        self.agent_given_It = 0  # ------------------->It
        self.It_actual = charging_efficiency_of_the_battery * self.agent_given_It
        self.agent_given_Ot = 0  # ----------------------->Ot
        self.Ot_actual = discharging_efficiency_of_the_battery * self.agent_given_Ot
        self.agent_given_Dt_plus_It_actual = self.agent_given_Dt + self.actual_battery_energy_supply_to_DCI_for_next_timestep

        '''-----------------------------------Other variables-------------------------------------------------------------------'''
        self.action_array = np.zeros((1, 3))
        self.monetary_cost_without_peak_cost_in_a_day = []
        self.monetary_cost_with_peak_cost = 0
        self.day_counter_for_only_monetary_cost = 0
        self.reward_daywise = []  # ---------------------------------------------list of everyday reward
        self.cumulative_reward_in_a_day = []  # -------------------------------->total accumulated reward till this day for the month, we just sum the elements of the above list
        # added penalty for not meeting minimum feasible BSOC at the end of the peak period
        self.peak_end_penalty_for_residual_energy_at_12am = 0  # lets keep zero for now

        '''------------------------------------Ec and Dc calculations-----------------------------------------------'''
        self.Ec = 0
        self.Dc = 0
        self.consumption_from_utility_after_taking_agent_action_into_account = None
        self.Ymax_old = 0
        self.is_new_Ymax_greater = None


    def step(self, action_vector):  # --------------------------action comes as a vector from RL agent It Ot Dt
        ''''''
        # done = False

        '''-------------------------------update It actual and Ot actual here-------------------------------------------'''
        self.It_actual = action_vector[0] * charging_efficiency_of_the_battery
        self.Ot_actual = action_vector[1]/discharging_efficiency_of_the_battery

        '''-------------------------------Say TD3 exists and gives us the action vector as-----------------------------'''
        #need to call TD3 here
        self.action_array = action_vector # It Ot Dt

        # update the avg demand per slot buffer using It and Dt.
        #self.utility_demand_buffer_for_timesteps_in_a_slot.append(action_vector[0] + action_vector[2])

        '''Update BSOC'''
        """Change the formulation later to avoid simultaneous charging and discharging of the battery."""
        self.old_BSOC = self.current_BSOC
        self.current_BSOC = self.current_BSOC + self.It_actual - self.Ot_actual

        self.reward_of_the_last_action_taken = 0

        # update according to timeslot and timestep
        if (self.current_timestep == 3):

            self.max_avg_peak_demand_comparision_variable = np.average(np.array(self.utility_demand_buffer_for_timesteps_in_a_slot))

            # ----------------Need to update Ymax before calculating self's reward for this ongoing slot
            if self.max_peak_demand_average < self.max_avg_peak_demand_comparision_variable:
                self.max_peak_demand_average = self.max_avg_peak_demand_comparision_variable  # --------2)Ymax

            self.utility_demand_buffer_for_timesteps_in_a_slot = []  # empty the buffer

            self.current_timestep = 1

            self.cumulative_reward_in_a_day.append(self.get_reward())  # maintain a list until the day ends and then append the sum of its elements to the list reward_daywise

            if self.current_slot == 24:

                # put the reward for the day in some new buffer here--------------------------VVI- IMP
                # update other variables accordingly
                # updataa BSOC

                reward = self.get_reward()
                self.cumulative_reward_in_a_day.append(reward)
                self.reward_daywise.append(np.sum(self.cumulative_reward_in_a_day))
                self.cumulative_reward_in_a_day = []

                self.current_slot = 1  # ----------------------------------------------3)Slot No
                return reward

                # done = True  # tells whether its time to reset the environment or not,
                # basically tells if the episode ended or not
                # we end the episode after each day, but continue the calculation of LSTM such that
                # we continue from 6pm the next day

            else:

                reward = self.get_reward()
                self.cumulative_reward_in_a_day.append(reward)
                self.current_slot = self.current_slot + 1
                return reward



        else:
            # store the timestep demand value
            self.utility_demand_buffer_for_timesteps_in_a_slot.append(action_vector[0] + action_vector[2])

            reward = self.get_reward()
            self.cumulative_reward_in_a_day.append(reward)
            self.current_timestep = self.current_timestep + 1  # -------------------------------4)time step no
            return reward

    def reset(self):  # -----------------------------------------------------------Reset some variables after peak period ends at 12am
        self.old_BSOC = BSOC_max
        self.current_BSOC = 0 #'''figure a way in the step() function to chane the value of this variable at the appropriate time'''

        self.day_counter_for_only_monetary_cost = 0
        # need not update timestep, timeslot, Ymax since already taken care of

    def get_current_state(self):
        return (self.DCI_demand_for_next_timestep, self.DCI_demand_for_this_slot, self.DCI_demand_till_peak_end,
                self.current_BSOC, self.max_peak_demand_average, self.current_timestep, self.current_slot)

    def get_demand_for_new_timestep(self):
        '''returns the true demand of the DCI for the next time step by looking at the ground truth'''
        return None #returning None for now, use self.demand_timestep_iteration_count = 0 to extract the demand for next time step and return that

    def get_reward(self):
        '''returns the reward of the last action taken'''

        '''Calculate Ec and Dc values'''
        self.Ec = math.min(max_energy_charging_possible_per_timestep, self.It_actual, BSOC_max - self.old_BSOC )
        self.Dc = math.min(max_energy_discharging_possible_per_timestep, self.Ot_actual, self.old_BSOC - BSOC_min)

        '''To calculate the reward, we will use the true demand of the data center at each time step.'''
        self.true_total_DCI_demand_next_timestep = self.get_demand_for_new_timestep()
        alpha_term_one = self.action_array[2] + self.Ec
        alpha_term_two = (self.true_total_DCI_demand_next_timestep - (self.action_array[2] + self.Dc))
        self.consumption_from_utility_after_taking_agent_action_into_account = alpha_term_one + alpha_term_two

        '''Keep in buffer for later use'''
        self.utility_demand_buffer_for_timesteps_in_a_slot.append(self.consumption_from_utility_after_taking_agent_action_into_account)

        if self.current_timestep == 3:
            '''take new Ymax value into account while calculating the reward'''

            '''calculate new Ymax first'''
            assert len(self.utility_demand_buffer_for_timesteps_in_a_slot) == 3
            self.Ymax_new = statistics.mean(self.utility_demand_buffer_for_timesteps_in_a_slot)
            if self.Ymax_new > self.Ymax_old:
                self.is_new_Ymax_greater = True
            else:
                self.is_new_Ymax_greater = False

            self.reward_of_the_last_action_taken = - (self.consumption_from_utility_after_taking_agent_action_into_account + self.Dc * battery_degradation_cost_per_unit_energy + self.is_new_Ymax_greater * peak_demand_cost_per_unit * (self.Ymax_new - self.Ymax_old) )
            self.cumulative_reward_in_a_day.append(self.reward_of_the_last_action_taken)
            self.utility_demand_buffer_for_timesteps_in_a_slot = []

        else:
            '''Do not take the new cost function into account while calculating the reward.'''
            self.reward_of_the_last_action_taken = - (self.consumption_from_utility_after_taking_agent_action_into_account + self.Dc * battery_degradation_cost_per_unit_energy + self.is_new_Ymax_greater)
            self.cumulative_reward_in_a_day.append(self.reward_of_the_last_action_taken)



        '''Update the old BSOC value here'''
        self.old_BSOC = self.current_BSOC

        return self.reward_of_the_last_action_taken